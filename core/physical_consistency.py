from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .data_loader import SMSPackage


@dataclass
class PhysicalConsistencySettings:
    """Tolerances for the first-level physical consistency panel.

    The panel is intentionally lightweight: it checks whether the LCP solution
    itself is physically admissible before judging engineering accuracy.  It does
    not replace high-fidelity FE validation or full D_valid governance.
    """

    gap_tolerance_mm: float = 1e-8
    force_tolerance_N: float = 1e-8
    complementarity_tolerance_Nmm: float = 1e-9
    equilibrium_tolerance_mm: float = 1e-10

    @classmethod
    def from_eps(cls, eps: float) -> "PhysicalConsistencySettings":
        e = float(eps) if eps is not None and np.isfinite(eps) else 1e-9
        return cls(
            gap_tolerance_mm=max(1e-8, e),
            force_tolerance_N=max(1e-8, e),
            complementarity_tolerance_Nmm=max(1e-9, e),
            equilibrium_tolerance_mm=max(1e-10, e),
        )


def _finite_status(values: np.ndarray) -> tuple[str, str]:
    if values.size == 0:
        return "FAIL", "空向量，无法判断。"
    if not np.all(np.isfinite(values)):
        return "FAIL", "存在 NaN 或 Inf。"
    return "PASS", "数值有限。"


def _status_by_violation(value: float, tol: float, warn_factor: float = 10.0) -> str:
    if not np.isfinite(value):
        return "FAIL"
    if value <= tol:
        return "PASS"
    if value <= warn_factor * tol:
        return "WARN"
    return "FAIL"


def _row(stage_id: str, check_item: str, value: float | int | str, tolerance: float | str, status: str, detail: str) -> dict:
    return {
        "stage_id": stage_id,
        "check_item": check_item,
        "value": value,
        "tolerance": tolerance,
        "status": status,
        "detail": detail,
    }


def _combine_statuses(statuses, *, default: str = "NOT_AVAILABLE") -> str:
    values = [str(s).upper() for s in statuses if str(s).upper() in {"PASS", "WARN", "FAIL"}]
    if not values:
        return default
    if "FAIL" in values:
        return "FAIL"
    if "WARN" in values:
        return "WARN"
    return "PASS"


def _contact_state(active_count: int, total_count: int) -> tuple[str, str, str]:
    if total_count <= 0:
        return "NO_POINTS", "WARN", "没有候选接触点，无法判断接触状态。"
    if active_count == 0:
        return "NO_CONTACT", "WARN", f"0/{total_count} 个主动点；数学上允许，但需确认该阶段是否应完全脱开。"
    if active_count == total_count:
        return "ALL_CONTACT", "WARN", f"{active_count}/{total_count} 个主动点；数学上允许，但需确认是否存在接触域过小或闭合量偏大。"
    return "MIXED_CONTACT", "PASS", f"{active_count}/{total_count} 个主动点，存在接触与开放区。"


def _stage_indices_from_solution(sol, n: int) -> list[int]:
    try:
        return [int(i) for i in sol.active_indices]
    except Exception:
        lam = np.asarray(sol.lambda_n, dtype=float).reshape(-1)
        return [int(i) for i in np.where(lam > 0.0)[0] if i < n]


def stage_physical_consistency(
    result: dict[str, dict],
    settings: PhysicalConsistencySettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stage-level summary and detailed LCP admissibility checks."""
    settings = settings or PhysicalConsistencySettings()
    stage_rows: list[dict] = []
    detail_rows: list[dict] = []
    result_by_step = {
        str(res.get("topology_step_id", key)): res for key, res in result.items()
    }

    for sid, res in result.items():
        sol = res["solution"]
        if str(res.get("solve_status", "")).upper() == "NOT_REQUIRED":
            state = res.get("stage_state")
            action = str(
                getattr(state, "mechanical_state_action", "")
                or res.get("mechanical_state_action", "")
            )
            reason = str(
                getattr(state, "not_required_reason", "")
                or res.get("not_required_reason", "")
            )
            parent_step_id = str(getattr(state, "parent_topology_step_id", "") or "")
            inherited_ok = True
            if action == "INHERIT_PARENT_UNCHANGED":
                parent_result = result_by_step.get(parent_step_id)
                inherited_ok = parent_result is not None and all(
                    np.allclose(
                        np.asarray(res[name], dtype=float),
                        np.asarray(parent_result[name], dtype=float),
                        equal_nan=True,
                    )
                    for name in ("lambda_full", "gap_full", "pressure", "local_compression")
                )
            status = (
                "PASS"
                if action in {"INITIALIZE_EMPTY", "INHERIT_PARENT_UNCHANGED"} and inherited_ok
                else "FAIL"
            )
            detail_rows.append(_row(
                sid, "mechanical_solve", "NOT_REQUIRED", "NOT_REQUIRED", status,
                f"solve_status=NOT_REQUIRED; reason={reason}; mechanical_state_action={action}; "
                f"inherited_unchanged={inherited_ok}",
            ))
            mask = np.asarray(res.get("active_index_mask", []), dtype=bool)
            lam = np.asarray(res.get("lambda_full", sol.lambda_n), dtype=float)
            active_count = len(getattr(state, "active_set", sol.active_indices))
            stage_rows.append({
                "stage_id": sid, "overall_status": status, "physics_status": status,
                "contact_state": "NOT_REQUIRED", "contact_state_status": status,
                "convergence_status": "NOT_REQUIRED", "active_count": active_count,
                "active_ratio": active_count / int(mask.sum()) if mask.any() else 0.0,
                "gap_min_mm": np.nan, "gap_violation_mm": 0.0, "lambda_min_N": 0.0,
                "force_violation_N": 0.0, "lambda_sum_N": float(np.nansum(lam)),
                "lambda_max_N": float(np.nanmax(lam)) if lam.size else 0.0,
                "pressure_max_MPa": float(np.nanmax(res.get("pressure", [0.0])))
                if np.isfinite(np.asarray(res.get("pressure", [0.0]), dtype=float)).any() else 0.0,
                "complementarity_residual_Nmm": 0.0,
                "equilibrium_residual_mm": 0.0,
                "mechanical_state_action": action,
                "not_required_reason": reason,
            })
            continue
        if "lambda_active" in res and "gap_active" in res:
            lam = np.asarray(res["lambda_active"], dtype=float).reshape(-1)
            gap = np.asarray(res["gap_active"], dtype=float).reshape(-1)
            q = np.asarray(res.get("q_active", np.zeros_like(gap)), dtype=float).reshape(-1)
            W = np.asarray(res.get("W_active", np.eye(gap.size)), dtype=float)
            full_pressure = np.asarray(res.get("pressure", np.zeros_like(sol.gap_g)), dtype=float).reshape(-1)
            mask = np.asarray(res.get("active_index_mask", np.ones(full_pressure.size, dtype=bool)), dtype=bool)
            pressure = full_pressure[mask]
        else:
            lam = np.asarray(sol.lambda_n, dtype=float).reshape(-1)
            gap = np.asarray(sol.gap_g, dtype=float).reshape(-1)
            q = np.asarray(res.get("q", np.zeros_like(gap)), dtype=float).reshape(-1)
            W = np.asarray(res.get("W_total", np.eye(gap.size)), dtype=float)
            pressure = np.asarray(res.get("pressure", np.zeros_like(gap)), dtype=float).reshape(-1)

        n = int(gap.size)
        active_indices = [int(i) for i in np.where(lam > 0.0)[0]]
        active_count = len(active_indices)
        active_ratio = active_count / n if n else np.nan

        finite_status, finite_detail = _finite_status(np.concatenate([gap, lam, q, W.reshape(-1)]))
        detail_rows.append(_row(sid, "finite_numbers", finite_status == "PASS", "all finite", finite_status, finite_detail))

        gap_min = float(np.min(gap)) if gap.size else np.nan
        lambda_min = float(np.min(lam)) if lam.size else np.nan
        gap_violation = float(max(0.0, -gap_min)) if np.isfinite(gap_min) else np.nan
        force_violation = float(max(0.0, -lambda_min)) if np.isfinite(lambda_min) else np.nan
        comp_res = float(np.max(np.abs(gap * lam))) if gap.size and lam.size else np.nan
        # For an extended LCP, the contact sub-vector also receives the coupling
        # contribution from locator/clamp/joint reactions. Reconstruct equilibrium
        # with the full extended system; checking q_contact + W_contact*lambda_contact
        # alone would incorrectly report a residual whenever extra reactions are active.
        ext = res.get("extended_lcp")
        if isinstance(ext, dict) and ext.get("enabled", False) and "solution" in ext:
            ext_sol = ext["solution"]
            q_eq = np.asarray(ext.get("q_ext", []), dtype=float).reshape(-1)
            W_eq = np.asarray(ext.get("W_ext", []), dtype=float)
            lam_eq = np.asarray(ext_sol.lambda_n, dtype=float).reshape(-1)
            gap_eq = np.asarray(ext_sol.gap_g, dtype=float).reshape(-1)
            n_eq = gap_eq.size
            eq_vec = gap_eq - (q_eq + W_eq @ lam_eq) if W_eq.shape == (n_eq, n_eq) else np.full(n_eq, np.nan)
            equilibrium_detail = "扩展LCP：检查 g_ext 是否等于 q_ext + W_ext lambda_ext。"
        else:
            eq_vec = gap - (q + W @ lam) if W.shape == (n, n) else np.full(n, np.nan)
            equilibrium_detail = "检查 g 是否等于 q + W lambda。"
        eq_res = float(np.linalg.norm(eq_vec)) if eq_vec.size and np.all(np.isfinite(eq_vec)) else np.nan
        lambda_sum = float(np.sum(lam)) if lam.size else np.nan
        lambda_max = float(np.max(lam)) if lam.size else np.nan
        pressure_max = float(np.max(pressure)) if pressure.size else np.nan

        gap_status = _status_by_violation(gap_violation, settings.gap_tolerance_mm)
        force_status = _status_by_violation(force_violation, settings.force_tolerance_N)
        comp_status = _status_by_violation(comp_res, settings.complementarity_tolerance_Nmm, warn_factor=100.0)
        eq_status = _status_by_violation(eq_res, settings.equilibrium_tolerance_mm, warn_factor=100.0)
        conv_status = "PASS" if str(sol.convergence_status).upper() == "CONVERGED" else "FAIL"
        contact_state, active_status, contact_detail = _contact_state(active_count, n)

        detail_rows.extend([
            _row(sid, "solver_convergence", sol.convergence_status, "CONVERGED", conv_status, "主动集/LCP求解器收敛状态。"),
            _row(sid, "gap_feasibility", gap_violation, settings.gap_tolerance_mm, gap_status, "要求平衡后间隙 g >= 0；小负值可视为数值误差。"),
            _row(sid, "force_feasibility", force_violation, settings.force_tolerance_N, force_status, "要求法向接触力 lambda >= 0；小负值可视为数值误差。"),
            _row(sid, "complementarity", comp_res, settings.complementarity_tolerance_Nmm, comp_status, "要求 g * lambda ≈ 0，即有间隙则无接触力、有接触力则间隙接近 0。"),
            _row(sid, "equilibrium_reconstruction", eq_res, settings.equilibrium_tolerance_mm, eq_status, equilibrium_detail),
            _row(sid, "active_contact_count", active_count, "0 < active_count < n is informative", active_status, contact_detail),
        ])

        physics_status = _combine_statuses([finite_status, conv_status, gap_status, force_status, comp_status, eq_status])
        overall = _combine_statuses([physics_status, active_status])
        stage_rows.append({
            "stage_id": sid,
            "overall_status": overall,
            "physics_status": physics_status,
            "contact_state": contact_state,
            "contact_state_status": active_status,
            "convergence_status": sol.convergence_status,
            "active_count": active_count,
            "active_ratio": active_ratio,
            "gap_min_mm": gap_min,
            "gap_violation_mm": gap_violation,
            "lambda_min_N": lambda_min,
            "force_violation_N": force_violation,
            "lambda_sum_N": lambda_sum,
            "lambda_max_N": lambda_max,
            "pressure_max_MPa": pressure_max,
            "complementarity_residual_Nmm": comp_res,
            "equilibrium_residual_mm": eq_res,
            "iteration_count": sol.iteration_count,
        })

    return pd.DataFrame(stage_rows), pd.DataFrame(detail_rows)


def kcp_anomaly_checks(kcp: pd.DataFrame, validation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lightweight KCP anomaly hints: finite value, tolerance band, validation mismatch."""
    if kcp is None or kcp.empty:
        return pd.DataFrame(columns=["kcp_id", "feature_type", "stage_id", "predicted_value", "status", "detail"])

    val = validation.copy() if isinstance(validation, pd.DataFrame) and not validation.empty else pd.DataFrame()
    val_by_id = val.set_index("kcp_id") if "kcp_id" in val.columns else pd.DataFrame()

    rows: list[dict] = []
    for _, r in kcp.iterrows():
        kcp_id = str(r.get("kcp_id", r.get("feature_id", "KCP_UNKNOWN")))
        ftype = str(r.get("feature_type", ""))
        sid = str(r.get("stage_id", ""))
        unit = str(r.get("unit", ""))
        pred = pd.to_numeric(pd.Series([r.get("predicted_value", np.nan)]), errors="coerce").iloc[0]
        tolerance_status = "PASS"
        tolerance_details: list[str] = []
        if not np.isfinite(pred):
            tolerance_status = "FAIL"
            tolerance_details.append("预测值为 NaN/Inf。")
        else:
            tolerance_details.append("预测值有限。")

        lower = pd.to_numeric(pd.Series([r.get("lower_tol", np.nan)]), errors="coerce").iloc[0]
        upper = pd.to_numeric(pd.Series([r.get("upper_tol", np.nan)]), errors="coerce").iloc[0]
        nominal = pd.to_numeric(pd.Series([r.get("nominal_value", np.nan)]), errors="coerce").iloc[0]
        has_band = np.isfinite(lower) and np.isfinite(upper)
        if np.isfinite(pred) and has_band:
            lo = lower
            hi = upper
            # Some input tables store tolerance band directly; others store nominal +/- tol.
            # Prefer direct band when lower <= upper.  If nominal is finite and the band appears
            # to be tolerance offsets around nominal, also check nominal+offset for transparency.
            slack = 1e-8
            in_direct_band = (lo - slack) <= pred <= (hi + slack) if lo <= hi else True
            in_nominal_band = True
            if np.isfinite(nominal):
                nlo = nominal + lower
                nhi = nominal + upper
                in_nominal_band = (nlo - slack) <= pred <= (nhi + slack) if nlo <= nhi else True
            in_band = in_direct_band or in_nominal_band
            if not in_band and tolerance_status != "FAIL":
                tolerance_status = "WARN"
            tolerance_details.append("在容差带内。" if in_band else f"超出容差提示：direct=[{lower}, {upper}], nominal={nominal}。")
        else:
            tolerance_details.append("未提供完整容差带，仅做有限值检查。")

        validation_status = "NOT_AVAILABLE"
        validation_sigma = np.nan
        validation_detail = "没有可用的独立验证值。"
        if not val_by_id.empty and kcp_id in val_by_id.index:
            vr = val_by_id.loc[kcp_id]
            if isinstance(vr, pd.DataFrame):
                vr = vr.iloc[0]
            abs_error = pd.to_numeric(pd.Series([vr.get("abs_error", np.nan)]), errors="coerce").iloc[0]
            uncertainty = pd.to_numeric(pd.Series([vr.get("uncertainty", np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(abs_error) and np.isfinite(uncertainty) and uncertainty > 0:
                ratio = abs_error / uncertainty
                validation_sigma = float(ratio)
                if ratio > 5:
                    validation_status = "FAIL"
                    validation_detail = f"与验证值偏差 {ratio:.2f}σ，超过 5σ。"
                elif ratio > 2:
                    validation_status = "WARN"
                    validation_detail = f"与验证值偏差 {ratio:.2f}σ，超过 2σ。"
                else:
                    validation_status = "PASS"
                    validation_detail = f"与验证值偏差 {ratio:.2f}σ。"

        status = _combine_statuses([tolerance_status, validation_status], default=tolerance_status)

        rows.append({
            "kcp_id": kcp_id,
            "feature_type": ftype,
            "stage_id": sid,
            "predicted_value": pred,
            "unit": unit,
            "lower_tol": lower,
            "upper_tol": upper,
            "tolerance_status": tolerance_status,
            "validation_status": validation_status,
            "validation_sigma": validation_sigma,
            "status": status,
            "tolerance_detail": " ".join(tolerance_details),
            "validation_detail": validation_detail,
            "detail": " ".join(tolerance_details + [validation_detail]),
        })
    return pd.DataFrame(rows)


def physical_consistency_report(
    pkg: SMSPackage,
    result: dict[str, dict],
    kcp: pd.DataFrame,
    validation: pd.DataFrame,
    eps: float = 1e-9,
    validation_comparable: bool = True,
    validation_context: str = "当前运行配置与验证基准一致。",
) -> dict[str, pd.DataFrame | dict]:
    settings = PhysicalConsistencySettings.from_eps(eps)
    stage_summary, check_details = stage_physical_consistency(result, settings)
    kcp_checks = kcp_anomaly_checks(kcp, validation)

    physics_status = _combine_statuses(stage_summary.get("physics_status", pd.Series(dtype=str)).tolist())
    contact_status = _combine_statuses(stage_summary.get("contact_state_status", pd.Series(dtype=str)).tolist())
    kcp_tolerance_status = _combine_statuses(kcp_checks.get("tolerance_status", pd.Series(dtype=str)).tolist())
    raw_validation_status = _combine_statuses(kcp_checks.get("validation_status", pd.Series(dtype=str)).tolist())
    kcp_validation_status = raw_validation_status if validation_comparable else "REFERENCE_ONLY"
    decision_inputs = [physics_status, contact_status, kcp_tolerance_status]
    if validation_comparable:
        decision_inputs.append(raw_validation_status)
    overall_status = _combine_statuses(decision_inputs)

    overall = {
        "overall_status": overall_status,
        "physics_status": physics_status,
        "contact_status": contact_status,
        "kcp_tolerance_status": kcp_tolerance_status,
        "kcp_validation_status": kcp_validation_status,
        "raw_kcp_validation_status": raw_validation_status,
        "validation_comparable": bool(validation_comparable),
        "validation_context": validation_context,
        "stage_pass_count": int((stage_summary.get("overall_status", pd.Series(dtype=str)) == "PASS").sum()) if not stage_summary.empty else 0,
        "stage_warn_count": int((stage_summary.get("overall_status", pd.Series(dtype=str)) == "WARN").sum()) if not stage_summary.empty else 0,
        "stage_fail_count": int((stage_summary.get("overall_status", pd.Series(dtype=str)) == "FAIL").sum()) if not stage_summary.empty else 0,
        "kcp_warn_count": int((kcp_checks.get("status", pd.Series(dtype=str)) == "WARN").sum()) if not kcp_checks.empty else 0,
        "kcp_fail_count": int((kcp_checks.get("status", pd.Series(dtype=str)) == "FAIL").sum()) if not kcp_checks.empty else 0,
        "min_gap_mm": float(stage_summary["gap_min_mm"].min()) if not stage_summary.empty else np.nan,
        "min_lambda_N": float(stage_summary["lambda_min_N"].min()) if not stage_summary.empty else np.nan,
        "max_complementarity_residual_Nmm": float(stage_summary["complementarity_residual_Nmm"].max()) if not stage_summary.empty else np.nan,
        "max_gap_violation_mm": float(stage_summary["gap_violation_mm"].max()) if not stage_summary.empty else np.nan,
        "max_force_violation_N": float(stage_summary["force_violation_N"].max()) if not stage_summary.empty else np.nan,
    }

    return {
        "settings": settings,
        "overall": overall,
        "stage_summary": stage_summary,
        "check_details": check_details,
        "kcp_anomalies": kcp_checks,
    }
