from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .schema_adapter import parse_literal


def _split_ids(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _stage_type(pkg: SMSPackage, stage_id: str) -> str:
    row = pkg.stage_plan[pkg.stage_plan["stage_id"].astype(str) == str(stage_id)]
    if row.empty:
        return str(stage_id)
    return str(row.iloc[0].get("operation_type", stage_id)).upper()


def _transition_row(pkg: SMSPackage, parent_stage_id: str | None, stage_id: str) -> pd.Series:
    transitions = pkg.raw_tables.get("I_stage/stage_transition_record.csv", pd.DataFrame())
    if transitions.empty or parent_stage_id is None:
        return pd.Series(dtype=object)
    mask = (
        transitions.get("from_stage_id", pd.Series(dtype=str)).astype(str).eq(str(parent_stage_id))
        & transitions.get("to_stage_id", pd.Series(dtype=str)).astype(str).eq(str(stage_id))
    )
    return transitions.loc[mask].iloc[0] if mask.any() else pd.Series(dtype=object)


def _stage_input_row(pkg: SMSPackage, stage_id: str) -> pd.Series:
    inputs = pkg.raw_tables.get("I_stage/stage_input.csv", pd.DataFrame())
    if inputs.empty:
        return pd.Series(dtype=object)
    mask = inputs.get("stage_id", pd.Series(dtype=str)).astype(str).eq(str(stage_id))
    return inputs.loc[mask].iloc[0] if mask.any() else pd.Series(dtype=object)


def _activation_changes(
    transition: pd.Series,
    current_input: pd.Series,
    parent_input: pd.Series,
    *,
    item_field: str,
    activated_field: str,
    deactivated_field: str,
) -> dict[str, list[str]]:
    current = set(_split_ids(current_input.get(item_field)))
    previous = set(_split_ids(parent_input.get(item_field)))
    declared_activated = _split_ids(transition.get(activated_field))
    declared_deactivated = _split_ids(transition.get(deactivated_field))
    return {
        "activated": declared_activated or sorted(current - previous),
        "deactivated": declared_deactivated or sorted(previous - current),
        "active": sorted(current),
    }


def _package_part_state(pkg: SMSPackage, stage_id: str) -> dict[str, dict[str, Any]]:
    states = pkg.raw_tables.get("I_stage/part_stage_state.csv", pd.DataFrame())
    if states.empty:
        return {}
    rows = states[states.get("stage_id", pd.Series(dtype=str)).astype(str) == str(stage_id)]
    output: dict[str, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        part_id = str(row.get("part_id", ""))
        vector_id = str(row.get("structural_displacement_vector_id", ""))
        vector = pkg.matrices.get(vector_id)
        output[part_id] = {
            "part_state_id": row.get("part_stage_state_id", ""),
            "parent_part_state_id": row.get("parent_part_state_id_optional", ""),
            "pose_state": parse_literal(row.get("pose_state"), []),
            "structural_displacement": np.asarray(vector, dtype=float).tolist() if vector is not None else [],
            "data_source": "PACKAGE_PRECOMPUTED_STATE",
        }
    return output


def _runtime_interface_state(pkg: SMSPackage, stage_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cp = pkg.contact_points.reset_index(drop=True)
    if "interface_id" not in cp.columns:
        return {}
    lam = np.asarray(stage_result["solution"].lambda_n, dtype=float)
    gap = np.asarray(stage_result["solution"].gap_g, dtype=float)
    pressure = np.asarray(stage_result["pressure"], dtype=float)
    compression = np.asarray(stage_result["local_compression"], dtype=float)
    source_states = pkg.raw_tables.get("I_stage/interface_stage_state.csv", pd.DataFrame())
    output: dict[str, dict[str, Any]] = {}
    for interface_id, indices in cp.groupby("interface_id", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        source = source_states[
            source_states.get("stage_id", pd.Series(dtype=str)).astype(str).eq(str(stage_result["stage_id"]))
            & source_states.get("interface_id", pd.Series(dtype=str)).astype(str).eq(str(interface_id))
        ] if not source_states.empty else pd.DataFrame()
        source_row = source.iloc[0] if not source.empty else pd.Series(dtype=object)
        output[str(interface_id)] = {
            "interface_state_id": source_row.get("interface_stage_state_id", f"RUNTIME_IF_{stage_result['stage_id']}_{interface_id}"),
            "parent_interface_state_id": source_row.get("parent_interface_state_id_optional", ""),
            "contact_domain_ids": cp.iloc[idx].get("contact_domain_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
            "gap": gap[idx].tolist(),
            "lambda_n": lam[idx].tolist(),
            "pressure": pressure[idx].tolist(),
            "local_compression": compression[idx].tolist(),
            "active_local_indices": cp.iloc[idx].loc[lam[idx] > 1e-9, "local_index"].astype(int).tolist(),
            "data_source": "RUNTIME_GLOBAL_COUPLED_LCP",
        }
    return output


def _package_lock_state(pkg: SMSPackage, stage_id: str) -> dict[str, Any]:
    history = pkg.raw_tables.get("I_stage/connection_lock_history.csv", pd.DataFrame())
    if history.empty:
        return {}
    rows = history[history.get("join_stage_id", pd.Series(dtype=str)).astype(str) == str(stage_id)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "lock_history_id": row.get("lock_history_id", ""),
        "joint_ids": _split_ids(row.get("joint_ids")),
        "locked_reference_dofs": parse_literal(row.get("locked_reference_dofs"), []),
        "locked_reference_values": parse_literal(row.get("locked_reference_values"), []),
        "preload_actual": parse_literal(row.get("preload_actual"), {}),
        "joint_stiffness": parse_literal(row.get("joint_stiffness"), {}),
        "data_source": "PACKAGE_PRECOMPUTED_LOCK_HISTORY",
    }


@dataclass
class StageState:
    stage_state_id: str
    stage_id: str
    stage_type: str
    parent_stage_id: str | None
    parent_stage_state_id: str | None
    part_state: dict[str, dict[str, Any]]
    interface_state: dict[str, dict[str, Any]]
    contact_structural_response: np.ndarray
    contact_structural_response_increment: np.ndarray
    gap: np.ndarray
    gap_increment: np.ndarray
    lambda_n: np.ndarray
    pressure: np.ndarray
    local_compression: np.ndarray
    active_set: list[int]
    boundary_state: dict[str, list[str]]
    load_state: dict[str, list[str]]
    joint_lock_state: dict[str, Any]
    reference_state: str
    vector_layout_id: str
    data_source: str
    fallback_flag: bool
    input_sources: list[str]
    physical_residuals: dict[str, float]
    notes: list[str] = field(default_factory=list)
    sample_id: str = "SAMPLE_001"
    topology_id: str = ""
    topology_step_id: str = ""
    parent_topology_step_id: str | None = None
    assembly_cycle_id: str = ""
    operation_type: str = ""
    current_subassembly_id: str = ""
    active_part_ids: list[str] = field(default_factory=list)
    active_interface_ids: list[str] = field(default_factory=list)
    active_joint_ids: list[str] = field(default_factory=list)
    active_boundary_ids: list[str] = field(default_factory=list)
    active_load_ids: list[str] = field(default_factory=list)
    operator_set_id: str = ""
    operator_source: str = ""
    solve_status: str = "CONVERGED"
    fallback_reason: str = ""
    active_index_mask: list[bool] = field(default_factory=list)
    lambda_active: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    gap_active: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    connection_lock_history_ids: list[str] = field(default_factory=list)
    release_history_ids: list[str] = field(default_factory=list)
    mechanical_state_action: str = "SOLVE_GLOBAL_LCP"
    not_required_reason: str = ""
    quality_flag: str = "PASS"
    state_role: str = "PREDICTED"
    measurement_checkpoint_id: str = ""
    measurement_update_id: str = ""
    predicted_state_id: str = ""
    posterior_parent_state_id: str = ""
    effective_state_id: str = ""
    update_state_layout_id: str = ""
    state_correction_vector: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    state_covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    measurement_ids: list[str] = field(default_factory=list)
    measurement_update_status: str = "NOT_CONFIGURED"
    posterior_accepted: bool = False
    rollback_record_id: str = ""
    covariance_source: str = ""
    covariance_transfer_id: str = ""
    source_checkpoint_id: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "topology_id": self.topology_id,
            "topology_step_id": self.topology_step_id,
            "parent_topology_step_id": self.parent_topology_step_id or "",
            "stage_state_id": self.stage_state_id,
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "parent_stage_id": self.parent_stage_id or "",
            "parent_stage_state_id": self.parent_stage_state_id or "",
            "assembly_cycle_id": self.assembly_cycle_id,
            "operation_type": self.operation_type or self.stage_type,
            "current_subassembly_id": self.current_subassembly_id,
            "active_part_ids": ";".join(self.active_part_ids),
            "active_interface_ids": ";".join(self.active_interface_ids),
            "active_joint_ids": ";".join(self.active_joint_ids),
            "active_count": len(self.active_set),
            "contact_structural_response_norm_mm": float(np.linalg.norm(self.contact_structural_response)),
            "contact_structural_response_increment_norm_mm": float(np.linalg.norm(self.contact_structural_response_increment)),
            "response_type": "CONTACT_STRUCTURAL_FLEXIBILITY_RESPONSE_W_STRUCT_TIMES_LAMBDA",
            "gap_increment_norm_mm": float(np.linalg.norm(self.gap_increment)),
            "lambda_sum_N": float(np.sum(self.lambda_n)),
            "activated_boundary_ids": ";".join(self.boundary_state.get("activated", [])),
            "deactivated_boundary_ids": ";".join(self.boundary_state.get("deactivated", [])),
            "active_boundary_ids": ";".join(self.boundary_state.get("active", [])),
            "activated_load_ids": ";".join(self.load_state.get("activated", [])),
            "removed_load_ids": ";".join(self.load_state.get("removed", [])),
            "active_load_ids": ";".join(self.load_state.get("active", [])),
            "joint_lock_history_id": self.joint_lock_state.get("lock_history_id", ""),
            "joint_lock_source": self.joint_lock_state.get("data_source", ""),
            "vector_layout_id": self.vector_layout_id,
            "operator_set_id": self.operator_set_id,
            "operator_source": self.operator_source or self.data_source,
            "solve_status": self.solve_status,
            "mechanical_state_action": self.mechanical_state_action,
            "not_required_reason": self.not_required_reason,
            "data_source": self.data_source,
            "fallback_flag": self.fallback_flag,
            "fallback_reason": self.fallback_reason,
            "active_index_count": int(sum(self.active_index_mask)) if self.active_index_mask else len(self.lambda_n),
            "connection_lock_history_ids": ";".join(self.connection_lock_history_ids),
            "release_history_ids": ";".join(self.release_history_ids),
            "quality_flag": self.quality_flag,
            "state_role": self.state_role,
            "measurement_checkpoint_id": self.measurement_checkpoint_id,
            "measurement_update_id": self.measurement_update_id,
            "predicted_state_id": self.predicted_state_id,
            "posterior_parent_state_id": self.posterior_parent_state_id,
            "effective_state_id": self.effective_state_id,
            "update_state_layout_id": self.update_state_layout_id,
            "state_correction_vector": self.state_correction_vector.tolist(),
            "state_correction_norm": (
                float(np.linalg.norm(self.state_correction_vector))
                if self.state_correction_vector.size else 0.0
            ),
            "state_covariance": self.state_covariance.tolist(),
            "state_covariance_trace": (
                float(np.trace(self.state_covariance))
                if self.state_covariance.ndim == 2 and self.state_covariance.size else np.nan
            ),
            "measurement_ids": ";".join(self.measurement_ids),
            "measurement_update_status": self.measurement_update_status,
            "posterior_accepted": self.posterior_accepted,
            "rollback_record_id": self.rollback_record_id,
            "covariance_source": self.covariance_source,
            "covariance_transfer_id": self.covariance_transfer_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "input_sources": ";".join(self.input_sources),
            "complementarity_residual": self.physical_residuals.get("complementarity_residual", np.nan),
            "notes": "；".join(self.notes),
        }


def build_runtime_stage_state(
    pkg: SMSPackage,
    stage_result: dict[str, Any],
    parent: StageState | None,
) -> StageState:
    stage_id = str(stage_result["stage_id"])
    parent_stage_id = parent.stage_id if parent is not None else None
    transition = _transition_row(pkg, parent_stage_id, stage_id)
    current_input = _stage_input_row(pkg, stage_id)
    parent_input = _stage_input_row(pkg, parent_stage_id) if parent_stage_id is not None else pd.Series(dtype=object)
    stage_type = _stage_type(pkg, stage_id)
    contact_structural_response = np.asarray(stage_result["W_struct"], dtype=float) @ np.asarray(stage_result["solution"].lambda_n, dtype=float)
    gap = np.asarray(stage_result["solution"].gap_g, dtype=float)
    if parent is None:
        contact_structural_response_increment = contact_structural_response.copy()
        gap_increment = gap.copy()
    else:
        contact_structural_response_increment = contact_structural_response - parent.contact_structural_response
        gap_increment = gap - parent.gap

    lock_state: dict[str, Any] = {}
    notes = [
        "q/U_FREE 与 W_struct 来自数据包预计算输入；lambda/g/pressure/active_set 由软件运行时统一全局 LCP 重算。",
        "contact_structural_response 是接触坐标下 W_struct @ lambda，仅表示接触力导致的结构柔性响应，不是全阶阶段位移。",
    ]
    if stage_type == "JOIN":
        lock_state = _package_lock_state(pkg, stage_id)
        if not lock_state:
            joints = pkg.raw_tables.get("I0/joint_definition.csv", pd.DataFrame())
            joint_ids = joints[
                joints.get("activation_stage_id", pd.Series(dtype=str)).astype(str).eq(stage_id)
            ].get("joint_id", pd.Series(dtype=str)).astype(str).tolist() if not joints.empty else []
            lock_state = {
                "lock_history_id": f"RUNTIME_LOCK_{stage_id}",
                "joint_ids": joint_ids,
                "locked_reference_values": np.asarray(stage_result["local_compression"], dtype=float).tolist(),
                "data_source": "RUNTIME_CONTACT_REFERENCE_SIMPLIFIED",
            }
            notes.append("数据包未提供连接锁定历史；仅保存运行时接触压缩参考，未构造全阶连接自由度锁定。")
    elif stage_type == "RELEASE" and parent is not None:
        lock_state = dict(parent.joint_lock_state)
        if lock_state:
            lock_state["inherited_from_stage_state_id"] = parent.stage_state_id
            notes.append("RELEASE 显式继承 JOIN 连接锁定历史。")
    elif parent is not None and parent.joint_lock_state:
        lock_state = dict(parent.joint_lock_state)

    layout = pkg.raw_tables.get("matrices/vector_layout.csv", pd.DataFrame())
    layout_id = str(layout.iloc[0].get("vector_layout_id", "LEGACY_IMPLICIT_LAYOUT")) if not layout.empty else "LEGACY_IMPLICIT_LAYOUT"
    fallback = True
    return StageState(
        stage_state_id=f"RUNTIME_STAGE_STATE_{stage_id}",
        stage_id=stage_id,
        stage_type=stage_type,
        parent_stage_id=parent_stage_id,
        parent_stage_state_id=parent.stage_state_id if parent is not None else None,
        part_state=_package_part_state(pkg, stage_id),
        interface_state=_runtime_interface_state(pkg, stage_result),
        contact_structural_response=contact_structural_response,
        contact_structural_response_increment=contact_structural_response_increment,
        gap=gap,
        gap_increment=gap_increment,
        lambda_n=np.asarray(stage_result["solution"].lambda_n, dtype=float),
        pressure=np.asarray(stage_result["pressure"], dtype=float),
        local_compression=np.asarray(stage_result["local_compression"], dtype=float),
        active_set=list(stage_result["solution"].active_indices),
        boundary_state=_activation_changes(
            transition, current_input, parent_input,
            item_field="boundary_item_ids",
            activated_field="activated_boundary_ids",
            deactivated_field="deactivated_boundary_ids",
        ),
        load_state={
            **_activation_changes(
                transition, current_input, parent_input,
                item_field="load_item_ids",
                activated_field="activated_load_ids",
                deactivated_field="removed_load_ids",
            ),
            "removed": (
                _split_ids(transition.get("removed_load_ids"))
                or sorted(
                    set(_split_ids(parent_input.get("load_item_ids")))
                    - set(_split_ids(current_input.get("load_item_ids")))
                )
            ),
        },
        joint_lock_state=lock_state,
        reference_state=str(current_input.get(
            "reference_state_id",
            "PARENT_RUNTIME_STATE" if parent is not None else "INITIAL_AGGREGATE_STATE",
        )),
        vector_layout_id=layout_id,
        data_source="PACKAGE_PRECOMPUTED_Q_U_FREE_PLUS_RUNTIME_GLOBAL_LCP",
        fallback_flag=fallback,
        input_sources=["package:q/u_free/W_struct/Cn", "runtime:global_lcp"],
        physical_residuals={k: float(v) for k, v in stage_result["solution"].residuals.items()},
        notes=notes,
    )


def stage_transition_runtime_table(result: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for stage in result.values():
        state = stage.get("stage_state")
        if isinstance(state, StageState):
            rows.append(state.to_record())
    return pd.DataFrame(rows)
