from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "core").is_dir() and (parent / "scripts").is_dir():
            return parent
    return here.parents[3]


REPOSITORY_ROOT = _repository_root()
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.data_loader import load_package
from core.multi_part import vector_layout
from core.rolling_prediction import (  # noqa: E402
    BASELINE_POSTERIOR_ONLY,
    DIRECT_SMS_ADD,
    DIRECT_SMS_ALREADY_INCLUDED,
    EMPIRICAL_FRACTION_LABEL,
    run_posterior_virtual_sms_rolling_prediction,
    stable_package_file_hash,
    validate_rolling_prediction_package,
)
from core.topology_step import run_topology_steps  # noqa: E402
from scripts.stage_measurement_update_package_validator import (  # noqa: E402
    validate_package as validate_measurement_package,
)


REQUIRED_FILES = (
    "I_pred/rolling_prediction_plan.csv",
    "I_pred/virtual_sms_sample_set.csv",
    "I_pred/virtual_sms_sample.csv",
    "I_pred/virtual_sms_component.csv",
    "I_pred/virtual_sms_coefficients.csv",
    "I_pred/future_sms_assignment.csv",
    "I_pred/sms_operator_mapping.csv",
    "I_pred/future_process_scenario.csv",
    "I_pred/rolling_kcp_config.csv",
    "validation/rolling_prediction_oracle.csv",
    "validation/rolling_kcp_oracle.csv",
    "validation/rolling_expected_summary.json",
)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    checks.append({
        "name": name,
        "passed": bool(passed),
        "status": "PASS" if passed else "FAIL",
        "blocking": bool(blocking),
        "detail": detail,
    })


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative, encoding="utf-8-sig")


def _vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    return np.asarray(json.loads(str(value)), dtype=float)


def _allclose(left: object, right: object, atol: float = 5e-8) -> bool:
    a = _vector(left)
    b = _vector(right)
    return a.shape == b.shape and bool(
        np.allclose(a, b, rtol=1e-7, atol=atol, equal_nan=True)
    )


def _sample_result_map(result: Any) -> dict[tuple[str, str], Any]:
    rows = {}
    for item in result.sample_results:
        rows[("POSTERIOR", item.virtual_sms_sample_id)] = item
    for item in result.predicted_baseline_results:
        rows[("PREDICTED", item.virtual_sms_sample_id)] = item
    return rows


def _attachment_matches(
    root: Path, current_checks: list[dict[str, Any]]
) -> tuple[bool, str]:
    result_path = root / "validation" / "test_results.json"
    run_log_path = root / "validation" / "run_log.csv"
    gate_path = root / "validation" / "quality_gate.csv"
    if not all(path.exists() for path in (
        result_path, run_log_path, gate_path
    )):
        return False, "one or more validation attachments are missing"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    stored_core = [
        (item.get("name"), item.get("status"), item.get("detail"))
        for item in stored.get("checks", [])
        if item.get("name")
        != "validation attachments match current rolling package"
    ]
    current_core = [
        (item["name"], item["status"], item["detail"])
        for item in current_checks
    ]
    with run_log_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        run_log = list(csv.DictReader(stream))
    with gate_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        gates = list(csv.DictReader(stream))
    passed = (
        stored.get("package_id") == root.name
        and stored_core == current_core
        and bool(run_log)
        and run_log[0].get("input_package_id") == root.name
        and bool(gates)
        and gates[0].get("target_object_ids") == root.name
        and gates[0].get("pass_fail") == "PASS"
    )
    return passed, f"stored_status={stored.get('status')}"


def validate_package(
    root: Path,
    *,
    include_attachment_check: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    base = validate_measurement_package(
        root, include_attachment_check=False
    )
    checks = [dict(item) for item in base["checks"]]
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    _check(
        checks, "rolling prediction required files", not missing,
        f"missing={missing}",
    )
    if missing:
        blocking_fail_count = sum(
            item["status"] == "FAIL" and item["blocking"]
            for item in checks
        )
        return {
            **base,
            "package": root.name,
            "package_id": root.name,
            "status": "FAIL",
            "passed": sum(item["status"] == "PASS" for item in checks),
            "total": len(checks),
            "blocking_fail_count": blocking_fail_count,
            "checks": checks,
        }

    manifest = json.loads(
        (root / "package_manifest.json").read_text(encoding="utf-8-sig")
    )
    plans = _read_csv(root, "I_pred/rolling_prediction_plan.csv")
    sets = _read_csv(root, "I_pred/virtual_sms_sample_set.csv")
    samples = _read_csv(root, "I_pred/virtual_sms_sample.csv")
    components = _read_csv(root, "I_pred/virtual_sms_component.csv")
    coefficients = _read_csv(root, "I_pred/virtual_sms_coefficients.csv")
    assignments = _read_csv(root, "I_pred/future_sms_assignment.csv")
    mappings = _read_csv(root, "I_pred/sms_operator_mapping.csv")
    scenarios = _read_csv(root, "I_pred/future_process_scenario.csv")
    kcp_configs = _read_csv(root, "I_pred/rolling_kcp_config.csv")
    matrix_manifest = _read_csv(root, "matrices/matrix_manifest.csv")
    oracle = _read_csv(root, "validation/rolling_prediction_oracle.csv")
    kcp_oracle = _read_csv(root, "validation/rolling_kcp_oracle.csv")
    expected = json.loads(
        (root / "validation" / "rolling_expected_summary.json")
        .read_text(encoding="utf-8")
    )

    plan_ids = plans["rolling_plan_id"].astype(str)
    _check(
        checks, "rolling plan primary key and active row",
        len(plans) > 0 and plan_ids.is_unique
        and plans["active_flag"].astype(str).str.lower().eq("true").all(),
        f"plans={plan_ids.tolist()}",
    )
    declared = int(sets.iloc[0]["sample_count"])
    _check(
        checks, "explicit virtual SMS sample cardinality",
        declared == len(samples) == int(expected["sample_count"]),
        f"declared={declared}, rows={len(samples)}",
    )
    component_orders = sorted(components["component_order"].astype(int))
    sample_component_counts = (
        coefficients.groupby("virtual_sms_sample_id")["component_order"]
        .nunique().to_dict()
    )
    _check(
        checks, "component layout and coefficient completeness",
        component_orders == list(range(len(component_orders)))
        and all(value == len(component_orders)
                for value in sample_component_counts.values()),
        f"component_orders={component_orders}, counts={sample_component_counts}",
    )
    reference_samples = samples[
        samples["coefficient_source"].astype(str).eq("REFERENCE_SMS")
    ]
    reference_ids = set(reference_samples["virtual_sms_sample_id"].astype(str))
    reference_values = coefficients[
        coefficients["virtual_sms_sample_id"].astype(str).isin(reference_ids)
    ]["value"].astype(float).to_numpy()
    _check(
        checks, "single zero reference SMS sample",
        len(reference_samples) == 1
        and bool(np.allclose(reference_values, 0.0)),
        f"reference_ids={sorted(reference_ids)}, values={reference_values.tolist()}",
    )
    assignment_keys = assignments[
        ["virtual_sms_sample_id", "part_id"]
    ].astype(str).agg("|".join, axis=1)
    _check(
        checks, "sample assignment keys unique and complete",
        assignment_keys.is_unique
        and set(samples["virtual_sms_sample_id"].astype(str))
        == set(assignments["virtual_sms_sample_id"].astype(str)),
        f"assignment_rows={len(assignments)}",
    )
    map_manifest = matrix_manifest.set_index("matrix_id")
    mapping_shapes_ok = True
    mapping_detail = []
    for _, row in mappings.iterrows():
        matrix_id = str(row["matrix_id"])
        if matrix_id not in map_manifest.index:
            mapping_shapes_ok = False
            mapping_detail.append(f"{matrix_id}:missing")
            continue
        manifest_row = map_manifest.loc[matrix_id]
        expected_shape = tuple(
            int(value)
            for value in json.loads(str(manifest_row["shape"]))
        )
        mapping_detail.append(f"{matrix_id}:{expected_shape}")
        mapping_shapes_ok &= expected_shape[1] == len(component_orders)
    _check(
        checks, "G_SMS manifest coverage and column layout",
        mapping_shapes_ok and mappings["matrix_id"].astype(str).is_unique,
        ", ".join(mapping_detail),
    )
    truth_ok = (
        manifest.get("probability_interpretation_allowed") is False
        and manifest.get("engineering_claim_allowed") is False
        and sets["probability_interpretation_allowed"]
        .astype(str).str.lower().eq("false").all()
        and sets["engineering_claim_allowed"]
        .astype(str).str.lower().eq("false").all()
        and scenarios["probability_interpretation_allowed"]
        .astype(str).str.lower().eq("false").all()
    )
    _check(
        checks, "synthetic descriptive truth boundary", truth_ok,
        "probability=false, engineering=false, explicit deterministic samples",
    )
    direct_tokens = (
        kcp_configs["final_state_includes_direct_sms_geometry"]
        .astype(str).str.strip().str.lower()
    )
    direct_values = set(direct_tokens.map({
        "true": True, "false": False
    }).dropna().tolist())
    _check(
        checks, "direct SMS final-state boolean semantics",
        direct_tokens.isin(["true", "false"]).all()
        and len(direct_values) == 1,
        f"tokens={sorted(set(direct_tokens))}, "
        f"values={sorted(direct_values)}",
    )
    checkpoint = _read_csv(
        root, "I_meas/measurement_checkpoint.csv"
    )
    plan_row = plans.iloc[0]
    checkpoint_rows = checkpoint[
        checkpoint["checkpoint_id"].astype(str).eq(
            str(plan_row["source_checkpoint_id"])
        )
    ]
    checkpoint_closure = (
        len(checkpoint_rows) == 1
        and str(checkpoint_rows.iloc[0]["topology_id"])
        == str(plan_row["topology_id"])
        and str(checkpoint_rows.iloc[0]["topology_step_id"])
        == str(plan_row["source_topology_step_id"])
    )
    _check(
        checks, "plan checkpoint topology and step linkage",
        checkpoint_closure,
        (
            f"plan_topology={plan_row['topology_id']}, "
            f"checkpoint_topology="
            f"{checkpoint_rows.iloc[0]['topology_id'] if len(checkpoint_rows) else 'missing'}, "
            f"plan_step={plan_row['source_topology_step_id']}, "
            f"checkpoint_step="
            f"{checkpoint_rows.iloc[0]['topology_step_id'] if len(checkpoint_rows) else 'missing'}"
        ),
    )

    package = load_package(root)
    static_gates = validate_rolling_prediction_package(package)
    _check(
        checks, "production static rolling quality gates",
        not static_gates["status"].eq("FAIL").any(),
        f"pass={int(static_gates['status'].eq('PASS').sum())}/"
        f"{len(static_gates)}",
    )
    package_hash_before = stable_package_file_hash(root)
    topology = run_topology_steps(package)
    plan_id = str(plans.iloc[0]["rolling_plan_id"])
    result = run_posterior_virtual_sms_rolling_prediction(
        package, topology, plan_id
    )
    package_hash_after = stable_package_file_hash(root)
    _check(
        checks, "source package immutable during rolling run",
        package_hash_before == package_hash_after
        and result.trace["immutability_status"] == "PASS",
        f"before={package_hash_before}, after={package_hash_after}",
    )
    _check(
        checks, "accepted posterior is formal source",
        result.source_posterior_state.posterior_accepted
        and result.source_posterior_state.state_role == "POSTERIOR"
        and result.source_posterior_state.quality_flag == "PASS",
        f"state_id={result.source_posterior_state.stage_state_id}",
    )
    source_linkage = result.trace.get("source_linkage", {})
    _check(
        checks, "runtime posterior source linkage closure",
        source_linkage.get("source_linkage_status") == "PASS"
        and source_linkage.get("plan_topology_id")
        == source_linkage.get("checkpoint_topology_id")
        and source_linkage.get("plan_source_step_id")
        == source_linkage.get("checkpoint_topology_step_id")
        and source_linkage.get("actual_source_state_id")
        == result.source_posterior_state.stage_state_id
        and source_linkage.get(
            "measurement_update_posterior_state_id"
        ) == result.source_posterior_state.stage_state_id,
        json.dumps(
            source_linkage, ensure_ascii=False, sort_keys=True
        ),
    )
    _check(
        checks, "all explicit samples complete with isolated branches",
        len(result.sample_results) == declared
        and not result.sample_failures
        and len(result.predicted_baseline_results) == declared
        and not result.predicted_baseline_failures
        and len({
            item.final_state_id for item in result.sample_results
        }) == declared,
        f"success={len(result.sample_results)}, "
        f"failure={len(result.sample_failures)}, "
        f"baseline_success={len(result.predicted_baseline_results)}, "
        f"baseline_failure={len(result.predicted_baseline_failures)}",
    )
    application_rows = [
        trace
        for item in result.sample_results
        for trace in item.trace.get("sms_application_trace", [])
    ]
    _check(
        checks, "expected SMS application matrix complete",
        len(application_rows) == declared * 4
        and all(
            int(row.get("application_count", 0)) == 1
            for row in application_rows
        )
        and all(
            item.trace.get(
                "application_completeness", {}
            ).get("status") == "PASS"
            for item in result.sample_results
        ),
        f"applications={len(application_rows)}, expected={declared * 4}",
    )

    result_map = _sample_result_map(result)
    q_checks = []
    lcp_checks = []
    for _, expected_row in oracle.iterrows():
        role = str(expected_row["source_state_role"])
        sample_id = str(expected_row["virtual_sms_sample_id"])
        step_id = str(expected_row["topology_step_id"])
        actual = result_map[(role, sample_id)].step_results[step_id]
        q_ok = all((
            _allclose(actual["q_operator_base"],
                      expected_row["q_operator_base"]),
            _allclose(actual["q_posterior_state_correction"],
                      expected_row["q_posterior_state_correction"]),
            _allclose(actual["q_virtual_sms_correction"],
                      expected_row["q_virtual_sms_correction"]),
            _allclose(actual["q_future_process_correction"],
                      expected_row["q_future_process_correction"]),
            _allclose(actual["q"], expected_row["q_effective"]),
        ))
        q_checks.append(q_ok)
        active = _vector(expected_row["active_indices"]).astype(int)
        lcp_checks.append(
            int(actual["lcp_call_count"])
            == int(expected_row["lcp_call_count"]) == 1
            and _allclose(actual["lambda_full"][active],
                          expected_row["lambda_active"])
            and _allclose(actual["gap_full"][active],
                          expected_row["gap_active"])
            and float(actual["solution"].residuals[
                "complementarity_residual"
            ]) <= 1e-7
        )
    _check(
        checks, "independent oracle q decomposition",
        all(q_checks), f"matched={sum(q_checks)}/{len(q_checks)}",
    )
    _check(
        checks, "independent oracle global LCP solutions",
        all(lcp_checks),
        f"matched={sum(lcp_checks)}/{len(lcp_checks)}, one_call_per_step=true",
    )
    oracle_independent = (
        oracle["production_rolling_runner_used"]
        .astype(str).str.lower().eq("false").all()
        and oracle["production_topology_runner_used"]
        .astype(str).str.lower().eq("false").all()
        and oracle["production_virtual_sms_function_used"]
        .astype(str).str.lower().eq("false").all()
        and kcp_oracle["production_kcp_entry_used"]
        .astype(str).str.lower().eq("false").all()
    )
    _check(
        checks, "oracle implementation independence declarations",
        oracle_independent,
        "active-set enumeration and direct KCP formula are independent",
    )

    actual_kcp_frames = []
    for (role, sample_id), item in result_map.items():
        frame = item.kcp_prediction_result.copy()
        frame["source_state_role"] = role
        frame["virtual_sms_sample_id"] = sample_id
        actual_kcp_frames.append(frame)
    actual_kcp = pd.concat(actual_kcp_frames, ignore_index=True)
    merged = kcp_oracle.merge(
        actual_kcp[[
            "source_state_role", "virtual_sms_sample_id",
            "kcp_id", "predicted_value",
        ]],
        on=["source_state_role", "virtual_sms_sample_id", "kcp_id"],
        suffixes=("_oracle", "_actual"),
        how="left",
    )
    kcp_error = np.abs(
        merged["predicted_value_oracle"].astype(float)
        - merged["predicted_value_actual"].astype(float)
    )
    _check(
        checks, "independent KCP oracle agreement",
        len(merged) == len(kcp_oracle)
        and not merged["predicted_value_actual"].isna().any()
        and float(kcp_error.max()) <= 5e-8,
        f"rows={len(merged)}, max_abs_error={float(kcp_error.max()):.3e}",
    )
    reference_results = [
        item for item in result.sample_results
        if item.virtual_sms_sample_id in reference_ids
    ]
    reference_zero = all(
        np.allclose(
            step["q_virtual_sms_correction"], 0.0, atol=1e-12
        )
        for item in reference_results
        for step in item.step_results.values()
    )
    _check(
        checks, "reference sample has zero virtual SMS correction",
        reference_zero, f"reference_count={len(reference_results)}",
    )
    distinct_kcp = (
        result.kcp_predictions.groupby("virtual_sms_sample_id")[
            "predicted_value"
        ].apply(lambda values: tuple(np.round(values, 12))).nunique()
    )
    _check(
        checks, "non-reference SMS sensitivity is observable",
        distinct_kcp > 1, f"distinct_kcp_vectors={distinct_kcp}",
    )
    all_steps = [
        step for item in result.sample_results
        for step in item.step_results.values()
    ]
    layout = vector_layout(package).set_index("object_id")
    cross_norms = []
    for step in all_steps:
        active_interfaces = [
            interface_id
            for interface_id in step["active_interface_ids"]
            if interface_id in layout.index
        ]
        for left_id, right_id in combinations(active_interfaces, 2):
            left = layout.loc[left_id]
            right = layout.loc[right_id]
            left_indices = np.arange(
                int(left["start_index"]), int(left["end_index"]) + 1
            )
            right_indices = np.arange(
                int(right["start_index"]), int(right["end_index"]) + 1
            )
            cross_norms.append(float(np.linalg.norm(
                step["W_struct"][np.ix_(left_indices, right_indices)]
            )))
    cross_block_nonzero = bool(cross_norms) and max(cross_norms) > 1e-12
    _check(
        checks, "coupled W_struct cross-interface blocks preserved",
        cross_block_nonzero,
        f"checked_steps={len(all_steps)}, max_norm="
        f"{max(cross_norms, default=0.0):.6e}",
    )
    _check(
        checks, "KCP contribution ledger and double-count gates",
        all(
            item.kcp_prediction_result["double_count_status"]
            .astype(str).eq("PASS").all()
            and not item.contribution_ledger.empty
            for item in result.sample_results
        ),
        f"samples={len(result.sample_results)}",
    )
    expected_included = next(iter(direct_values), None)
    expected_action = (
        DIRECT_SMS_ALREADY_INCLUDED
        if expected_included else DIRECT_SMS_ADD
    )
    direct_semantics_ok = all(
        isinstance(expected_included, bool)
        and item.trace.get("ledger", {}).get(
            "final_state_includes_direct_sms_geometry"
        ) is expected_included
        and item.trace.get("ledger", {}).get(
            "direct_sms_aggregation_action"
        ) == expected_action
        and item.kcp_prediction_result[
            "final_state_includes_direct_sms_geometry"
        ].map(lambda value: bool(value) is expected_included).all()
        and item.kcp_prediction_result[
            "direct_sms_aggregation_action"
        ].astype(str).eq(expected_action).all()
        for item in result.sample_results
    )
    _check(
        checks, "runtime direct SMS aggregation semantics",
        direct_semantics_ok
        and result.status_summary[
            "final_state_includes_direct_sms_geometry"
        ] is expected_included
        and result.status_summary[
            "direct_sms_aggregation_action"
        ] == expected_action,
        f"included={expected_included}, action={expected_action}",
    )
    direct_ledger_ok = all(
        (
            "FUTURE_SMS_DIRECT_GEOMETRY"
            not in set(item.contribution_ledger["source_class"].astype(str))
        )
        if expected_included
        else (
            "FUTURE_SMS_DIRECT_GEOMETRY"
            in set(item.contribution_ledger["source_class"].astype(str))
        )
        for item in result.sample_results
    )
    _check(
        checks, "direct SMS contribution ledger action",
        direct_ledger_ok,
        f"expected_action={expected_action}",
    )
    baseline_policy = str(plan_row["baseline_comparison_policy"])
    baseline_expected = (
        baseline_policy != BASELINE_POSTERIOR_ONLY
    )
    _check(
        checks, "predicted-cutoff comparison is complete",
        (
            len(result.baseline_comparison)
            == len(result.kcp_predictions)
            and not result.baseline_comparison[
                "posterior_minus_predicted"
            ].isna().any()
            and result.status_summary["baseline_quality_status"]
            == "PASS"
        )
        if baseline_expected
        else (
            result.baseline_comparison.empty
            and len(result.baseline_comparison.columns) > 0
            and result.status_summary["baseline_attempt_count"] == 0
            and result.status_summary["baseline_success_count"] == 0
            and result.status_summary["baseline_failure_count"] == 0
            and result.status_summary["baseline_quality_status"]
            == "NOT_APPLICABLE"
        ),
        f"policy={baseline_policy}, rows={len(result.baseline_comparison)}",
    )
    summary_ok = True
    summary_detail = []
    actual_summary = result.descriptive_summary.kcp_summary.set_index("kcp_id")
    summary_columns = {
        "count": "count",
        "mean": "descriptive_mean",
        "std": "descriptive_std",
        "min": "empirical_min",
        "max": "empirical_max",
        "p05": "empirical_p05",
        "p50": "empirical_p50",
        "p95": "empirical_p95",
    }
    for kcp_id, expected_values in expected["kcp_summary"].items():
        row = actual_summary.loc[kcp_id]
        for field, expected_value in expected_values.items():
            summary_ok &= bool(np.isclose(
                float(row[summary_columns[field]]), float(expected_value),
                rtol=1e-7, atol=5e-8,
            ))
        summary_detail.append(kcp_id)
    summary_ok &= (
        result.descriptive_summary.probability_interpretation_allowed is False
        and result.descriptive_summary.engineering_claim_allowed is False
        and result.descriptive_summary.kcp_summary[
            "fraction_semantics"
        ].astype(str).eq(EMPIRICAL_FRACTION_LABEL).all()
    )
    _check(
        checks, "descriptive summary oracle and probability labels",
        summary_ok, f"kcp_ids={summary_detail}",
    )
    _check(
        checks, "runtime rolling quality gates",
        result.quality_status == "PASS"
        and result.trace["quality_status"] == "PASS"
        and result.descriptive_summary.quality_status == "PASS"
        and result.status_summary["run_quality_status"] == "PASS"
        and result.status_summary["baseline_attempt_count"] == declared
        and result.status_summary["baseline_success_count"] == declared
        and result.status_summary["baseline_failure_count"] == 0
        and result.status_summary["baseline_quality_status"]
        == expected["baseline_quality_status"]
        and result.status_summary["source_linkage_status"]
        == expected["source_linkage_status"]
        and result.trace["checkpoint_topology_step_id"]
        == expected["source_topology_step_id"]
        and result.source_posterior_state.source_checkpoint_id
        == expected["source_checkpoint_id"]
        and result.status_summary[
            "final_state_includes_direct_sms_geometry"
        ] is expected[
            "final_state_includes_direct_sms_geometry"
        ]
        and result.status_summary["direct_sms_aggregation_action"]
        == expected["direct_sms_aggregation_action"]
        and result.status_summary["double_count_fail_count"] == 0
        and result.status_summary["application_fail_count"] == 0
        and result.status_summary["immutability_fail_count"] == 0
        and not result.quality_gates["status"].eq("FAIL").any(),
        f"pass={int(result.quality_gates['status'].eq('PASS').sum())}/"
        f"{len(result.quality_gates)}",
    )

    if include_attachment_check:
        attachment_ok, attachment_detail = _attachment_matches(root, checks)
        _check(
            checks,
            "validation attachments match current rolling package",
            attachment_ok,
            attachment_detail,
        )
    blocking_fail_count = sum(
        item["status"] == "FAIL" and item["blocking"] for item in checks
    )
    return {
        "package": root.name,
        "package_id": root.name,
        "status": "PASS" if blocking_fail_count == 0 else "FAIL",
        "passed": sum(item["status"] == "PASS" for item in checks),
        "total": len(checks),
        "blocking_fail_count": blocking_fail_count,
        "matrix_manifest_count": base["matrix_manifest_count"],
        "npz_key_count": base["npz_key_count"],
        "checks": checks,
        "matrix_digests": base["matrix_digests"],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['package_id']} validation",
        "",
        f"- Status: {report['status']}",
        f"- Passed: {report['passed']}/{report['total']}",
        f"- Checks: {report['passed']}/{report['total']} PASS",
        f"- Blocking failures: {report['blocking_fail_count']}",
        f"- MatrixManifest rows: {report['matrix_manifest_count']}",
        f"- NPZ keys: {report['npz_key_count']}",
        "",
        "| Check | Status | Blocking | Detail |",
        "|---|---:|---:|---|",
    ]
    for item in report["checks"]:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{str(item['blocking']).lower()} | {detail} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--skip-attachment-check", action="store_true"
    )
    args = parser.parse_args()
    report = validate_package(
        args.package,
        include_attachment_check=not args.skip_attachment_check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
