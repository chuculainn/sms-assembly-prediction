from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts" / "topology_step_package_validator.py").exists():
            return parent
    raise RuntimeError("cannot locate repository root")


ROOT = _repository_root()
sys.path.insert(0, str(ROOT / "scripts"))

from topology_step_package_validator import (  # noqa: E402
    validate_package as validate_topology_package,
)


REQUIRED_FILES = {
    "I_meas/measurement_checkpoint.csv",
    "I_meas/measurement_observation_map.csv",
    "I_meas/measurement_update_config.csv",
    "I_meas/measurement_record.csv",
    "I_stage/state_update_basis.csv",
    "I_stage/stage_covariance_transfer.csv",
    "validation/stage_measurement_update_oracle.csv",
    "validation/measurement_update_expected_summary.json",
}

REQUIRED_OBJECTS = {
    "MeasurementCheckpointSpec",
    "StateUpdateBasis",
    "MeasurementObservationSpec",
    "MeasurementUpdateConfig",
    "UpdateDecisionRecord",
    "ReSolveRequirement",
    "UpdateRollbackRecord",
    "StageMeasurementUpdateResult",
    "PredictedStageState",
    "PosteriorStageState",
    "StageCovarianceTransfer",
    "StageMeasurementUpdateOracle",
    "MeasurementInnovationReport",
    "PosteriorStateReport",
}


def read_rows(root: Path, relative: str) -> list[dict[str, str]]:
    path = root / relative
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def validate_package(
    root: Path,
    *,
    include_attachment_check: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    base = validate_topology_package(root, include_attachment_check=False)
    checks = [dict(item) for item in base["checks"]]
    manifest = json.loads(
        (root / "package_manifest.json").read_text(encoding="utf-8")
    )
    missing_files = sorted(
        relative for relative in REQUIRED_FILES
        if not (root / relative).exists()
    )
    _check(
        checks,
        "stage measurement update required files",
        not missing_files,
        f"missing={missing_files}",
    )

    route = read_rows(root, "I0/assembly_topology.csv")
    checkpoints = read_rows(root, "I_meas/measurement_checkpoint.csv")
    observations = read_rows(root, "I_meas/measurement_observation_map.csv")
    configs = read_rows(root, "I_meas/measurement_update_config.csv")
    measurements = read_rows(root, "I_meas/measurement_record.csv")
    basis = read_rows(root, "I_stage/state_update_basis.csv")
    transfers = read_rows(root, "I_stage/stage_covariance_transfer.csv")
    route_by_id = {
        row.get("topology_step_id", ""): row for row in route
    }
    route_order = {
        row.get("topology_step_id", ""): index
        for index, row in enumerate(route)
    }
    checkpoint_ids = [row.get("checkpoint_id", "") for row in checkpoints]
    target_steps = [row.get("topology_step_id", "") for row in checkpoints]
    checkpoint_ok = (
        bool(checkpoints)
        and len(checkpoint_ids) == len(set(checkpoint_ids))
        and len(target_steps) == len(set(target_steps))
    )
    checkpoint_errors: list[str] = []
    for row in checkpoints:
        checkpoint_id = row.get("checkpoint_id", "")
        target = row.get("topology_step_id", "")
        source = row.get("source_topology_step_id", "")
        route_row = route_by_id.get(target, {})
        if (
            not checkpoint_id
            or target not in route_by_id
            or source not in route_by_id
            or route_order.get(source, 10**9) >= route_order.get(target, -1)
            or route_row.get("measurement_checkpoint_id", "") != checkpoint_id
            or truthy(route_row.get("solve_required"))
            or route_row.get("operation_type", "").upper() != "MEASURE"
        ):
            checkpoint_errors.append(checkpoint_id or "<blank>")
    _check(
        checks,
        "checkpoint route relation and explicit source step",
        checkpoint_ok and not checkpoint_errors,
        f"checkpoints={checkpoint_ids}, invalid={checkpoint_errors}",
    )

    measurement_ids = [row.get("measurement_id", "") for row in measurements]
    observation_ids = [row.get("measurement_id", "") for row in observations]
    config_ids = {row.get("update_config_id", "") for row in configs}
    set_ids = {row.get("measurement_set_id", "") for row in measurements}
    foreign_keys_ok = all(
        row.get("measurement_id", "") in set(measurement_ids)
        and row.get("checkpoint_id", "") in set(checkpoint_ids)
        for row in observations
    ) and all(
        row.get("update_config_id", "") in config_ids
        and row.get("measurement_set_id", "") in set_ids
        for row in checkpoints
    )
    _check(
        checks,
        "measurement/config/checkpoint foreign keys",
        foreign_keys_ok
        and len(measurement_ids) == len(set(measurement_ids))
        and len(observation_ids) == len(set(observation_ids)),
        (
            f"measurements={measurement_ids}, observations={observation_ids}, "
            f"configs={sorted(config_ids)}"
        ),
    )
    governance_ok = all(
        row.get("data_role", "").upper() in {"CALIBRATE", "UPDATE"}
        and row.get("update_target", "").upper()
        in {
            "STAGE_STATE", "POSE", "POSE_BIAS", "GAP_BIAS",
            "ACTUAL_LOAD_CORRECTION", "PROCESS_LOAD",
            "LOCATOR_ZERO_BIAS", "OVERCONSTRAINT_STATE",
        }
        and row.get("quality_flag", "").upper() == "PASS"
        for row in measurements
        if row.get("measurement_id", "") in observation_ids
    ) and all(
        not truthy(row.get("parameter_update_allowed"))
        and row.get("quality_flag", "").upper() == "PASS"
        for row in configs
    )
    _check(
        checks,
        "measurement data_role and frozen parameter governance",
        governance_ok,
        "only CALIBRATE/UPDATE state targets; parameter_update_allowed=false",
    )

    with np.load(
        root / "matrices" / "multi_part_matrices.npz",
        allow_pickle=False,
    ) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    layout_ids = {row.get("update_state_layout_id", "") for row in basis}
    matrix_errors: list[str] = []
    mapping_count = 0
    for config in configs:
        state_layout = config.get("update_state_layout_id", "")
        dimension = sum(
            row.get("update_state_layout_id", "") == state_layout
            for row in basis
        )
        keys = [
            config.get("prior_mean_matrix_id", ""),
            config.get("prior_covariance_matrix_id", ""),
            config.get("observation_jacobian_matrix_id", ""),
            config.get("measurement_covariance_matrix_id_optional", ""),
        ]
        if any(key not in arrays for key in keys):
            matrix_errors.append(f"{config.get('update_config_id')}:missing")
            continue
        eta, P, H, R = (arrays[key] for key in keys)
        if (
            state_layout not in layout_ids
            or eta.reshape(-1).size != dimension
            or P.shape != (dimension, dimension)
            or H.shape != (len(observations), dimension)
            or R.shape != (len(observations), len(observations))
            or not np.allclose(P, P.T, atol=1e-12)
            or not np.allclose(R, R.T, atol=1e-12)
            or np.min(np.linalg.eigvalsh(P)) < -1e-12
            or np.min(np.linalg.eigvalsh(R)) <= 0.0
        ):
            matrix_errors.append(f"{config.get('update_config_id')}:shape_or_psd")
        checkpoint = next(
            (
                row for row in checkpoints
                if row.get("update_config_id") == config.get("update_config_id")
            ),
            {},
        )
        checkpoint_id = checkpoint.get("checkpoint_id", "")
        source_step = checkpoint.get("source_topology_step_id", "")
        source_position = route_order.get(source_step, -1)
        operator_sets = [
            row.get("operator_set_id", "")
            for index, row in enumerate(route)
            if (
                (row.get("topology_step_id") == source_step or index > source_position)
                and truthy(row.get("solve_required"))
            )
        ]
        rule = config.get("state_to_q_mapping_rule", "")
        for operator_set in dict.fromkeys(operator_sets):
            key = rule.format(
                checkpoint_id=checkpoint_id,
                operator_set_id=operator_set,
            )
            mapping_count += 1
            if key not in arrays or arrays[key].shape != (
                len(read_rows(root, "I_Gamma/contact_point.csv")),
                dimension,
            ):
                matrix_errors.append(f"{checkpoint_id}:{operator_set}:G_q")
    _check(
        checks,
        "posterior P/H/R/G_q matrix completeness",
        not matrix_errors and mapping_count > 0,
        f"mapping_count={mapping_count}, errors={matrix_errors}",
    )
    transfer_ok = bool(transfers) and all(
        row.get("state_jacobian_F_matrix_id", "") in arrays
        and row.get("process_noise_Q_matrix_id", "") in arrays
        for row in transfers
    )
    _check(
        checks,
        "stage covariance transfer F/Q registration",
        transfer_ok,
        f"transfers={len(transfers)}",
    )

    oracle = read_rows(
        root, "validation/stage_measurement_update_oracle.csv"
    )
    expected_path = (
        root / "validation" / "measurement_update_expected_summary.json"
    )
    expected = (
        json.loads(expected_path.read_text(encoding="utf-8"))
        if expected_path.exists() else {}
    )
    oracle_ok = (
        len(oracle) == 1
        and oracle[0].get("oracle_method")
        == "INDEPENDENT_NUMPY_AND_ACTIVE_SET_ENUMERATION"
        and float(oracle[0].get("posterior_residual_norm", "inf"))
        < float(oracle[0].get("prior_residual_norm", "-inf"))
        and float(oracle[0].get("P_posterior_trace", "inf"))
        < float(oracle[0].get("P_prior_trace", "-inf"))
        and float(oracle[0].get("complementarity_residual", "inf"))
        <= float(oracle[0].get("lcp_tolerance", "0"))
        and expected.get("posterior_accepted") is True
        and expected.get("resolve_lcp_call_count") == 1
        and expected.get("engineering_claim_allowed") is False
    )
    _check(
        checks,
        "independent posterior and LCP oracle",
        oracle_ok,
        (
            f"oracle_rows={len(oracle)}, "
            f"expected_status={expected.get('posterior_status')}"
        ),
    )
    oracle_row = oracle[0] if len(oracle) == 1 else {}
    required_physical_fields = {
        "z_predicted_prior_physical",
        "z_predicted_posterior_linearized",
        "z_predicted_posterior_physical",
        "residual_prior_physical",
        "residual_posterior_linearized",
        "residual_posterior_physical",
        "weighted_residual_prior_physical",
        "weighted_residual_posterior_physical",
        "linearization_error",
        "physical_residual_improved",
    }
    _check(
        checks,
        "oracle physical observation fields",
        required_physical_fields <= set(oracle_row),
        f"missing={sorted(required_physical_fields - set(oracle_row))}",
    )
    physical_improved = (
        truthy(oracle_row.get("physical_residual_improved"))
        and float(
            oracle_row.get(
                "weighted_residual_posterior_physical", "inf"
            )
        )
        < float(
            oracle_row.get(
                "weighted_residual_prior_physical", "-inf"
            )
        )
        and float(oracle_row.get("posterior_residual_norm", "inf"))
        < float(oracle_row.get("prior_residual_norm", "-inf"))
    )
    _check(
        checks,
        "post-LCP physical residual improvement",
        physical_improved,
        (
            "raw="
            f"{oracle_row.get('prior_residual_norm')}->"
            f"{oracle_row.get('posterior_residual_norm')}; "
            "weighted="
            f"{oracle_row.get('weighted_residual_prior_physical')}->"
            f"{oracle_row.get('weighted_residual_posterior_physical')}"
        ),
    )
    finite_difference_ok = (
        oracle_row.get("H_derivation_source")
        == "INDEPENDENT_GLOBAL_LCP_CENTRAL_FINITE_DIFFERENCE"
        and oracle_row.get("finite_difference_method")
        == "CENTRAL_DIFFERENCE"
        and float(
            oracle_row.get("finite_difference_epsilon", "0")
        ) > 0.0
        and truthy(oracle_row.get("active_set_stable"))
    )
    _check(
        checks,
        "H independent physical finite difference",
        finite_difference_ok,
        (
            f"source={oracle_row.get('H_derivation_source')};"
            f" method={oracle_row.get('finite_difference_method')};"
            f" stable={oracle_row.get('active_set_stable')}"
        ),
    )
    _check(
        checks,
        "oracle uses unified observation extractor",
        oracle_row.get("observation_extractor")
        == (
            "core.stage_measurement_update."
            "extract_observation_vector"
        ),
        f"extractor={oracle_row.get('observation_extractor')}",
    )
    frozen_hash_ok = (
        bool(oracle_row.get("frozen_package_hash_before"))
        and oracle_row.get("frozen_package_hash_before")
        == oracle_row.get("frozen_package_hash_after")
        == expected.get("frozen_package_hash_before")
        == expected.get("frozen_package_hash_after")
    )
    _check(
        checks,
        "parameter SMS Cn W_struct frozen package hash",
        frozen_hash_ok,
        (
            f"before={oracle_row.get('frozen_package_hash_before')};"
            f" after={oracle_row.get('frozen_package_hash_after')}"
        ),
    )
    _check(
        checks,
        "single posterior global LCP call",
        expected.get("resolve_lcp_call_count") == 1,
        f"resolve_lcp_call_count={expected.get('resolve_lcp_call_count')}",
    )
    _check(
        checks,
        "posterior covariance trace reduction",
        float(oracle_row.get("P_posterior_trace", "inf"))
        < float(oracle_row.get("P_prior_trace", "-inf")),
        (
            f"trace={oracle_row.get('P_prior_trace')}->"
            f"{oracle_row.get('P_posterior_trace')}"
        ),
    )

    object_map = read_rows(root, "object_file_map.csv")
    object_names = {row.get("object_name", "") for row in object_map}
    _check(
        checks,
        "measurement update objects registered",
        REQUIRED_OBJECTS <= object_names,
        f"missing={sorted(REQUIRED_OBJECTS - object_names)}",
    )
    truth_ok = (
        manifest.get("data_nature")
        == "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE"
        and manifest.get("engineering_claim_allowed") is False
        and manifest.get("measurement_data_nature")
        == "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE"
    )
    _check(
        checks,
        "measurement truthfulness boundary",
        truth_ok,
        (
            f"data_nature={manifest.get('data_nature')}, "
            f"measurement_data_nature={manifest.get('measurement_data_nature')}, "
            f"engineering_claim_allowed={manifest.get('engineering_claim_allowed')}"
        ),
    )

    if include_attachment_check:
        stored_path = root / "validation" / "test_results.json"
        stored = (
            json.loads(stored_path.read_text(encoding="utf-8"))
            if stored_path.exists() else {}
        )
        stored_core = [
            (item.get("name"), item.get("status"), item.get("detail"))
            for item in stored.get("checks", [])
            if item.get("name")
            != "validation attachments match current package"
        ]
        current_core = [
            (item["name"], item["status"], item["detail"])
            for item in checks
        ]
        run_log = read_rows(root, "validation/run_log.csv")
        quality_gate = read_rows(root, "validation/quality_gate.csv")
        attachment_ok = (
            stored.get("package_id") == root.name
            and stored_core == current_core
            and bool(run_log)
            and run_log[0].get("input_package_id") == root.name
            and bool(quality_gate)
            and quality_gate[0].get("target_object_ids") == root.name
            and quality_gate[0].get("pass_fail") == "PASS"
        )
        _check(
            checks,
            "validation attachments match current package",
            attachment_ok,
            f"stored_status={stored.get('status')}",
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
    args = parser.parse_args()
    report = validate_package(args.package)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
