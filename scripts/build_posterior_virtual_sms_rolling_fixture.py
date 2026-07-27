from __future__ import annotations

import csv
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rolling_prediction import stable_package_file_hash


SOURCE = ROOT / "data" / "03_STAGE_MEASUREMENT_UPDATE_MIN_CASE"
TARGET = ROOT / "data" / "04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE"
PACKAGE_ID = TARGET.name
FIXED_TIMESTAMP = "2026-07-26T00:00:00+08:00"
PLAN_ID = "ROLL_PLAN_ABC_TO_ABCD"
SAMPLE_SET_ID = "VSMS_SET_D_EXPLICIT_5"
SMS_LAYOUT_ID = "SMS_LAYOUT_P_D_2"
REFERENCE_SMS_ID = "SMS_REF_P_D_OPERATOR"
SCENARIO_ID = "FUTURE_PROCESS_BASELINE_ZERO"
SAMPLES = [
    ("VSMS_REF", 0, "REFERENCE_SMS", (0.0, 0.0)),
    ("VSMS_M1_POS", 1, "EXPLICIT_VALUE", (1.0, 0.0)),
    ("VSMS_M1_NEG", 2, "EXPLICIT_VALUE", (-1.0, 0.0)),
    ("VSMS_M2_POS", 3, "EXPLICIT_VALUE", (0.0, 1.0)),
    ("VSMS_COMBO", 4, "EXPLICIT_VALUE", (0.75, -0.5)),
]

FIELD_COLUMNS = [
    "file_path", "object_name", "field_name", "data_type", "required",
    "cardinality", "unit", "enum_or_format", "key_semantics",
    "missing_handling", "description", "example_value",
]
OBJECT_COLUMNS = [
    "object_name", "input_package", "file_path", "primary_key",
    "code_consumer", "producer", "is_synthetic", "is_runtime_result",
    "group", "row_mode",
]


def _write_deterministic_npz(
    path: Path, arrays: dict[str, np.ndarray]
) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for key, array in sorted(arrays.items()):
            buffer = BytesIO()
            np.lib.format.write_array(
                buffer, np.asanyarray(array), allow_pickle=False
            )
            entry = zipfile.ZipInfo(
                f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue(), compresslevel=9)


def _safe_recreate() -> None:
    target = TARGET.resolve()
    data_root = (ROOT / "data").resolve()
    if target.parent != data_root or target.name != PACKAGE_ID:
        raise RuntimeError(f"refusing to rebuild unexpected path: {target}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE, TARGET)
    for stale in (
        TARGET / "field_dictionary.csv",
        TARGET / "object_file_map.csv",
        TARGET / "validation" / "test_results.json",
        TARGET / "validation" / "TEST_RESULTS.md",
        TARGET / "validation" / "validate_package.py",
    ):
        if stale.exists():
            stale.unlink()
    pred = TARGET / "I_pred"
    pred.mkdir(exist_ok=True)


def _write_rolling_tables() -> None:
    pred = TARGET / "I_pred"
    pd.DataFrame([{
        "rolling_plan_id": PLAN_ID,
        "topology_id": "TOPOLOGY_STEP_MIN_CASE",
        "source_checkpoint_id": "MCP_AFTER_ABC",
        "source_topology_step_id": "TS205",
        "source_state_policy": "REQUIRE_ACCEPTED_POSTERIOR",
        "source_posterior_state_id_optional": "",
        "prediction_start_step_id": "TS301",
        "prediction_end_step_id_optional": "TS304",
        "virtual_sms_sample_set_id": SAMPLE_SET_ID,
        "future_part_ids": "P_D",
        "future_process_scenario_id": SCENARIO_ID,
        "kcp_set_id": "KCP_SET_MULTIPART",
        "baseline_comparison_policy": "POSTERIOR_AND_PREDICTED_CUTOFF",
        "aggregation_policy":
            "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE",
        "failure_policy": "FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS",
        "active_flag": True,
        "engineering_claim_allowed": False,
        "quality_flag": "PASS",
        "notes": "Deterministic explicit virtual SMS rolling fixture.",
    }]).to_csv(
        pred / "rolling_prediction_plan.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "sample_set_id": SAMPLE_SET_ID,
        "sample_set_name": "P_D explicit deterministic virtual SMS library",
        "sample_nature": "EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY",
        "sample_count": len(SAMPLES),
        "sms_layout_id": SMS_LAYOUT_ID,
        "reference_sms_id": REFERENCE_SMS_ID,
        "generation_method": "DETERMINISTIC_EXPLICIT_VALUES",
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
        "source": "SYNTHETIC_NUMERICAL_CONSISTENCY_FIXTURE",
        "quality_flag": "PASS",
        "notes": "Not Monte Carlo and not an engineering probability model.",
    }]).to_csv(
        pred / "virtual_sms_sample_set.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "virtual_sms_sample_id": sample_id,
            "sample_set_id": SAMPLE_SET_ID,
            "part_id": "P_D",
            "sms_layout_id": SMS_LAYOUT_ID,
            "reference_sms_id": REFERENCE_SMS_ID,
            "coefficient_source": source,
            "sample_order": order,
            "quality_flag": "PASS",
            "source": "DETERMINISTIC_EXPLICIT_VALUES",
            "notes": "Synthetic explicit coefficient vector.",
        }
        for sample_id, order, source, _ in SAMPLES
    ]).to_csv(
        pred / "virtual_sms_sample.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "sms_layout_id": SMS_LAYOUT_ID,
            "component_order": 0,
            "component_id": "P_D_MODE_1",
            "component_type": "SYNTHETIC_SHAPE_MODE",
            "part_id": "P_D",
            "unit": "1",
            "reference_state_id": REFERENCE_SMS_ID,
            "quality_flag": "PASS",
            "description": "Synthetic future-part SMS mode 1.",
        },
        {
            "sms_layout_id": SMS_LAYOUT_ID,
            "component_order": 1,
            "component_id": "P_D_MODE_2",
            "component_type": "SYNTHETIC_SHAPE_MODE",
            "part_id": "P_D",
            "unit": "1",
            "reference_state_id": REFERENCE_SMS_ID,
            "quality_flag": "PASS",
            "description": "Synthetic future-part SMS mode 2.",
        },
    ]).to_csv(
        pred / "virtual_sms_component.csv",
        index=False, encoding="utf-8-sig",
    )
    coefficient_rows = []
    for sample_id, _, _, coefficients in SAMPLES:
        for component_order, value in enumerate(coefficients):
            coefficient_rows.append({
                "virtual_sms_sample_id": sample_id,
                "part_id": "P_D",
                "component_id": f"P_D_MODE_{component_order + 1}",
                "component_order": component_order,
                "value": value,
                "unit": "1",
                "reference_sms_id": REFERENCE_SMS_ID,
                "quality_flag": "PASS",
            })
    pd.DataFrame(coefficient_rows).to_csv(
        pred / "virtual_sms_coefficients.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "assignment_id": f"ASSIGN_{sample_id}_P_D",
            "rolling_plan_id": PLAN_ID,
            "virtual_sms_sample_id": sample_id,
            "part_id": "P_D",
            "first_effective_topology_step_id": "TS301",
            "last_effective_topology_step_id_optional": "TS304",
            "mapping_semantics": "DELTA_FROM_OPERATOR_REFERENCE",
            "reference_sms_id": REFERENCE_SMS_ID,
            "quality_flag": "PASS",
        }
        for sample_id, *_ in SAMPLES
    ]).to_csv(
        pred / "future_sms_assignment.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "mapping_id": f"MAP_SMS_P_D_{operator_id}",
            "rolling_plan_id": PLAN_ID,
            "part_id": "P_D",
            "operator_set_id": operator_id,
            "matrix_id": f"G_SMS_P_D_{operator_id}",
            "row_vector_layout_id": "VL_CONTACT_ALL_12",
            "column_sms_layout_id": SMS_LAYOUT_ID,
            "mapping_semantics": "DELTA_FROM_OPERATOR_REFERENCE",
            "reference_sms_id": REFERENCE_SMS_ID,
            "first_valid_step_id": step_id,
            "last_valid_step_id_optional": step_id,
            "mapping_role": "EFFECTIVE_MAPPING",
            "derivation_source":
                "SYNTHETIC_PRECOMPUTED_VIRTUAL_SMS_TO_Q_MAPPING",
            "quality_flag": "PASS",
        }
        for step_id, operator_id in (
            ("TS301", "OP_TS301"), ("TS302", "OP_TS302"),
            ("TS303", "OP_TS303"), ("TS304", "OP_TS304"),
        )
    ]).to_csv(
        pred / "sms_operator_mapping.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "future_process_scenario_id": SCENARIO_ID,
        "scenario_type": "DETERMINISTIC_BASELINE",
        "q_correction_matrix_id_optional": "Q_PROCESS_ROLLING_ZERO",
        "parameter_override_allowed": False,
        "probability_weight_optional": "",
        "probability_interpretation_allowed": False,
        "quality_flag": "PASS",
    }]).to_csv(
        pred / "future_process_scenario.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "rolling_plan_id": PLAN_ID,
        "part_id": "P_D",
        "kcp_set_id": "KCP_SET_MULTIPART",
        "direct_sms_matrix_id_optional": "J_SMS_ROLL_P_D",
        "aggregation_policy":
            "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE",
        "final_state_includes_direct_sms_geometry": False,
        "quality_flag": "PASS",
    }]).to_csv(
        pred / "rolling_kcp_config.csv",
        index=False, encoding="utf-8-sig",
    )
    kcp_path = TARGET / "I_key" / "kcp_definition.csv"
    kcp_definitions = pd.read_csv(kcp_path, encoding="utf-8-sig")
    kcp_definitions["kcp_set_id"] = "KCP_SET_MULTIPART"
    kcp_definitions.to_csv(
        kcp_path, index=False, encoding="utf-8-sig"
    )


def _write_matrices() -> dict[str, np.ndarray]:
    npz_path = TARGET / "matrices" / "multi_part_matrices.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    layout = pd.read_csv(
        TARGET / "matrices" / "vector_layout.csv", encoding="utf-8-sig"
    )
    G = np.zeros((len(pd.read_csv(
        TARGET / "I_Gamma" / "contact_point.csv", encoding="utf-8-sig"
    )), 2))
    for interface_id, mode_1, mode_2 in (
        ("G_AD", (0.0018, -0.0009, 0.0005), (0.0003, 0.0012, -0.0010)),
        ("G_DB", (-0.0011, 0.0015, 0.0007), (0.0010, -0.0004, 0.0013)),
    ):
        row = layout[layout["object_id"].astype(str).eq(interface_id)].iloc[0]
        sl = slice(int(row["start_index"]), int(row["end_index"]) + 1)
        G[sl, 0] = mode_1
        G[sl, 1] = mode_2
    for scale, operator_id in (
        (1.0, "OP_TS301"), (0.85, "OP_TS302"),
        (0.7, "OP_TS303"), (0.9, "OP_TS304"),
    ):
        arrays[f"G_SMS_P_D_{operator_id}"] = scale * G
    arrays["Q_PROCESS_ROLLING_ZERO"] = np.zeros(G.shape[0])
    arrays["J_SMS_ROLL_P_D"] = np.array([
        [0.0010, 0.00035],
        [-0.00045, 0.00055],
        [0.00080, -0.00025],
    ])
    _write_deterministic_npz(npz_path, arrays)

    manifest_path = TARGET / "matrices" / "matrix_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    for column in (
        "future_part_id_optional", "reference_sms_id_optional",
        "mapping_semantics_optional", "mapping_role_optional",
        "probability_interpretation_allowed",
    ):
        if column not in manifest.columns:
            manifest[column] = ""
    new_rows = []
    for step_id, operator_id in (
        ("TS301", "OP_TS301"), ("TS302", "OP_TS302"),
        ("TS303", "OP_TS303"), ("TS304", "OP_TS304"),
    ):
        key = f"G_SMS_P_D_{operator_id}"
        new_rows.append({
            "matrix_id": key,
            "npz_file": "multi_part_matrices.npz",
            "npz_key": key,
            "shape": json.dumps(list(arrays[key].shape)),
            "dtype": str(arrays[key].dtype),
            "unit": "mm_per_sms_coefficient",
            "row_layout_id_optional": "VL_CONTACT_ALL_12",
            "column_layout_id_optional": SMS_LAYOUT_ID,
            "derivation_source":
                "SYNTHETIC_PRECOMPUTED_VIRTUAL_SMS_TO_Q_MAPPING",
            "description": f"Synthetic P_D virtual SMS mapping for {step_id}.",
            "quality_flag": "PASS",
            "measurement_checkpoint_id_optional": "",
            "operator_set_id_optional": operator_id,
            "data_nature":
                "SYNTHETIC_DETERMINISTIC_VIRTUAL_SMS_ROLLING_CASE",
            "engineering_claim_allowed": False,
            "future_part_id_optional": "P_D",
            "reference_sms_id_optional": REFERENCE_SMS_ID,
            "mapping_semantics_optional":
                "DELTA_FROM_OPERATOR_REFERENCE",
            "mapping_role_optional": "EFFECTIVE_MAPPING",
            "probability_interpretation_allowed": False,
        })
    for key, shape, row_layout, column_layout, role in (
        ("Q_PROCESS_ROLLING_ZERO", arrays["Q_PROCESS_ROLLING_ZERO"].shape,
         "VL_CONTACT_ALL_12", "", "EXPLICIT_ZERO_NO_EFFECT"),
        ("J_SMS_ROLL_P_D", arrays["J_SMS_ROLL_P_D"].shape,
         "KCP_SET_MULTIPART", SMS_LAYOUT_ID, "SMS_TO_KCP_DIRECT"),
    ):
        new_rows.append({
            "matrix_id": key,
            "npz_file": "multi_part_matrices.npz",
            "npz_key": key,
            "shape": json.dumps(list(shape)),
            "dtype": str(arrays[key].dtype),
            "unit": "per_object_definition",
            "row_layout_id_optional": row_layout,
            "column_layout_id_optional": column_layout,
            "derivation_source":
                "SYNTHETIC_DETERMINISTIC_ROLLING_FIXTURE",
            "description": f"Synthetic rolling matrix {key}.",
            "quality_flag": "PASS",
            "measurement_checkpoint_id_optional": "",
            "operator_set_id_optional": "",
            "data_nature":
                "SYNTHETIC_DETERMINISTIC_VIRTUAL_SMS_ROLLING_CASE",
            "engineering_claim_allowed": False,
            "future_part_id_optional": (
                "P_D" if key == "J_SMS_ROLL_P_D" else ""
            ),
            "reference_sms_id_optional": (
                REFERENCE_SMS_ID if key == "J_SMS_ROLL_P_D" else ""
            ),
            "mapping_semantics_optional": (
                "DELTA_FROM_OPERATOR_REFERENCE"
                if key == "J_SMS_ROLL_P_D" else "EXPLICIT_ZERO"
            ),
            "mapping_role_optional": role,
            "probability_interpretation_allowed": False,
        })
    manifest = pd.concat(
        [manifest, pd.DataFrame(new_rows)], ignore_index=True
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return arrays


def _independent_lcp(q: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(q)
    best = None
    for size in range(n + 1):
        for active in combinations(range(n), size):
            lam = np.zeros(n)
            if active:
                idx = np.asarray(active, dtype=int)
                try:
                    lam[idx] = np.linalg.solve(
                        W[np.ix_(idx, idx)], -q[idx]
                    )
                except np.linalg.LinAlgError:
                    continue
            gap = q + W @ lam
            if (
                np.min(lam, initial=0.0) >= -1e-9
                and np.min(gap, initial=0.0) >= -1e-9
                and np.max(np.abs(lam * gap), initial=0.0) <= 1e-8
            ):
                score = (
                    float(np.linalg.norm(np.minimum(lam, 0.0))),
                    float(np.linalg.norm(np.minimum(gap, 0.0))),
                    float(np.linalg.norm(lam * gap)),
                    active,
                )
                if best is None or score < best[0]:
                    best = (score, np.maximum(lam, 0.0), np.maximum(gap, 0.0))
    if best is None:
        raise RuntimeError("independent active-set enumeration found no LCP solution")
    return best[1], best[2]


def _active_indices_by_step(route: pd.DataFrame, layout: pd.DataFrame) -> dict[str, np.ndarray]:
    active: list[str] = []
    out = {}
    for _, row in route.sort_values(
        ["step_order", "topology_step_id"], kind="stable"
    ).iterrows():
        for item in str(row.get("activated_interface_ids", "")).split(";"):
            if item and item not in active:
                active.append(item)
        for item in str(row.get("deactivated_interface_ids", "")).split(";"):
            if item in active:
                active.remove(item)
        indices = []
        for interface_id in active:
            block = layout[
                layout["object_id"].astype(str).eq(interface_id)
            ].iloc[0]
            indices.extend(range(
                int(block["start_index"]), int(block["end_index"]) + 1
            ))
        out[str(row["topology_step_id"])] = np.asarray(sorted(indices), dtype=int)
    return out


def _write_independent_oracle(arrays: dict[str, np.ndarray]) -> None:
    route = pd.read_csv(
        TARGET / "I0" / "assembly_topology.csv", encoding="utf-8-sig"
    )
    layout = pd.read_csv(
        TARGET / "matrices" / "vector_layout.csv", encoding="utf-8-sig"
    )
    active_by_step = _active_indices_by_step(route, layout)
    measurement_oracle = pd.read_csv(
        TARGET / "validation" / "stage_measurement_update_oracle.csv",
        encoding="utf-8-sig",
    ).iloc[0]
    eta_post = np.asarray(json.loads(measurement_oracle["eta_posterior"]), float)
    eta_pred = np.asarray(json.loads(measurement_oracle["eta_prior"]), float)
    base_prediction = pd.read_csv(
        TARGET / "prediction" / "kcp_prediction_result.csv",
        encoding="utf-8-sig",
    ).iloc[0]
    sms_base = np.asarray(json.loads(base_prediction["sms_contribution"]), float)
    other = np.zeros_like(sms_base)
    for column in (
        "process_contribution", "rebound_contribution",
        "slip_contribution_optional",
        "measurement_uncertainty_contribution_optional",
    ):
        value = np.asarray(json.loads(base_prediction[column]), float)
        if value.size:
            if value.shape != sms_base.shape:
                raise ValueError(
                    f"{column} shape {value.shape} does not match "
                    f"KCP vector shape {sms_base.shape}"
                )
            other = other + value
    kcp_defs = pd.read_csv(
        TARGET / "I_key" / "kcp_definition.csv", encoding="utf-8-sig"
    )
    kcp_ids = kcp_defs["kcp_id"].astype(str).tolist()
    J_interface = arrays["J_INTERFACE_ALL"]
    J_sms = arrays["J_SMS_ROLL_P_D"]
    rows = []
    kcp_rows = []
    for source_role, eta in (("PREDICTED", eta_pred), ("POSTERIOR", eta_post)):
        for sample_id, _, _, coefficients in SAMPLES:
            delta = np.asarray(coefficients, float)
            final_local = None
            for step_id, operator_id in (
                ("TS301", "OP_TS301"), ("TS302", "OP_TS302"),
                ("TS303", "OP_TS303"), ("TS304", "OP_TS304"),
            ):
                q_operator = arrays[f"Q_{operator_id}"]
                G_eta = arrays[
                    f"G_Q_UPDATE__MCP_AFTER_ABC__{operator_id}"
                ]
                q_state = G_eta @ eta
                q_sms = arrays[f"G_SMS_P_D_{operator_id}"] @ delta
                q = q_operator + q_state + q_sms
                W_struct = arrays[f"W_STRUCT_{operator_id}"]
                Cn = arrays[f"CN_{operator_id}"]
                W_total = W_struct + Cn
                active = active_by_step[step_id]
                lam_active, gap_active = _independent_lcp(
                    q[active], W_total[np.ix_(active, active)]
                )
                lam = np.zeros_like(q)
                gap = np.full_like(q, np.nan)
                lam[active] = lam_active
                gap[active] = gap_active
                local = np.full_like(q, np.nan)
                local[active] = Cn[np.ix_(active, active)] @ lam_active
                final_local = np.nan_to_num(local)
                rows.append({
                    "source_state_role": source_role,
                    "virtual_sms_sample_id": sample_id,
                    "topology_step_id": step_id,
                    "operator_set_id": operator_id,
                    "sms_coefficients": json.dumps(delta.tolist()),
                    "sms_reference_coefficients": "[0.0,0.0]",
                    "sms_delta_coefficients": json.dumps(delta.tolist()),
                    "q_operator_base": json.dumps(q_operator.tolist()),
                    "q_posterior_state_correction": json.dumps(q_state.tolist()),
                    "q_virtual_sms_correction": json.dumps(q_sms.tolist()),
                    "q_future_process_correction": json.dumps(np.zeros_like(q).tolist()),
                    "q_effective": json.dumps(q.tolist()),
                    "active_indices": json.dumps(active.tolist()),
                    "lambda_active": json.dumps(lam_active.tolist()),
                    "gap_active": json.dumps(gap_active.tolist()),
                    "complementarity_residual":
                        float(np.max(np.abs(lam_active * gap_active), initial=0.0)),
                    "lcp_call_count": 1,
                    "oracle_method": "INDEPENDENT_ACTIVE_SET_ENUMERATION",
                    "production_rolling_runner_used": False,
                    "production_topology_runner_used": False,
                    "production_virtual_sms_function_used": False,
                })
            kcp_values = (
                sms_base + J_interface @ final_local + other + J_sms @ delta
            )
            for kcp_id, value in zip(kcp_ids, kcp_values):
                kcp_rows.append({
                    "source_state_role": source_role,
                    "virtual_sms_sample_id": sample_id,
                    "kcp_id": kcp_id,
                    "predicted_value": value,
                    "aggregation_policy":
                        "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE",
                    "oracle_method":
                        "INDEPENDENT_FORMULA_AND_ACTIVE_SET_ENUMERATION",
                    "production_kcp_entry_used": False,
                    "production_rolling_runner_used": False,
                    "engineering_claim_allowed": False,
                    "probability_interpretation_allowed": False,
                })
    pd.DataFrame(rows).to_csv(
        TARGET / "validation" / "rolling_prediction_oracle.csv",
        index=False, encoding="utf-8-sig",
    )
    kcp_frame = pd.DataFrame(kcp_rows)
    kcp_frame.to_csv(
        TARGET / "validation" / "rolling_kcp_oracle.csv",
        index=False, encoding="utf-8-sig",
    )
    posterior = kcp_frame[
        kcp_frame["source_state_role"].eq("POSTERIOR")
    ]
    summary = {}
    for kcp_id, group in posterior.groupby("kcp_id", sort=False):
        values = group["predicted_value"].to_numpy(float)
        summary[kcp_id] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    (TARGET / "validation" / "rolling_expected_summary.json").write_text(
        json.dumps({
            "rolling_plan_id": PLAN_ID,
            "sample_count": len(SAMPLES),
            "posterior_success_count": len(SAMPLES),
            "posterior_failure_count": 0,
            "baseline_attempt_count": len(SAMPLES),
            "baseline_success_count": len(SAMPLES),
            "baseline_failure_count": 0,
            "baseline_quality_status": "PASS",
            "sms_application_count": len(SAMPLES) * 4,
            "double_count_fail_count": 0,
            "immutability_status": "PASS",
            "source_linkage_status": "PASS",
            "source_checkpoint_id": "MCP_AFTER_ABC",
            "source_topology_step_id": "TS205",
            "final_state_includes_direct_sms_geometry": False,
            "direct_sms_aggregation_action":
                "ADD_DIRECT_SMS_CONTRIBUTION",
            "probability_interpretation_allowed": False,
            "engineering_claim_allowed": False,
            "kcp_summary": summary,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _object_name(relative: str) -> str:
    names = {
        "I_pred/rolling_prediction_plan.csv": "RollingPredictionPlan",
        "I_pred/virtual_sms_sample_set.csv": "VirtualSMSSampleSet",
        "I_pred/virtual_sms_sample.csv": "VirtualSMSSample",
        "I_pred/virtual_sms_component.csv": "VirtualSMSComponent",
        "I_pred/future_sms_assignment.csv": "FutureSMSAssignment",
        "I_pred/sms_operator_mapping.csv": "SMSOperatorMappingSpec",
        "I_pred/future_process_scenario.csv": "FutureProcessScenario",
        "validation/rolling_prediction_oracle.csv": "RollingPredictionOracle",
        "validation/rolling_kcp_oracle.csv": "RollingKCPOracle",
    }
    return names.get(relative, Path(relative).stem.title().replace("_", ""))


def _write_governance() -> None:
    existing_map = pd.read_csv(
        SOURCE / "object_file_map.csv", encoding="utf-8-sig"
    )
    existing_map["input_package"] = PACKAGE_ID
    object_rows = existing_map[OBJECT_COLUMNS].to_dict("records")
    for path in sorted(TARGET.rglob("*.csv")):
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        primary = next(
            (name for name in header if name.endswith("_id")),
            header[0] if header else "",
        )
        object_rows.append({
            "object_name": _object_name(relative),
            "input_package": PACKAGE_ID,
            "file_path": relative,
            "primary_key": primary,
            "code_consumer":
                "core.rolling_prediction;core.reporting;app;scripts.cli_check",
            "producer":
                "scripts/build_posterior_virtual_sms_rolling_fixture.py",
            "is_synthetic": True,
            "is_runtime_result": False,
            "group": Path(relative).parent.as_posix(),
            "row_mode": "synthetic_deterministic_virtual_sms_rolling",
        })
    for object_name, relative, primary in (
        ("RollingSampleResult", "runtime/rolling_sample_summary.csv",
         "virtual_sms_sample_id"),
        ("RollingPredictionSummary", "runtime/rolling_kcp_summary.csv",
         "rolling_run_id"),
        ("RollingPredictionTrace", "runtime/rolling_prediction_trace.json",
         "rolling_run_id"),
        ("RollingKCPPrediction", "runtime/rolling_kcp_predictions.csv",
         "virtual_sms_sample_id+kcp_id"),
        ("RollingBaselineComparison", "runtime/rolling_baseline_comparison.csv",
         "virtual_sms_sample_id+kcp_id"),
        ("SMSApplicationTrace", "runtime/rolling_sms_application_trace.csv",
         "virtual_sms_sample_id+part_id+topology_step_id"),
    ):
        object_rows.append({
            "object_name": object_name,
            "input_package": PACKAGE_ID,
            "file_path": relative,
            "primary_key": primary,
            "code_consumer": "core.reporting;app",
            "producer": "core.reporting.build_rolling_prediction_report_zip",
            "is_synthetic": True,
            "is_runtime_result": True,
            "group": "runtime",
            "row_mode": "runtime_result",
        })
    pd.DataFrame(object_rows, columns=OBJECT_COLUMNS).drop_duplicates(
        ["object_name", "file_path"]
    ).to_csv(
        TARGET / "object_file_map.csv",
        index=False, encoding="utf-8-sig",
    )
    schemas = {}
    for path in sorted(TARGET.rglob("*.csv")):
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            schemas[relative] = (reader.fieldnames or [], list(reader))
    schemas["field_dictionary.csv"] = (FIELD_COLUMNS, [])
    rows = []
    for relative, (fields, records) in sorted(schemas.items()):
        for field_name in fields:
            values = [str(row.get(field_name, "")) for row in records]
            rows.append({
                "file_path": relative,
                "object_name": _object_name(relative),
                "field_name": field_name,
                "data_type": "string",
                "required": field_name.endswith("_id"),
                "cardinality": "1" if field_name.endswith("_id") else "0..1",
                "unit": "per_object_definition",
                "enum_or_format": "",
                "key_semantics":
                    "primary or foreign key" if field_name.endswith("_id") else "",
                "missing_handling":
                    "blocking if required" if field_name.endswith("_id") else "blank allowed",
                "description": f"{_object_name(relative)}.{field_name}",
                "example_value": next((v for v in values if v.strip()), ""),
            })
    pd.DataFrame(rows, columns=FIELD_COLUMNS).to_csv(
        TARGET / "field_dictionary.csv",
        index=False, encoding="utf-8-sig",
    )


def _write_manifest_and_readme() -> None:
    path = TARGET / "package_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    manifest.update({
        "package_name": PACKAGE_ID,
        "schema_version": "V2.5_POSTERIOR_VIRTUAL_SMS_ROLLING",
        "schema_revision": "POSTERIOR_VIRTUAL_SMS_ROLLING_2026-07",
        "created_at": FIXED_TIMESTAMP,
        "description":
            "Synthetic deterministic posterior-driven virtual SMS rolling fixture.",
        "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "measurement_data_nature":
            "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "virtual_sms_sample_nature":
            "EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY",
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
        "rolling_plan_count": 1,
        "virtual_sms_sample_count": len(SAMPLES),
    })
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (TARGET / "README.md").write_text(
        "# 04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE\n\n"
        "Deterministic synthetic integration fixture. The formal route starts "
        "from the accepted posterior at MCP_AFTER_ABC, applies five explicit "
        "P_D virtual-SMS coefficient vectors from TS301 through TS304, preserves "
        "the common contact VectorLayout and one coupled global LCP per solve "
        "step, and produces final KCP values plus a predicted-cutoff comparison.\n\n"
        "This is not Monte Carlo, does not define a probability distribution or "
        "failure probability, and is not engineering validation. "
        "`probability_interpretation_allowed=false` and "
        "`engineering_claim_allowed=false`.\n",
        encoding="utf-8",
    )


def package_digest(root: Path = TARGET) -> str:
    return stable_package_file_hash(root)


def _write_validation_attachments() -> dict[str, object]:
    from posterior_virtual_sms_rolling_package_validator import (
        markdown_report,
        validate_package,
    )

    validation = TARGET / "validation"
    initial = validate_package(TARGET, include_attachment_check=False)
    if initial["status"] != "PASS":
        failed = [
            item["name"] for item in initial["checks"]
            if item["status"] == "FAIL" and item["blocking"]
        ]
        raise RuntimeError(f"rolling fixture self-validation failed: {failed}")
    pd.DataFrame([{
        "run_id": "ROLLING_FIXTURE_SELF_VALIDATION",
        "input_package_id": PACKAGE_ID,
        "started_at": FIXED_TIMESTAMP,
        "finished_at": FIXED_TIMESTAMP,
        "status": "PASS",
        "passed_checks": initial["passed"],
        "total_checks": initial["total"],
        "matrix_manifest_count": initial["matrix_manifest_count"],
        "npz_key_count": initial["npz_key_count"],
        "validator":
            "scripts/posterior_virtual_sms_rolling_package_validator.py",
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "pandas_version": pd.__version__,
        "package_file_hash": stable_package_file_hash(TARGET),
    }]).to_csv(
        validation / "run_log.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        "gate_id": "ROLLING_PACKAGE_GATE",
        "gate_name": "POSTERIOR_VIRTUAL_SMS_ROLLING_SELF_VALIDATION",
        "target_object_ids": PACKAGE_ID,
        "rule_expression":
            "all blocking structural, oracle, physical and truth gates pass",
        "pass_fail": "PASS",
        "evidence_reference": "validation/test_results.json",
        "approved_by": "DETERMINISTIC_GENERATOR",
        "approval_time": FIXED_TIMESTAMP,
        "notes": "Synthetic numerical-consistency fixture only.",
    }]).to_csv(
        validation / "quality_gate.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "validation_id": f"VAL_{index + 1:03d}",
        "check_item": item["name"],
        "status": item["status"],
        "blocking": item["blocking"],
        "detail": item["detail"],
    } for index, item in enumerate(initial["checks"])]).to_csv(
        validation / "validation_result.csv",
        index=False, encoding="utf-8-sig",
    )
    (validation / "test_results.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (validation / "TEST_RESULTS.md").write_text(
        markdown_report(initial), encoding="utf-8"
    )
    # The validation CSVs are themselves governed package objects. Refresh the
    # dictionaries after their schemas exist, then store the resulting stable
    # check set used by the attachment-consistency gate.
    _write_governance()
    initial = validate_package(TARGET, include_attachment_check=False)
    if initial["status"] != "PASS":
        raise RuntimeError(
            "rolling fixture governance refresh validation failed"
        )
    pd.DataFrame([{
        "run_id": "ROLLING_FIXTURE_SELF_VALIDATION",
        "input_package_id": PACKAGE_ID,
        "started_at": FIXED_TIMESTAMP,
        "finished_at": FIXED_TIMESTAMP,
        "status": "PASS",
        "passed_checks": initial["passed"],
        "total_checks": initial["total"],
        "matrix_manifest_count": initial["matrix_manifest_count"],
        "npz_key_count": initial["npz_key_count"],
        "validator":
            "scripts/posterior_virtual_sms_rolling_package_validator.py",
    }]).to_csv(
        validation / "run_log.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{
        "validation_id": f"VAL_{index + 1:03d}",
        "check_item": item["name"],
        "status": item["status"],
        "blocking": item["blocking"],
        "detail": item["detail"],
    } for index, item in enumerate(initial["checks"])]).to_csv(
        validation / "validation_result.csv",
        index=False, encoding="utf-8-sig",
    )
    (validation / "test_results.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (validation / "TEST_RESULTS.md").write_text(
        markdown_report(initial), encoding="utf-8"
    )
    final = validate_package(TARGET, include_attachment_check=True)
    if final["status"] != "PASS":
        raise RuntimeError("rolling validation attachment check failed")
    (validation / "test_results.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (validation / "TEST_RESULTS.md").write_text(
        markdown_report(final), encoding="utf-8"
    )
    return {
        "validation_status": final["status"],
        "validation_passed": final["passed"],
        "validation_total": final["total"],
    }


def build() -> dict[str, object]:
    _safe_recreate()
    _write_rolling_tables()
    arrays = _write_matrices()
    _write_independent_oracle(arrays)
    _write_manifest_and_readme()
    _write_governance()
    validator_source = (
        ROOT / "scripts" /
        "posterior_virtual_sms_rolling_package_validator.py"
    )
    if validator_source.exists():
        shutil.copyfile(
            validator_source,
            TARGET / "validation" / "validate_package.py",
        )
    validation_summary = _write_validation_attachments()
    return {
        "package": PACKAGE_ID,
        "rolling_plan_count": 1,
        "sample_count": len(SAMPLES),
        "matrix_count": len(arrays),
        "package_digest": package_digest(),
        **validation_summary,
    }


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
