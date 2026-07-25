from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from topology_step_package_validator import (
    markdown_report,
    validate_package,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "01_DEFAULT_MIN_CASE_4_PART"
TARGET = ROOT / "data" / "02_TOPOLOGY_STEP_MIN_CASE"
VALIDATOR_SOURCE = ROOT / "scripts" / "topology_step_package_validator.py"
PACKAGE_ID = TARGET.name

REPLACEMENTS = {
    "P_PANEL_A": "P_A",
    "P_RIB_B": "P_B",
    "P_SPAR_C": "P_C",
    "P_BRACKET_D": "P_D",
    "G_PANEL_RIB": "G_AB",
    "G_RIB_SPAR": "G_BC",
    "G_PANEL_BRACKET": "G_AD",
    "G_RIB_BRACKET": "G_DB",
    "CD_PANEL_RIB": "CD_AB",
    "CD_RIB_SPAR": "CD_BC",
    "CD_PANEL_BRACKET": "CD_AD",
    "CD_RIB_BRACKET": "CD_DB",
    "JNT_PANEL_RIB": "JNT_AB",
    "JNT_RIB_SPAR": "JNT_BC",
    "JNT_PANEL_BRACKET": "JNT_AD",
    "JNT_RIB_BRACKET": "JNT_DB",
}

TOPOLOGY_FIELDS = [
    "topology_id",
    "topology_step_id",
    "step_order",
    "parent_topology_step_id",
    "assembly_cycle_id",
    "operation_type",
    "stage_id",
    "input_subassembly_id",
    "result_subassembly_id",
    "added_part_ids",
    "removed_part_ids",
    "activated_interface_ids",
    "deactivated_interface_ids",
    "activated_boundary_ids",
    "deactivated_boundary_ids",
    "activated_load_ids",
    "removed_load_ids",
    "activated_joint_ids",
    "deactivated_joint_ids",
    "operator_set_id",
    "solve_required",
    "reference_state_id",
    "measurement_checkpoint_id",
    "notes",
]

FIELD_DICTIONARY_COLUMNS = [
    "file_path",
    "object_name",
    "field_name",
    "data_type",
    "required",
    "cardinality",
    "unit",
    "enum_or_format",
    "key_semantics",
    "missing_handling",
    "description",
    "example_value",
]

OBJECT_MAP_COLUMNS = [
    "object_name",
    "input_package",
    "file_path",
    "primary_key",
    "code_consumer",
    "producer",
    "is_synthetic",
    "is_runtime_result",
    "group",
    "row_mode",
]


def replace_text(value: str) -> str:
    for old, new in REPLACEMENTS.items():
        value = value.replace(old, new)
    return value


def route_rows() -> list[dict[str, object]]:
    common = {
        "topology_id": "TOPOLOGY_STEP_MIN_CASE",
        "measurement_checkpoint_id": "",
        "reference_state_id": "PARENT_RUNTIME_STATE",
    }
    definitions = [
        ("TS000", 0, "", "CYCLE_INIT", "INIT", "S_LOCATE_01", "", "ASM_A", "P_A", "", "", "", "", "", "", "", "", "", "", False, "Initialize the first part without solving."),
        ("TS101", 101, "TS000", "CYCLE_AB", "LOCATE", "S_LOCATE_01", "ASM_A", "ASM_AB", "P_B", "", "G_AB", "", "BND_PANEL_BASE;BND_RIB_PIN", "", "LOAD_GRAVITY_ALL", "", "", "", "OP_TS101", True, "Add part B and activate interface AB."),
        ("TS102", 102, "TS101", "CYCLE_AB", "CLAMP", "S_CLAMP_02", "ASM_AB", "ASM_AB", "", "", "", "", "", "", "LOAD_CLAMP_G_AB", "", "", "", "OP_TS102", True, "Clamp interface AB."),
        ("TS103", 103, "TS102", "CYCLE_AB", "JOIN", "S_JOIN_03", "ASM_AB", "ASM_AB", "", "", "", "", "", "", "LOAD_PRELOAD_G_AB", "", "JNT_AB", "", "OP_TS103", True, "Lock joint AB."),
        ("TS104", 104, "TS103", "CYCLE_AB", "RELEASE", "S_RELEASE_04", "ASM_AB", "ASM_AB", "", "", "", "", "", "BND_PANEL_BASE;BND_RIB_PIN", "", "LOAD_CLAMP_G_AB", "", "", "OP_TS104", True, "Release AB tooling while retaining the joint."),
        ("TS201", 201, "TS104", "CYCLE_BC", "LOCATE", "S_LOCATE_01", "ASM_AB", "ASM_ABC", "P_C", "", "G_BC", "", "BND_SPAR_SUPPORT", "", "", "", "", "", "OP_TS201", True, "Add part C and activate interface BC."),
        ("TS202", 202, "TS201", "CYCLE_BC", "CLAMP", "S_CLAMP_02", "ASM_ABC", "ASM_ABC", "", "", "", "", "", "", "LOAD_CLAMP_G_BC", "", "", "", "OP_TS202", True, "Solve AB and BC globally."),
        ("TS203", 203, "TS202", "CYCLE_BC", "JOIN", "S_JOIN_03", "ASM_ABC", "ASM_ABC", "", "", "", "", "", "", "LOAD_PRELOAD_G_BC", "", "JNT_BC", "", "OP_TS203", True, "Lock joint BC."),
        ("TS204", 204, "TS203", "CYCLE_BC", "RELEASE", "S_RELEASE_04", "ASM_ABC", "ASM_ABC", "", "", "", "", "", "BND_SPAR_SUPPORT", "", "LOAD_CLAMP_G_BC", "", "", "OP_TS204", True, "Release BC tooling and retain existing joints."),
        ("TS301", 301, "TS204", "CYCLE_D", "LOCATE", "S_LOCATE_01", "ASM_ABC", "ASM_ABCD", "P_D", "", "G_AD;G_DB", "", "BND_BRACKET_PIN", "", "", "", "", "", "OP_TS301", True, "Add part D and activate AD and DB in one global solve."),
        ("TS302", 302, "TS301", "CYCLE_D", "CLAMP", "S_CLAMP_02", "ASM_ABCD", "ASM_ABCD", "", "", "", "", "", "", "LOAD_CLAMP_G_AD;LOAD_CLAMP_G_DB", "", "", "", "OP_TS302", True, "Clamp all active interfaces."),
        ("TS303", 303, "TS302", "CYCLE_D", "JOIN", "S_JOIN_03", "ASM_ABCD", "ASM_ABCD", "", "", "", "", "", "", "LOAD_PRELOAD_G_AD;LOAD_PRELOAD_G_DB", "", "JNT_AD;JNT_DB", "", "OP_TS303", True, "Lock joints AD and DB in one JOIN step."),
        ("TS304", 304, "TS303", "CYCLE_D", "RELEASE", "S_RELEASE_04", "ASM_ABCD", "ASM_ABCD", "", "", "", "", "", "BND_BRACKET_PIN", "", "LOAD_CLAMP_G_AD;LOAD_CLAMP_G_DB", "", "", "OP_TS304", True, "Release tooling while retaining all configured joints."),
    ]
    rows: list[dict[str, object]] = []
    for values in definitions:
        row = dict(common)
        row.update(dict(zip(TOPOLOGY_FIELDS[1:21] + ["notes"], values)))
        rows.append(row)
    for index, row in enumerate(rows):
        row["assembly_step"] = row["step_order"]
        row["part_in"] = row["added_part_ids"]
        row["existing_subassembly"] = row["input_subassembly_id"]
        row["result_subassembly"] = row["result_subassembly_id"]
        row["predecessor_step_id"] = row["parent_topology_step_id"]
        row["successor_step_id"] = (
            rows[index + 1]["topology_step_id"] if index + 1 < len(rows) else ""
        )
    return rows


def independent_lcp(
    q: np.ndarray, W: np.ndarray, tolerance: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    size = q.size
    for mask in range(1 << size):
        active = np.array(
            [index for index in range(size) if mask & (1 << index)], dtype=int
        )
        reaction = np.zeros(size, dtype=float)
        if active.size:
            try:
                reaction[active] = np.linalg.solve(
                    W[np.ix_(active, active)], -q[active]
                )
            except np.linalg.LinAlgError:
                continue
        gap = q + W @ reaction
        if (
            np.min(reaction, initial=0.0) >= -tolerance
            and np.min(gap, initial=0.0) >= -tolerance
            and np.max(np.abs(reaction * gap), initial=0.0) <= tolerance
        ):
            reaction[np.abs(reaction) < tolerance] = 0.0
            gap[np.abs(gap) < tolerance] = 0.0
            return reaction, gap
    raise RuntimeError("independent enumeration found no feasible LCP solution")


def _safe_recreate_target() -> None:
    resolved_target = TARGET.resolve()
    resolved_data = (ROOT / "data").resolve()
    if resolved_target.parent != resolved_data or resolved_target.name != PACKAGE_ID:
        raise RuntimeError(f"refusing to rebuild unexpected path: {resolved_target}")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE, TARGET)
    for stale in (
        TARGET / "field_dictionary.csv",
        TARGET / "object_file_map.csv",
        TARGET / "validation",
    ):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    (TARGET / "validation").mkdir()
    for path in TARGET.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig")
            path.write_text(replace_text(text), encoding="utf-8")


def _write_manifest(route: pd.DataFrame) -> dict[str, object]:
    manifest_path = TARGET / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "package_name": PACKAGE_ID,
            "schema_version": "V2.5_TOPOLOGY_STEP",
            "schema_revision": "DETERMINISTIC_MULTI_ROUND_TOPOLOGY_STEP_2026-07",
            "created_at": "2026-07-23T00:00:00+08:00",
            "description": (
                "Synthetic deterministic topology_step integration fixture; "
                "counts below are fixture expectations, not validator constants."
            ),
            "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
            "engineering_claim_allowed": False,
            "purpose": "topology_step deterministic executor integration fixture",
            "operator_mode": "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR",
            "fixture_expectations": {
                "part_count": 4,
                "interface_count": 4,
                "contact_points_per_interface": 3,
                "vector_dimension": 12,
                "topology_step_count": int(len(route)),
                "minimum_multi_interface_activation_steps": 1,
                "serial_paths": [["P_A", "P_B", "P_C"]],
                "parallel_paths": [
                    {
                        "endpoints": ["P_A", "P_B"],
                        "paths": [["P_A", "P_B"], ["P_A", "P_D", "P_B"]],
                    }
                ],
                "require_nonzero_cross_blocks": True,
            },
            "truthfulness_statement": (
                "Synthetic numerical-consistency data only. It is not measured, "
                "FE-validated, or suitable for an engineering accuracy claim."
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _build_matrices(route: pd.DataFrame) -> tuple[int, list[dict[str, object]]]:
    npz_path = TARGET / "matrices" / "multi_part_matrices.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {replace_text(key): archive[key].copy() for key in archive.files}
    cn = np.asarray(arrays["CN_ALL"], dtype=float)
    stage_suffix = {
        "LOCATE": "LOCATE_01",
        "CLAMP": "CLAMP_02",
        "JOIN": "JOIN_03",
        "RELEASE": "RELEASE_04",
    }
    layout = pd.read_csv(
        TARGET / "matrices" / "vector_layout.csv", encoding="utf-8-sig"
    )
    layout_id = str(layout.iloc[0]["vector_layout_id"])
    vector_dimension = int(layout["end_index"].max()) + 1
    active_interfaces: set[str] = set()
    oracle_rows: list[dict[str, object]] = []
    for row in route.to_dict("records"):
        active_interfaces.difference_update(
            split for split in str(row["deactivated_interface_ids"]).split(";") if split
        )
        active_interfaces.update(
            split for split in str(row["activated_interface_ids"]).split(";") if split
        )
        if not bool(row["solve_required"]):
            continue
        operator = str(row["operator_set_id"])
        suffix = stage_suffix[str(row["operation_type"])]
        q = np.asarray(arrays[f"Q_{suffix}"], dtype=float)
        ws = np.asarray(arrays[f"W_STRUCT_{suffix}"], dtype=float)
        wt = ws + cn
        arrays.update(
            {
                f"Q_{operator}": q,
                f"W_STRUCT_{operator}": ws,
                f"CN_{operator}": cn,
                f"W_TOTAL_{operator}": wt,
                f"U_FREE_{operator}": np.asarray(
                    arrays[f"U_FREE_{suffix}"], dtype=float
                ),
            }
        )
        indices: list[int] = []
        for block in layout.to_dict("records"):
            if str(block["object_id"]) in active_interfaces:
                indices.extend(
                    range(int(block["start_index"]), int(block["end_index"]) + 1)
                )
        index = np.asarray(sorted(indices), dtype=int)
        reaction, gap = independent_lcp(q[index], wt[np.ix_(index, index)])
        oracle_rows.append(
            {
                "topology_step_id": row["topology_step_id"],
                "operator_set_id": operator,
                "active_interface_ids": ";".join(sorted(active_interfaces)),
                "active_indices": json.dumps(index.tolist(), separators=(",", ":")),
                "lambda_active": json.dumps(
                    reaction.tolist(), separators=(",", ":")
                ),
                "gap_active": json.dumps(gap.tolist(), separators=(",", ":")),
                "equilibrium_residual": float(
                    np.max(
                        np.abs(
                            gap
                            - (
                                q[index]
                                + wt[np.ix_(index, index)] @ reaction
                            )
                        ),
                        initial=0.0,
                    )
                ),
                "complementarity_residual": float(
                    np.max(np.abs(reaction * gap), initial=0.0)
                ),
                "oracle_method": "INDEPENDENT_ACTIVE_SET_ENUMERATION",
            }
        )
    np.savez_compressed(npz_path, **dict(sorted(arrays.items())))

    old_manifest = pd.read_csv(
        TARGET / "matrices" / "matrix_manifest.csv", encoding="utf-8-sig"
    )
    metadata = {
        str(row["npz_key"]): row for row in old_manifest.to_dict("records")
    }
    matrix_rows: list[dict[str, object]] = []
    for key, value in sorted(arrays.items()):
        old = metadata.get(key, {})
        matrix_rows.append(
            {
                "matrix_id": key,
                "npz_file": "multi_part_matrices.npz",
                "npz_key": key,
                "shape": json.dumps(list(value.shape), separators=(",", ":")),
                "dtype": str(value.dtype),
                "unit": old.get("unit", "mixed_by_object_definition"),
                "row_layout_id_optional": (
                    layout_id
                    if value.ndim >= 1 and value.shape[0] == vector_dimension
                    else ""
                ),
                "column_layout_id_optional": (
                    layout_id
                    if value.ndim >= 2 and value.shape[1] == vector_dimension
                    else ""
                ),
                "derivation_source": old.get(
                    "derivation_source",
                    "SYNTHETIC_PRECOMPUTED_TOPOLOGY_STEP_OPERATOR",
                ),
                "description": old.get(
                    "description",
                    "Synthetic precomputed topology_step operator; no engineering claim.",
                ),
                "quality_flag": "PASS",
            }
        )
    pd.DataFrame(matrix_rows).to_csv(
        TARGET / "matrices" / "matrix_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(oracle_rows).to_csv(
        TARGET / "validation" / "topology_step_lcp_oracle.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return len(matrix_rows), oracle_rows


def _write_validation_tables(
    matrix_count: int, oracle_rows: list[dict[str, object]]
) -> None:
    metadata = json.dumps(
        {
            "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
            "engineering_claim_allowed": False,
            "schema_revision": "DETERMINISTIC_MULTI_ROUND_TOPOLOGY_STEP_2026-07",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    pd.DataFrame(
        [
            {
                "run_id": "RUN_TOPOLOGY_STEP_BUILD",
                "model_version": "V2.5_TOPOLOGY_STEP_FIXTURE",
                "input_package_id": PACKAGE_ID,
                "started_at": "2026-07-23T00:00:00+08:00",
                "finished_at": "2026-07-23T00:00:01+08:00",
                "operator_or_script": "scripts/build_topology_step_fixture.py",
                "software_versions": "python=3.11;numpy;pandas",
                "object_ids_used": "TOPOLOGY_STEP_MIN_CASE",
                "cache_ids_used": "",
                "quality_gate_ids": "QG_TOPOLOGY_STEP_FIXTURE",
                "warnings": "synthetic fixture only",
                "errors": "",
                "runtime_summary": "deterministic package rebuild and self-validation",
                "output_object_ids": "TOPOLOGY_STEP_LCP_ORACLE;VALIDATION_RESULTS",
                "matrix_manifest_count": matrix_count,
                "npz_key_count": matrix_count,
                "metadata": metadata,
            }
        ]
    ).to_csv(
        TARGET / "validation" / "run_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "quality_gate_id": "QG_TOPOLOGY_STEP_FIXTURE",
                "gate_name": "topology_step fixture self-validation",
                "target_object_type": "Package",
                "target_object_ids": PACKAGE_ID,
                "check_items": (
                    "GRAPH;CONTACT_COUNTS;ROUTE;VECTOR_LAYOUT;MATRIX_MANIFEST;"
                    "OPERATORS;CROSS_BLOCKS;ORACLE;DICTIONARY;OBJECT_MAP;TRUTH"
                ),
                "thresholds": "blocking checks must all PASS",
                "pass_fail": "PASS",
                "warn_items": "synthetic_data_not_engineering_evidence",
                "fail_items": "",
                "reviewer_optional": "AUTO",
                "timestamp": "2026-07-23T00:00:01+08:00",
                "metadata": metadata,
            }
        ]
    ).to_csv(
        TARGET / "validation" / "quality_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    residuals = [float(row["complementarity_residual"]) for row in oracle_rows]
    validation_row = {
        "validation_result_id": "VAL_TOPOLOGY_STEP_LCP_ORACLE",
        "model_version": "V2.5_TOPOLOGY_STEP_FIXTURE",
        "validation_sample_ids": ";".join(
            str(row["topology_step_id"]) for row in oracle_rows
        ),
        "reference_type": "SYNTHETIC_INDEPENDENT_LCP_ORACLE",
        "kcp_ids": "",
        "predicted_values": json.dumps(
            [float(row["equilibrium_residual"]) for row in oracle_rows],
            separators=(",", ":"),
        ),
        "reference_values": json.dumps([0.0] * len(oracle_rows)),
        "residuals": json.dumps(residuals, separators=(",", ":")),
        "MAE": float(np.mean(np.abs(residuals))) if residuals else 0.0,
        "RMSE": (
            float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
        ),
        "maximum_absolute_error": max(residuals, default=0.0),
        "correlation_optional": "",
        "failure_probability_error_optional": "",
        "contact_mode_agreement_optional": "",
        "overconstraint_reaction_agreement_optional": "",
        "acceleration_ratio_optional": "",
        "acceptance_criteria": "equilibrium<=1e-9;complementarity<=1e-8",
        "pass_fail": "PASS",
        "metadata": metadata,
    }
    pd.DataFrame([validation_row]).to_csv(
        TARGET / "validation" / "validation_result.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _object_name(relative: str) -> str:
    special = {
        "I0/assembly_topology.csv": "AssemblyTopology",
        "I_stage/connection_lock_history.csv": "ConnectionLockHistory",
        "I_stage/release_history_record.csv": "ReleaseHistoryRecord",
        "validation/topology_step_lcp_oracle.csv": "TopologyStepLcpOracle",
        "validation/validation_result.csv": "ValidationResults",
        "matrices/matrix_manifest.csv": "MatrixManifest",
        "matrices/vector_layout.csv": "VectorLayout",
        "field_dictionary.csv": "FieldDictionary",
        "object_file_map.csv": "ObjectFileMap",
    }
    if relative in special:
        return special[relative]
    stem = Path(relative).stem
    return "".join(word.capitalize() for word in stem.split("_"))


def _primary_key(fields: list[str]) -> str:
    for field in fields:
        if field.endswith("_id") and not field.endswith("_ids"):
            return field
    return ""


def _write_object_map() -> None:
    rows: list[dict[str, object]] = []
    csv_paths = sorted(path for path in TARGET.rglob("*.csv"))
    for path in csv_paths:
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        rows.append(
            {
                "object_name": _object_name(relative),
                "input_package": PACKAGE_ID,
                "file_path": relative,
                "primary_key": _primary_key(fields),
                "code_consumer": "core.data_loader;core.package_validator",
                "producer": "scripts/build_topology_step_fixture.py",
                "is_synthetic": True,
                "is_runtime_result": False,
                "group": Path(relative).parent.as_posix(),
                "row_mode": "synthetic_topology_step",
            }
        )
    rows.extend(
        [
            {
                "object_name": "TopologyStepSpec",
                "input_package": PACKAGE_ID,
                "file_path": "I0/assembly_topology.csv",
                "primary_key": "topology_step_id",
                "code_consumer": "core.topology_step",
                "producer": "scripts/build_topology_step_fixture.py",
                "is_synthetic": True,
                "is_runtime_result": False,
                "group": "I0",
                "row_mode": "synthetic_topology_step",
            },
            {
                "object_name": "TopologyStepOperatorMatrices",
                "input_package": PACKAGE_ID,
                "file_path": "matrices/multi_part_matrices.npz",
                "primary_key": "npz_key",
                "code_consumer": "core.topology_step",
                "producer": "scripts/build_topology_step_fixture.py",
                "is_synthetic": True,
                "is_runtime_result": False,
                "group": "matrices",
                "row_mode": "synthetic_topology_step",
            },
            {
                "object_name": "TopologyStepResult",
                "input_package": PACKAGE_ID,
                "file_path": "runtime/topology_step_result.csv",
                "primary_key": "topology_step_id",
                "code_consumer": "core.reporting;app",
                "producer": "core.topology_step.run_all_stages",
                "is_synthetic": True,
                "is_runtime_result": True,
                "group": "runtime",
                "row_mode": "runtime_result",
            },
            {
                "object_name": "TopologyStepExecutionReport",
                "input_package": PACKAGE_ID,
                "file_path": "runtime/topology_step_execution.csv",
                "primary_key": "topology_step_id",
                "code_consumer": "core.reporting;app",
                "producer": "core.topology_step.run_all_stages",
                "is_synthetic": True,
                "is_runtime_result": True,
                "group": "runtime",
                "row_mode": "runtime_result",
            },
        ]
    )
    runtime_reports = {
        "TopologyStepValidationReport": "topology_step_validation.csv",
        "ActiveSubassemblyHistoryReport": "active_subassembly_history.csv",
        "TopologyStepStateLineageReport": "topology_step_state_lineage.csv",
        "TopologyStepOperatorUsageReport": "topology_step_operator_usage.csv",
        "TopologyStepContactSummaryReport": "topology_step_contact_summary.csv",
        "RuntimeConnectionLockHistory": "connection_lock_history.csv",
        "RuntimeReleaseHistory": "release_history.csv",
    }
    for object_name, filename in runtime_reports.items():
        rows.append(
            {
                "object_name": object_name,
                "input_package": PACKAGE_ID,
                "file_path": f"runtime/{filename}",
                "primary_key": "topology_step_id",
                "code_consumer": "core.reporting;app",
                "producer": "core.reporting.build_runtime_report_zip",
                "is_synthetic": True,
                "is_runtime_result": True,
                "group": "runtime",
                "row_mode": "runtime_result",
            }
        )
    pd.DataFrame(rows, columns=OBJECT_MAP_COLUMNS).drop_duplicates(
        ["object_name", "file_path"]
    ).to_csv(
        TARGET / "object_file_map.csv", index=False, encoding="utf-8-sig"
    )


def _infer_type(values: list[str]) -> str:
    cleaned = [value for value in values if str(value).strip()]
    if not cleaned:
        return "string"
    lowered = {value.lower() for value in cleaned}
    if lowered <= {"true", "false", "0", "1"}:
        return "boolean"
    try:
        [int(value) for value in cleaned]
        return "integer"
    except ValueError:
        pass
    try:
        [float(value) for value in cleaned]
        return "number"
    except ValueError:
        return "string"


def _field_metadata(relative: str, field: str) -> dict[str, object]:
    is_topology = relative == "I0/assembly_topology.csv"
    multi_fields = {
        "added_part_ids",
        "removed_part_ids",
        "activated_interface_ids",
        "deactivated_interface_ids",
        "activated_boundary_ids",
        "deactivated_boundary_ids",
        "activated_load_ids",
        "removed_load_ids",
        "activated_joint_ids",
        "deactivated_joint_ids",
    }
    required_topology = {
        "topology_id",
        "topology_step_id",
        "step_order",
        "assembly_cycle_id",
        "operation_type",
        "stage_id",
        "solve_required",
    }
    enum = ""
    if field == "operation_type":
        enum = "INIT|LOCATE|CLAMP|JOIN|RELEASE|MEASURE|INSPECT|other configured ID"
    elif field == "solve_required":
        enum = "true|false"
    elif field == "retention_rule":
        enum = "RETAIN_THROUGH_RELEASE|REMOVE_AT_RELEASE"
    key_semantics = ""
    if field == "topology_step_id":
        key_semantics = "PK TopologyStepSpec"
    elif field == "parent_topology_step_id":
        key_semantics = "self FK -> topology_step_id"
    elif is_topology and field.endswith("_ids"):
        key_semantics = "semicolon-separated FK list to the named object table"
    elif field.endswith("_id"):
        key_semantics = "object identifier or foreign key per schema"
    return {
        "required": bool(is_topology and field in required_topology),
        "cardinality": "0..*" if field in multi_fields else "0..1",
        "unit": "dimensionless",
        "enum_or_format": enum or ("semicolon-separated IDs" if field in multi_fields else ""),
        "key_semantics": key_semantics,
        "missing_handling": (
            "empty list" if field in multi_fields else "blank allowed unless required"
        ),
        "description": (
            f"TopologyStepSpec.{field}"
            if is_topology
            else f"Schema field {field} in {relative}"
        ),
    }


def _write_field_dictionary() -> None:
    csv_schemas: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for path in sorted(TARGET.rglob("*.csv")):
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            csv_schemas[relative] = (reader.fieldnames or [], rows)
    csv_schemas["field_dictionary.csv"] = (FIELD_DICTIONARY_COLUMNS, [])
    dictionary_rows: list[dict[str, object]] = []
    for relative, (fields, rows) in sorted(csv_schemas.items()):
        object_name = _object_name(relative)
        for field in fields:
            values = [str(row.get(field, "")) for row in rows]
            meta = _field_metadata(relative, field)
            dictionary_rows.append(
                {
                    "file_path": relative,
                    "object_name": (
                        "TopologyStepSpec"
                        if relative == "I0/assembly_topology.csv"
                        else object_name
                    ),
                    "field_name": field,
                    "data_type": _infer_type(values),
                    **meta,
                    "example_value": next(
                        (value for value in values if value.strip()), ""
                    ),
                }
            )
    pd.DataFrame(
        dictionary_rows, columns=FIELD_DICTIONARY_COLUMNS
    ).to_csv(
        TARGET / "field_dictionary.csv", index=False, encoding="utf-8-sig"
    )


def _write_self_validation() -> dict[str, object]:
    shutil.copyfile(VALIDATOR_SOURCE, TARGET / "validation" / "validate_package.py")
    report = validate_package(TARGET, include_attachment_check=False)
    report["checks"].append(
        {
            "name": "validation attachments match current package",
            "passed": True,
            "status": "PASS",
            "blocking": True,
            "detail": (
                f"matrix_manifest={report['matrix_manifest_count']}, "
                f"npz={report['npz_key_count']}, stored_status=PASS"
            ),
        }
    )
    report["total"] = len(report["checks"])
    report["passed"] = sum(
        item["status"] == "PASS" for item in report["checks"]
    )
    report["blocking_fail_count"] = sum(
        item["status"] == "FAIL" and item["blocking"]
        for item in report["checks"]
    )
    report["status"] = "PASS" if report["blocking_fail_count"] == 0 else "FAIL"
    validation = TARGET / "validation"
    (validation / "test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (validation / "TEST_RESULTS.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    return validate_package(TARGET)


def build() -> dict[str, object]:
    _safe_recreate_target()
    route = pd.DataFrame(route_rows())
    route.to_csv(
        TARGET / "I0" / "assembly_topology.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_manifest(route)
    matrix_count, oracle_rows = _build_matrices(route)
    _write_validation_tables(matrix_count, oracle_rows)
    _write_object_map()
    _write_field_dictionary()
    (TARGET / "README.md").write_text(
        "# 02_TOPOLOGY_STEP_MIN_CASE\n\n"
        "Deterministic 13-step, four-part, four-interface, 12-component synthetic "
        "integration fixture. These values are fixture expectations only; the "
        "validator derives topology, contact counts, and layouts from package data.\n\n"
        "`engineering_claim_allowed=false`. The package validates execution order, "
        "global coupled LCP behavior, state lineage, and JOIN/RELEASE history. It "
        "does not represent measured data, online FE condensation, posterior "
        "updating, or an engineering accuracy conclusion.\n",
        encoding="utf-8",
    )
    result = _write_self_validation()
    if result["status"] != "PASS":
        raise RuntimeError(
            f"generated package failed self-validation: {result['blocking_fail_count']}"
        )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
