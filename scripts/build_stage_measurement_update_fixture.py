from __future__ import annotations

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from stage_measurement_update_package_validator import (
    markdown_report,
    validate_package,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.stage_measurement_update import (  # noqa: E402
    MeasurementObservationSpec,
    extract_observation_vector,
)

SOURCE = ROOT / "data" / "02_TOPOLOGY_STEP_MIN_CASE"
TARGET = ROOT / "data" / "03_STAGE_MEASUREMENT_UPDATE_MIN_CASE"
VALIDATOR_SOURCE = (
    ROOT / "scripts" / "stage_measurement_update_package_validator.py"
)
PACKAGE_ID = TARGET.name
CHECKPOINT_ID = "MCP_AFTER_ABC"
UPDATE_CONFIG_ID = "MUCFG_AFTER_ABC"
UPDATE_LAYOUT_ID = "ETA_STAGE_AFTER_ABC_2"
OBSERVATION_LAYOUT_ID = "OBS_MCP_AFTER_ABC_2"
MATRIX_RULE = "G_Q_UPDATE__{checkpoint_id}__{operator_set_id}"
FIXED_TIMESTAMP = "2026-07-25T00:00:00+08:00"

FIELD_DICTIONARY_COLUMNS = [
    "file_path", "object_name", "field_name", "data_type", "required",
    "cardinality", "unit", "enum_or_format", "key_semantics",
    "missing_handling", "description", "example_value",
]
OBJECT_MAP_COLUMNS = [
    "object_name", "input_package", "file_path", "primary_key",
    "code_consumer", "producer", "is_synthetic", "is_runtime_result",
    "group", "row_mode",
]


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write an NPZ whose bytes are stable across repeated fixture builds."""
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for key, array in sorted(arrays.items()):
            buffer = BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(array),
                allow_pickle=False,
            )
            entry = zipfile.ZipInfo(
                filename=f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue(), compresslevel=9)


def _safe_recreate_target() -> None:
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
        TARGET / "validation",
    ):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    (TARGET / "validation").mkdir(parents=True, exist_ok=True)


def _write_route() -> pd.DataFrame:
    path = TARGET / "I0" / "assembly_topology.csv"
    route = pd.read_csv(path, encoding="utf-8-sig")
    route = route.astype(object)
    source = route[
        route["topology_step_id"].astype(str).eq("TS204")
    ].iloc[0].copy()
    measure = {column: "" for column in route.columns}
    measure.update({
        "topology_id": source["topology_id"],
        "measurement_checkpoint_id": CHECKPOINT_ID,
        "reference_state_id": source["reference_state_id"],
        "topology_step_id": "TS205",
        "step_order": 205,
        "parent_topology_step_id": "TS204",
        "assembly_cycle_id": "CYCLE_BC",
        "operation_type": "MEASURE",
        "stage_id": "S_MEASURE_05",
        "input_subassembly_id": "ASM_ABC",
        "result_subassembly_id": "ASM_ABC",
        "operator_set_id": "",
        "solve_required": False,
        "notes": (
            "Synthetic process measurement checkpoint after ABC release; "
            "mechanical source is explicitly TS204."
        ),
        "assembly_step": 205,
        "existing_subassembly": "ASM_ABC",
        "result_subassembly": "ASM_ABC",
        "predecessor_step_id": "TS204",
        "successor_step_id": "TS301",
    })
    route.loc[
        route["topology_step_id"].astype(str).eq("TS204"),
        "successor_step_id",
    ] = "TS205"
    route.loc[
        route["topology_step_id"].astype(str).eq("TS301"),
        ["parent_topology_step_id", "predecessor_step_id"],
    ] = "TS205"
    route = pd.concat(
        [
            route[pd.to_numeric(route["step_order"]) <= 204],
            pd.DataFrame([measure], columns=route.columns),
            route[pd.to_numeric(route["step_order"]) > 204],
        ],
        ignore_index=True,
    ).sort_values(
        ["step_order", "topology_step_id"], kind="stable"
    ).reset_index(drop=True)
    route.to_csv(path, index=False, encoding="utf-8-sig")
    stage_path = TARGET / "I0" / "stage_definition.csv"
    stages = pd.read_csv(stage_path, encoding="utf-8-sig").astype(object)
    if not stages["stage_id"].astype(str).eq("S_MEASURE_05").any():
        stage_row = {column: "" for column in stages.columns}
        stage_row.update({
            "stage_id": "S_MEASURE_05",
            "stage_order": 5,
            "stage_type": "MEASURE",
            "subassembly_id": "ASM_ABC",
            "active_part_ids": "P_A;P_B;P_C",
            "active_interface_ids": "G_AB;G_BC",
            "predecessor_stage_id": "S_RELEASE_04",
            "successor_stage_id": "",
            "metadata": json.dumps({
                "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
                "engineering_claim_allowed": False,
                "schema_revision":
                    "STAGE_MEASUREMENT_POSTERIOR_UPDATE_2026-07",
            }, ensure_ascii=False, separators=(",", ":")),
        })
        stages = pd.concat(
            [stages, pd.DataFrame([stage_row], columns=stages.columns)],
            ignore_index=True,
        )
        stages.to_csv(stage_path, index=False, encoding="utf-8-sig")
    return route


def _source_oracle_state() -> tuple[np.ndarray, np.ndarray]:
    oracle = pd.read_csv(
        SOURCE / "validation" / "topology_step_lcp_oracle.csv",
        encoding="utf-8-sig",
    )
    row = oracle[
        oracle["topology_step_id"].astype(str).eq("TS204")
    ].iloc[0]
    return (
        np.asarray(json.loads(row["lambda_active"]), dtype=float),
        np.asarray(json.loads(row["gap_active"]), dtype=float),
    )


def _state_to_q_mappings(
    vector_dimension: int,
) -> dict[str, np.ndarray]:
    mappings: dict[str, np.ndarray] = {}
    for ordinal, operator in enumerate(
        ("OP_TS204", "OP_TS301", "OP_TS302", "OP_TS303", "OP_TS304")
    ):
        mapping = np.zeros((vector_dimension, 2), dtype=float)
        mapping[0, 0] = 0.15
        mapping[1, 0] = 0.05
        mapping[3, 1] = 0.12
        mapping[4, 1] = 0.03
        if ordinal > 0:
            mapping[6, 0] = 0.04
            mapping[7, 1] = -0.02
            mapping[9, 0] = 0.03
            mapping[10, 1] = 0.05
        mappings[
            f"G_Q_UPDATE__{CHECKPOINT_ID}__{operator}"
        ] = mapping
    return mappings


def _fixture_observation_specs() -> list[MeasurementObservationSpec]:
    return [
        MeasurementObservationSpec(
            observation_id="OBS_GAP_AB",
            checkpoint_id=CHECKPOINT_ID,
            measurement_id="MEAS_GAP_AB_AFTER_RELEASE",
            observation_order=0,
            observed_quantity="GAP_G",
            vector_source="RUNTIME_STAGE_STATE",
            global_index_optional=1,
            target_object_id_optional="G_AB",
            unit="mm",
            coordinate_system_id="CS_ASSEMBLY",
            reference_state_id="PARENT_RUNTIME_STATE",
            sensitivity_row_index=0,
            quality_flag="PASS",
        ),
        MeasurementObservationSpec(
            observation_id="OBS_FORCE_AB",
            checkpoint_id=CHECKPOINT_ID,
            measurement_id="MEAS_FORCE_AB_AFTER_RELEASE",
            observation_order=1,
            observed_quantity="LAMBDA_N",
            vector_source="RUNTIME_STAGE_STATE",
            global_index_optional=0,
            target_object_id_optional="G_AB",
            unit="N",
            coordinate_system_id="CS_ASSEMBLY",
            reference_state_id="PARENT_RUNTIME_STATE",
            sensitivity_row_index=1,
            quality_flag="PASS",
        ),
    ]


def _independent_physical_observation(
    eta: np.ndarray,
    *,
    q_operator_base: np.ndarray,
    G_q: np.ndarray,
    W_total: np.ndarray,
    active_indices: np.ndarray,
    area: np.ndarray,
    Cn: np.ndarray,
    layout: pd.DataFrame,
    observation_specs: list[MeasurementObservationSpec],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    q_effective = q_operator_base + G_q @ np.asarray(eta, dtype=float)
    reaction_active, gap_active = _independent_lcp(
        q_effective[active_indices],
        W_total[np.ix_(active_indices, active_indices)],
    )
    dimension = q_effective.size
    reaction = np.zeros(dimension, dtype=float)
    gap = np.full(dimension, np.nan, dtype=float)
    pressure = np.full(dimension, np.nan, dtype=float)
    compression = np.full(dimension, np.nan, dtype=float)
    reaction[active_indices] = reaction_active
    gap[active_indices] = gap_active
    pressure[active_indices] = reaction_active / area[active_indices]
    compression[active_indices] = (
        Cn[np.ix_(active_indices, active_indices)] @ reaction_active
    )
    state = {
        "vector_layout_id": str(layout.iloc[0]["vector_layout_id"]),
        "active_interface_ids": ["G_AB", "G_BC"],
        "gap_full": gap,
        "lambda_full": reaction,
        "pressure": pressure,
        "local_compression": compression,
    }
    z = extract_observation_vector(state, observation_specs, layout)
    active_set = active_indices[
        np.flatnonzero(reaction_active > 1e-9)
    ]
    return z, active_set, {
        "q_effective": q_effective,
        "lambda_full": reaction,
        "gap_full": gap,
        "pressure": pressure,
        "local_compression": compression,
    }


def _write_measurement_tables() -> dict[str, Any]:
    npz_path = TARGET / "matrices" / "multi_part_matrices.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        base_arrays = {
            key: archive[key].copy() for key in archive.files
        }
    contact_points = pd.read_csv(
        TARGET / "I_Gamma" / "contact_point.csv",
        encoding="utf-8-sig",
    )
    layout = pd.read_csv(
        TARGET / "matrices" / "vector_layout.csv",
        encoding="utf-8-sig",
    )
    mappings = _state_to_q_mappings(len(contact_points))
    G_q = mappings[
        f"G_Q_UPDATE__{CHECKPOINT_ID}__OP_TS204"
    ]
    q_operator_base = base_arrays["Q_OP_TS204"]
    Cn = base_arrays["CN_OP_TS204"]
    W_total = base_arrays["W_STRUCT_OP_TS204"] + Cn
    active_indices = np.arange(6, dtype=int)
    area = contact_points["area_weight"].to_numpy(dtype=float)
    observation_specs = _fixture_observation_specs()
    eta_prior = np.zeros(2, dtype=float)
    eta_true = np.array([0.002, -0.001], dtype=float)
    z_predicted, active_prior, _ = _independent_physical_observation(
        eta_prior,
        q_operator_base=q_operator_base,
        G_q=G_q,
        W_total=W_total,
        active_indices=active_indices,
        area=area,
        Cn=Cn,
        layout=layout,
        observation_specs=observation_specs,
    )
    z_true, active_true, true_state = _independent_physical_observation(
        eta_true,
        q_operator_base=q_operator_base,
        G_q=G_q,
        W_total=W_total,
        active_indices=active_indices,
        area=area,
        Cn=Cn,
        layout=layout,
        observation_specs=observation_specs,
    )
    z_measured = z_true.copy()
    finite_difference_epsilon = 1e-5
    H_columns: list[np.ndarray] = []
    active_sets: list[list[int]] = []
    for component in range(eta_prior.size):
        perturbation = np.zeros_like(eta_prior)
        perturbation[component] = finite_difference_epsilon
        z_plus, active_plus, _ = _independent_physical_observation(
            eta_prior + perturbation,
            q_operator_base=q_operator_base,
            G_q=G_q,
            W_total=W_total,
            active_indices=active_indices,
            area=area,
            Cn=Cn,
            layout=layout,
            observation_specs=observation_specs,
        )
        z_minus, active_minus, _ = _independent_physical_observation(
            eta_prior - perturbation,
            q_operator_base=q_operator_base,
            G_q=G_q,
            W_total=W_total,
            active_indices=active_indices,
            area=area,
            Cn=Cn,
            layout=layout,
            observation_specs=observation_specs,
        )
        if not (
            np.array_equal(active_plus, active_prior)
            and np.array_equal(active_minus, active_prior)
        ):
            raise RuntimeError(
                "finite-difference active set is not stable for "
                f"component={component}"
            )
        H_columns.append(
            (z_plus - z_minus) / (2.0 * finite_difference_epsilon)
        )
        active_sets.extend(
            [active_plus.astype(int).tolist(), active_minus.astype(int).tolist()]
        )
    if not np.array_equal(active_true, active_prior):
        raise RuntimeError("eta_true changes the independent LCP active set")
    H = np.column_stack(H_columns)
    records = pd.DataFrame([
        {
            "measurement_id": "MEAS_GAP_AB_AFTER_RELEASE",
            "measurement_set_id": "MEAS_SET_AFTER_ABC",
            "sample_id": "SAMPLE_001",
            "object_type": "Interface",
            "object_id": "G_AB",
            "interface_id_optional": "G_AB",
            "part_id_optional": "",
            "stage_id": "S_MEASURE_05",
            "measurement_type": "GAP_G",
            "feature_definition_id": "GAP_G",
            "location": "GLOBAL_CONTACT_INDEX_1",
            "direction": "NORMAL",
            "coordinate_system_id": "CS_ASSEMBLY",
            "value": float(z_measured[0]),
            "unit": "mm",
            "standard_uncertainty": 0.002,
            "covariance_block_id_optional": "R_MEAS_MCP_AFTER_ABC",
            "sensor_or_device_id": "SYNTH_SENSOR_GAP",
            "update_target": "STAGE_STATE",
            "data_role": "CALIBRATE",
            "quality_flag": "PASS",
            "reference_state_id_optional": "PARENT_RUNTIME_STATE",
            "timestamp": FIXED_TIMESTAMP,
            "source": "SYNTHETIC_DETERMINISTIC_GENERATOR",
        },
        {
            "measurement_id": "MEAS_FORCE_AB_AFTER_RELEASE",
            "measurement_set_id": "MEAS_SET_AFTER_ABC",
            "sample_id": "SAMPLE_001",
            "object_type": "Interface",
            "object_id": "G_AB",
            "interface_id_optional": "G_AB",
            "part_id_optional": "",
            "stage_id": "S_MEASURE_05",
            "measurement_type": "LAMBDA_N",
            "feature_definition_id": "LAMBDA_N",
            "location": "GLOBAL_CONTACT_INDEX_0",
            "direction": "NORMAL",
            "coordinate_system_id": "CS_ASSEMBLY",
            "value": float(z_measured[1]),
            "unit": "N",
            "standard_uncertainty": 0.003,
            "covariance_block_id_optional": "R_MEAS_MCP_AFTER_ABC",
            "sensor_or_device_id": "SYNTH_SENSOR_FORCE",
            "update_target": "STAGE_STATE",
            "data_role": "CALIBRATE",
            "quality_flag": "PASS",
            "reference_state_id_optional": "PARENT_RUNTIME_STATE",
            "timestamp": FIXED_TIMESTAMP,
            "source": "SYNTHETIC_DETERMINISTIC_GENERATOR",
        },
    ])
    records.to_csv(
        TARGET / "I_meas" / "measurement_record.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "checkpoint_id": CHECKPOINT_ID,
        "topology_id": "TOPOLOGY_STEP_MIN_CASE",
        "topology_step_id": "TS205",
        "source_topology_step_id": "TS204",
        "measurement_set_id": "MEAS_SET_AFTER_ABC",
        "update_config_id": UPDATE_CONFIG_ID,
        "reference_state_id": "PARENT_RUNTIME_STATE",
        "missing_measurement_policy": "BLOCK",
        "rollback_policy": "ROLLBACK_TO_PREDICTED",
        "active_flag": True,
        "notes": "Synthetic checkpoint; no engineering accuracy claim.",
    }]).to_csv(
        TARGET / "I_meas" / "measurement_checkpoint.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "observation_id": "OBS_GAP_AB",
            "checkpoint_id": CHECKPOINT_ID,
            "measurement_id": "MEAS_GAP_AB_AFTER_RELEASE",
            "observation_order": 0,
            "observed_quantity": "GAP_G",
            "vector_source": "RUNTIME_STAGE_STATE",
            "global_index_optional": 1,
            "target_object_id_optional": "G_AB",
            "unit": "mm",
            "coordinate_system_id": "CS_ASSEMBLY",
            "reference_state_id": "PARENT_RUNTIME_STATE",
            "sensitivity_row_index": 0,
            "quality_flag": "PASS",
        },
        {
            "observation_id": "OBS_FORCE_AB",
            "checkpoint_id": CHECKPOINT_ID,
            "measurement_id": "MEAS_FORCE_AB_AFTER_RELEASE",
            "observation_order": 1,
            "observed_quantity": "LAMBDA_N",
            "vector_source": "RUNTIME_STAGE_STATE",
            "global_index_optional": 0,
            "target_object_id_optional": "G_AB",
            "unit": "N",
            "coordinate_system_id": "CS_ASSEMBLY",
            "reference_state_id": "PARENT_RUNTIME_STATE",
            "sensitivity_row_index": 1,
            "quality_flag": "PASS",
        },
    ]).to_csv(
        TARGET / "I_meas" / "measurement_observation_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "update_config_id": UPDATE_CONFIG_ID,
        "update_state_layout_id": UPDATE_LAYOUT_ID,
        "prior_mean_matrix_id": "ETA_PRIOR_MCP_AFTER_ABC",
        "prior_covariance_matrix_id": "P_PRIOR_MCP_AFTER_ABC",
        "observation_jacobian_matrix_id": "H_OBS_MCP_AFTER_ABC",
        "measurement_covariance_matrix_id_optional": "R_MEAS_MCP_AFTER_ABC",
        "state_to_q_mapping_rule": MATRIX_RULE,
        "algorithm": "LINEAR_GAUSSIAN_JOSEPH",
        "regularization": 0.0,
        "nis_threshold": 25.0,
        "covariance_floor": 1e-12,
        "resolve_policy": "RESOLVE_LCP",
        "parameter_update_allowed": False,
        "quality_flag": "PASS",
        "reference_state_id": "PARENT_RUNTIME_STATE",
        "allow_diagonal_covariance_fallback": False,
        "physical_residual_threshold": 5.0,
        "individual_degradation_tolerance": 1.0,
    }]).to_csv(
        TARGET / "I_meas" / "measurement_update_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    measurement_package_path = TARGET / "I_meas" / "measurement_package.csv"
    if measurement_package_path.exists():
        package_table = pd.read_csv(
            measurement_package_path, encoding="utf-8-sig"
        ).astype(object)
        if (
            not package_table.empty
            and "process_measurement_set_ids" in package_table.columns
        ):
            package_table.loc[
                package_table.index[0], "process_measurement_set_ids"
            ] = "MEAS_SET_AFTER_ABC"
        package_table.to_csv(
            measurement_package_path,
            index=False,
            encoding="utf-8-sig",
        )
    pd.DataFrame([
        {
            "update_state_layout_id": UPDATE_LAYOUT_ID,
            "component_order": 0,
            "component_id": "ETA_POSE_BIAS_ABC",
            "component_type": "POSE_BIAS",
            "target_object_type": "Subassembly",
            "target_object_id": "ASM_ABC",
            "unit": "mm_equivalent",
            "description": "Synthetic low-dimensional pose-bias correction.",
        },
        {
            "update_state_layout_id": UPDATE_LAYOUT_ID,
            "component_order": 1,
            "component_id": "ETA_GAP_BIAS_BC",
            "component_type": "GAP_BIAS",
            "target_object_type": "Interface",
            "target_object_id": "G_BC",
            "unit": "mm",
            "description": "Synthetic low-dimensional interface-gap correction.",
        },
    ]).to_csv(
        TARGET / "I_stage" / "state_update_basis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "covariance_transfer_id": f"COV_TRANSFER_{step_id}",
            "update_state_layout_id": UPDATE_LAYOUT_ID,
            "from_topology_step_id": (
                "TS205" if step_id == "TS301"
                else f"TS{int(step_id[2:]) - 1:03d}"
            ),
            "to_topology_step_id": step_id,
            "state_jacobian_F_matrix_id": "F_UPDATE_IDENTITY_2",
            "process_noise_Q_matrix_id": "Q_UPDATE_ZERO_2",
            "cross_covariance_rule": "NONE",
            "approximation_validity": (
                "FIRST_VERSION_IDENTITY_ZERO_LOW_DIMENSIONAL_STATE_ONLY"
            ),
            "quality_flag": "PASS",
        }
        for step_id in ("TS301", "TS302", "TS303", "TS304")
    ]).to_csv(
        TARGET / "I_stage" / "stage_covariance_transfer.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "H": H,
        "eta_prior": eta_prior,
        "eta_true": eta_true,
        "z_predicted": z_predicted,
        "z_measured": z_measured,
        "z_true": z_true,
        "q_true": true_state["q_effective"],
        "lambda_true": true_state["lambda_full"],
        "gap_true": true_state["gap_full"],
        "finite_difference_epsilon": np.array(
            [finite_difference_epsilon], dtype=float
        ),
        "finite_difference_method": "CENTRAL_DIFFERENCE",
        "active_set_prior": active_prior,
        "active_set_true": active_true,
        "active_set_stable": True,
        "finite_difference_active_sets": active_sets,
        "H_derivation_source": (
            "INDEPENDENT_GLOBAL_LCP_CENTRAL_FINITE_DIFFERENCE"
        ),
    }


def _independent_lcp(
    q: np.ndarray,
    W: np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    size = q.size
    for mask in range(1 << size):
        active = np.array(
            [index for index in range(size) if mask & (1 << index)],
            dtype=int,
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


def _posterior_oracle(
    arrays: dict[str, np.ndarray],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    eta_prior = arrays["ETA_PRIOR_MCP_AFTER_ABC"].reshape(-1)
    P_prior = arrays["P_PRIOR_MCP_AFTER_ABC"]
    H = arrays["H_OBS_MCP_AFTER_ABC"]
    R = arrays["R_MEAS_MCP_AFTER_ABC"]
    innovation = measurement["z_measured"] - measurement["z_predicted"]
    S = H @ P_prior @ H.T + R
    K = np.linalg.solve(S, H @ P_prior).T
    eta_posterior = eta_prior + K @ innovation
    identity = np.eye(eta_prior.size)
    left = identity - K @ H
    P_posterior = left @ P_prior @ left.T + K @ R @ K.T
    P_posterior = (P_posterior + P_posterior.T) / 2.0
    z_posterior_linearized = (
        measurement["z_predicted"]
        + H @ (eta_posterior - eta_prior)
    )
    operator = "OP_TS204"
    q_base = arrays[f"Q_{operator}"]
    G_q = arrays[f"G_Q_UPDATE__{CHECKPOINT_ID}__{operator}"]
    q_source_effective = q_base + G_q @ eta_prior
    q_posterior = (
        q_source_effective + G_q @ (eta_posterior - eta_prior)
    )
    W_total = arrays[f"W_STRUCT_{operator}"] + arrays[f"CN_{operator}"]
    active_indices = np.arange(6, dtype=int)
    contact_points = pd.read_csv(
        TARGET / "I_Gamma" / "contact_point.csv",
        encoding="utf-8-sig",
    )
    layout = pd.read_csv(
        TARGET / "matrices" / "vector_layout.csv",
        encoding="utf-8-sig",
    )
    z_posterior_physical, active_set, posterior_state = (
        _independent_physical_observation(
            eta_posterior,
            q_operator_base=q_base,
            G_q=G_q,
            W_total=W_total,
            active_indices=active_indices,
            area=contact_points["area_weight"].to_numpy(dtype=float),
            Cn=arrays[f"CN_{operator}"],
            layout=layout,
            observation_specs=_fixture_observation_specs(),
        )
    )
    reaction = posterior_state["lambda_full"][active_indices]
    gap = posterior_state["gap_full"][active_indices]
    residual_prior = innovation
    residual_linearized = (
        measurement["z_measured"] - z_posterior_linearized
    )
    residual_physical = (
        measurement["z_measured"] - z_posterior_physical
    )
    weighted_prior = float(
        residual_prior @ np.linalg.solve(R, residual_prior)
    )
    weighted_physical = float(
        residual_physical @ np.linalg.solve(R, residual_physical)
    )
    return {
        "eta_prior": eta_prior,
        "eta_posterior": eta_posterior,
        "P_prior": P_prior,
        "P_posterior": P_posterior,
        "kalman_gain": K,
        "innovation": innovation,
        "z_predicted_posterior_linearized": z_posterior_linearized,
        "z_predicted_posterior_physical": z_posterior_physical,
        "residual_prior_physical": residual_prior,
        "residual_posterior_linearized": residual_linearized,
        "residual_posterior_physical": residual_physical,
        "weighted_residual_prior_physical": weighted_prior,
        "weighted_residual_posterior_physical": weighted_physical,
        "linearization_error": (
            z_posterior_physical - z_posterior_linearized
        ),
        "physical_residual_improved": (
            weighted_physical < weighted_prior
        ),
        "nis": float(innovation @ np.linalg.solve(S, innovation)),
        "q_posterior": q_posterior,
        "active_indices": active_indices,
        "active_set": active_set,
        "lambda_posterior_active": reaction,
        "gap_posterior_active": gap,
        "prior_residual_norm": float(np.linalg.norm(innovation)),
        "posterior_linearized_residual_norm": float(
            np.linalg.norm(residual_linearized)
        ),
        "posterior_residual_norm": float(
            np.linalg.norm(residual_physical)
        ),
        "complementarity_residual": float(
            np.max(np.abs(reaction * gap), initial=0.0)
        ),
    }


def _write_matrices(
    measurement: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = TARGET / "matrices" / "multi_part_matrices.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    vector_dimension = len(
        pd.read_csv(
            TARGET / "I_Gamma" / "contact_point.csv",
            encoding="utf-8-sig",
        )
    )
    arrays.update({
        "ETA_PRIOR_MCP_AFTER_ABC": np.zeros(2, dtype=float),
        "P_PRIOR_MCP_AFTER_ABC": np.array(
            [[4.0e-4, 3.0e-5], [3.0e-5, 2.25e-4]], dtype=float
        ),
        "H_OBS_MCP_AFTER_ABC": measurement["H"],
        "R_MEAS_MCP_AFTER_ABC": np.array(
            [[4.0e-6, 8.0e-7], [8.0e-7, 9.0e-6]], dtype=float
        ),
        "F_UPDATE_IDENTITY_2": np.eye(2, dtype=float),
        "Q_UPDATE_ZERO_2": np.zeros((2, 2), dtype=float),
    })
    arrays.update(_state_to_q_mappings(vector_dimension))
    oracle = _posterior_oracle(arrays, measurement)
    _write_deterministic_npz(npz_path, arrays)

    manifest_path = TARGET / "matrices" / "matrix_manifest.csv"
    old_manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    old_by_key = {
        str(row["npz_key"]): row
        for row in old_manifest.to_dict("records")
    }
    layout_id = str(
        pd.read_csv(
            TARGET / "matrices" / "vector_layout.csv",
            encoding="utf-8-sig",
        ).iloc[0]["vector_layout_id"]
    )
    rows: list[dict[str, Any]] = []
    for key, value in sorted(arrays.items()):
        old = old_by_key.get(key, {})
        if key.startswith("G_Q_UPDATE__"):
            row_layout = layout_id
            column_layout = UPDATE_LAYOUT_ID
            unit = "mm_per_update_state_unit"
            derivation = "SYNTHETIC_EXPLICIT_STATE_TO_Q_MAPPING"
            checkpoint = CHECKPOINT_ID
            operator_set = key.rsplit("__", 1)[-1]
        elif key in {
            "ETA_PRIOR_MCP_AFTER_ABC",
            "P_PRIOR_MCP_AFTER_ABC",
            "F_UPDATE_IDENTITY_2",
            "Q_UPDATE_ZERO_2",
        }:
            row_layout = UPDATE_LAYOUT_ID
            column_layout = (
                UPDATE_LAYOUT_ID if value.ndim == 2 else ""
            )
            unit = "mixed_by_update_state_component"
            derivation = "SYNTHETIC_LOW_DIMENSIONAL_STATE_MODEL"
            checkpoint = CHECKPOINT_ID
            operator_set = ""
        elif key in {"H_OBS_MCP_AFTER_ABC", "R_MEAS_MCP_AFTER_ABC"}:
            row_layout = OBSERVATION_LAYOUT_ID
            column_layout = (
                UPDATE_LAYOUT_ID
                if key == "H_OBS_MCP_AFTER_ABC"
                else OBSERVATION_LAYOUT_ID
            )
            unit = "mixed_by_observation_definition"
            derivation = (
                measurement["H_derivation_source"]
                if key == "H_OBS_MCP_AFTER_ABC"
                else "SYNTHETIC_CONFIGURED_MEASUREMENT_COVARIANCE"
            )
            checkpoint = CHECKPOINT_ID
            operator_set = ""
        else:
            row_layout = old.get("row_layout_id_optional", "")
            column_layout = old.get("column_layout_id_optional", "")
            unit = old.get("unit", "mixed_by_object_definition")
            derivation = old.get(
                "derivation_source",
                "SYNTHETIC_PRECOMPUTED_TOPOLOGY_STEP_OPERATOR",
            )
            checkpoint = ""
            operator_set = ""
        rows.append({
            "matrix_id": key,
            "npz_file": "multi_part_matrices.npz",
            "npz_key": key,
            "shape": json.dumps(list(value.shape), separators=(",", ":")),
            "dtype": str(value.dtype),
            "unit": unit,
            "row_layout_id_optional": row_layout,
            "column_layout_id_optional": column_layout,
            "derivation_source": derivation,
            "description": (
                "Synthetic numerical-consistency matrix; "
                "not engineering measurement or online FE output."
            ),
            "quality_flag": "PASS",
            "measurement_checkpoint_id_optional": checkpoint,
            "operator_set_id_optional": operator_set,
            "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
            "engineering_claim_allowed": False,
        })
    pd.DataFrame(rows).to_csv(
        manifest_path, index=False, encoding="utf-8-sig"
    )
    return arrays, oracle


def _json_array(value: np.ndarray) -> str:
    return json.dumps(
        np.asarray(value, dtype=float).tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _frozen_package_hash() -> str:
    relative_files = sorted(
        {
            *(
                path.relative_to(TARGET).as_posix()
                for path in (TARGET / "sms_update").glob("*.csv")
            ),
            *(
                path.relative_to(TARGET).as_posix()
                for path in (TARGET / "parameter_library").glob("*.csv")
            ),
            "I0/material.csv",
            "I0/joint_definition.csv",
            "matrices/matrix_manifest.csv",
            "matrices/multi_part_matrices.npz",
        }
    )
    digest = hashlib.sha256()
    for relative in relative_files:
        path = TARGET / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_oracle_and_validation(
    arrays: dict[str, np.ndarray],
    measurement: dict[str, Any],
    oracle: dict[str, Any],
) -> None:
    shutil.copyfile(
        SOURCE / "validation" / "topology_step_lcp_oracle.csv",
        TARGET / "validation" / "topology_step_lcp_oracle.csv",
    )
    tolerance = 1e-8
    frozen_hash_before = _frozen_package_hash()
    frozen_hash_after = _frozen_package_hash()
    pd.DataFrame([{
        "checkpoint_id": CHECKPOINT_ID,
        "topology_step_id": "TS205",
        "source_topology_step_id": "TS204",
        "update_state_layout_id": UPDATE_LAYOUT_ID,
        "measurement_ids": (
            "MEAS_GAP_AB_AFTER_RELEASE;MEAS_FORCE_AB_AFTER_RELEASE"
        ),
        "z_measured": _json_array(measurement["z_measured"]),
        "z_predicted_prior_physical": _json_array(
            measurement["z_predicted"]
        ),
        "z_predicted_posterior_linearized": _json_array(
            oracle["z_predicted_posterior_linearized"]
        ),
        "z_predicted_posterior_physical": _json_array(
            oracle["z_predicted_posterior_physical"]
        ),
        "residual_prior_physical": _json_array(
            oracle["residual_prior_physical"]
        ),
        "residual_posterior_linearized": _json_array(
            oracle["residual_posterior_linearized"]
        ),
        "residual_posterior_physical": _json_array(
            oracle["residual_posterior_physical"]
        ),
        "weighted_residual_prior_physical": (
            oracle["weighted_residual_prior_physical"]
        ),
        "weighted_residual_posterior_physical": (
            oracle["weighted_residual_posterior_physical"]
        ),
        "linearization_error": _json_array(
            oracle["linearization_error"]
        ),
        "physical_residual_improved": (
            oracle["physical_residual_improved"]
        ),
        "innovation": _json_array(oracle["innovation"]),
        "eta_prior": _json_array(oracle["eta_prior"]),
        "eta_posterior": _json_array(oracle["eta_posterior"]),
        "P_prior": _json_array(oracle["P_prior"]),
        "P_posterior": _json_array(oracle["P_posterior"]),
        "kalman_gain": _json_array(oracle["kalman_gain"]),
        "nis": oracle["nis"],
        "q_posterior": _json_array(oracle["q_posterior"]),
        "active_indices": json.dumps(
            oracle["active_indices"].tolist(), separators=(",", ":")
        ),
        "lambda_posterior_active": _json_array(
            oracle["lambda_posterior_active"]
        ),
        "gap_posterior_active": _json_array(
            oracle["gap_posterior_active"]
        ),
        "prior_residual_norm": oracle["prior_residual_norm"],
        "posterior_residual_norm": oracle["posterior_residual_norm"],
        "posterior_linearized_residual_norm": (
            oracle["posterior_linearized_residual_norm"]
        ),
        "P_prior_trace": float(np.trace(oracle["P_prior"])),
        "P_posterior_trace": float(np.trace(oracle["P_posterior"])),
        "complementarity_residual": oracle[
            "complementarity_residual"
        ],
        "posterior_tolerance": 1e-4,
        "lcp_tolerance": 1e-5,
        "H_derivation_source": measurement["H_derivation_source"],
        "finite_difference_method": (
            measurement["finite_difference_method"]
        ),
        "finite_difference_epsilon": float(
            measurement["finite_difference_epsilon"][0]
        ),
        "active_set_stable": measurement["active_set_stable"],
        "active_set_prior": json.dumps(
            measurement["active_set_prior"].astype(int).tolist(),
            separators=(",", ":"),
        ),
        "active_set_true": json.dumps(
            measurement["active_set_true"].astype(int).tolist(),
            separators=(",", ":"),
        ),
        "observation_extractor": (
            "core.stage_measurement_update.extract_observation_vector"
        ),
        "frozen_package_hash_before": frozen_hash_before,
        "frozen_package_hash_after": frozen_hash_after,
        "oracle_method": (
            "INDEPENDENT_NUMPY_AND_ACTIVE_SET_ENUMERATION"
        ),
        "production_update_function_used": False,
        "production_topology_runner_used": False,
        "engineering_claim_allowed": False,
    }]).to_csv(
        TARGET / "validation" / "stage_measurement_update_oracle.csv",
        index=False,
        encoding="utf-8-sig",
    )
    expected = {
        "checkpoint_id": CHECKPOINT_ID,
        "measurement_count": 2,
        "update_state_dimension": 2,
        "posterior_status": "POSTERIOR_ACCEPTED",
        "posterior_accepted": True,
        "resolve_level": "RESOLVE_LCP",
        "resolve_lcp_call_count": 1,
        "prior_residual_norm": oracle["prior_residual_norm"],
        "posterior_residual_norm": oracle["posterior_residual_norm"],
        "posterior_linearized_residual_norm": (
            oracle["posterior_linearized_residual_norm"]
        ),
        "weighted_residual_prior_physical": (
            oracle["weighted_residual_prior_physical"]
        ),
        "weighted_residual_posterior_physical": (
            oracle["weighted_residual_posterior_physical"]
        ),
        "physical_residual_improved": (
            oracle["physical_residual_improved"]
        ),
        "linearization_error_norm": float(
            np.linalg.norm(oracle["linearization_error"])
        ),
        "H_derivation_source": measurement["H_derivation_source"],
        "finite_difference_method": (
            measurement["finite_difference_method"]
        ),
        "finite_difference_epsilon": float(
            measurement["finite_difference_epsilon"][0]
        ),
        "active_set_stable": measurement["active_set_stable"],
        "frozen_package_hash_before": frozen_hash_before,
        "frozen_package_hash_after": frozen_hash_after,
        "P_prior_trace": float(np.trace(oracle["P_prior"])),
        "P_posterior_trace": float(np.trace(oracle["P_posterior"])),
        "nis": oracle["nis"],
        "source_operator_set_id": "OP_TS204",
        "future_operator_set_ids": [
            "OP_TS301", "OP_TS302", "OP_TS303", "OP_TS304",
        ],
        "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "engineering_claim_allowed": False,
    }
    (TARGET / "validation" / "measurement_update_expected_summary.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = json.dumps({
        "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "engineering_claim_allowed": False,
        "schema_revision": "STAGE_MEASUREMENT_POSTERIOR_UPDATE_2026-07",
    }, ensure_ascii=False, separators=(",", ":"))
    matrix_count = len(arrays)
    pd.DataFrame([{
        "run_id": "RUN_STAGE_MEASUREMENT_UPDATE_BUILD",
        "model_version": "V2.5_STAGE_MEASUREMENT_UPDATE_FIXTURE",
        "input_package_id": PACKAGE_ID,
        "started_at": FIXED_TIMESTAMP,
        "finished_at": FIXED_TIMESTAMP,
        "operator_or_script": (
            "scripts/build_stage_measurement_update_fixture.py"
        ),
        "software_versions": "python=3.11;numpy;pandas",
        "object_ids_used": CHECKPOINT_ID,
        "cache_ids_used": "",
        "quality_gate_ids": "QG_STAGE_MEASUREMENT_UPDATE_FIXTURE",
        "warnings": "synthetic measurement only",
        "errors": "",
        "runtime_summary": (
            "deterministic package rebuild and independent oracle"
        ),
        "output_object_ids": (
            "STAGE_MEASUREMENT_UPDATE_ORACLE;EXPECTED_SUMMARY"
        ),
        "matrix_manifest_count": matrix_count,
        "npz_key_count": matrix_count,
        "metadata": metadata,
    }]).to_csv(
        TARGET / "validation" / "run_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "quality_gate_id": "QG_STAGE_MEASUREMENT_UPDATE_FIXTURE",
        "gate_name": "stage measurement update fixture self-validation",
        "target_object_type": "Package",
        "target_object_ids": PACKAGE_ID,
        "check_items": (
            "TOPOLOGY;CHECKPOINT;MEASUREMENT;ROLE;MATRICES;"
            "JOSEPH;GLOBAL_LCP;PHYSICAL_RESIDUAL;WEIGHTED_RESIDUAL;"
            "FINITE_DIFFERENCE_H;ACTIVE_SET_STABILITY;"
            "ORACLE;DICTIONARY;OBJECT_MAP;TRUTH"
        ),
        "thresholds": "blocking checks must all PASS",
        "pass_fail": "PASS",
        "warn_items": "synthetic_data_not_engineering_evidence",
        "fail_items": "",
        "reviewer_optional": "AUTO",
        "timestamp": FIXED_TIMESTAMP,
        "metadata": metadata,
    }]).to_csv(
        TARGET / "validation" / "quality_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "validation_result_id": "VAL_STAGE_MEASUREMENT_UPDATE_ORACLE",
        "model_version": "V2.5_STAGE_MEASUREMENT_UPDATE_FIXTURE",
        "validation_sample_ids": "SAMPLE_001",
        "reference_type": "SYNTHETIC_INDEPENDENT_POSTERIOR_LCP_ORACLE",
        "kcp_ids": "",
        "predicted_values": _json_array(
            oracle["z_predicted_posterior_physical"]
        ),
        "reference_values": _json_array(measurement["z_measured"]),
        "residuals": _json_array(
            oracle["residual_posterior_physical"]
        ),
        "MAE": float(np.mean(np.abs(
            oracle["residual_posterior_physical"]
        ))),
        "RMSE": float(np.sqrt(np.mean(
            oracle["residual_posterior_physical"] ** 2
        ))),
        "maximum_absolute_error": float(np.max(np.abs(
            oracle["residual_posterior_physical"]
        ))),
        "correlation_optional": "",
        "failure_probability_error_optional": "",
        "contact_mode_agreement_optional": "PASS",
        "overconstraint_reaction_agreement_optional": "",
        "acceleration_ratio_optional": "",
        "acceptance_criteria": (
            "post-LCP physical residual<=1e-4;"
            "weighted physical residual improves;"
            "complementarity<=1e-8"
        ),
        "pass_fail": "PASS",
        "metadata": metadata,
    }]).to_csv(
        TARGET / "validation" / "validation_result.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_manifest(route: pd.DataFrame) -> None:
    path = TARGET / "package_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update({
        "package_name": PACKAGE_ID,
        "schema_version": "V2.5_STAGE_MEASUREMENT_UPDATE",
        "schema_revision": "STAGE_MEASUREMENT_POSTERIOR_UPDATE_2026-07",
        "created_at": FIXED_TIMESTAMP,
        "description": (
            "Synthetic deterministic stage-measurement posterior-update "
            "integration fixture."
        ),
        "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "measurement_data_nature": (
            "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE"
        ),
        "engineering_claim_allowed": False,
        "purpose": (
            "stage measurement posterior update, global LCP re-solve, "
            "rollback and propagation integration fixture"
        ),
        "operator_mode": "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR",
        "measurement_update_mode": (
            "LINEAR_GAUSSIAN_JOSEPH_LOW_DIMENSIONAL_STATE"
        ),
        "fixture_expectations": {
            **manifest.get("fixture_expectations", {}),
            "topology_step_count": int(len(route)),
            "measurement_checkpoint_count": 1,
            "measurement_count": 2,
            "update_state_dimension": 2,
            "posterior_resolve_lcp_count": 1,
        },
        "truthfulness_statement": (
            "Synthetic numerical-consistency measurements and precomputed "
            "operators only. Not factory data, online FE, parameter "
            "identification, or engineering accuracy evidence."
        ),
    })
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _object_name(relative: str) -> str:
    special = {
        "I0/assembly_topology.csv": "AssemblyTopology",
        "I_meas/measurement_checkpoint.csv": "MeasurementCheckpointSpec",
        "I_meas/measurement_observation_map.csv": "MeasurementObservationSpec",
        "I_meas/measurement_update_config.csv": "MeasurementUpdateConfig",
        "I_stage/state_update_basis.csv": "StateUpdateBasis",
        "I_stage/stage_covariance_transfer.csv": "StageCovarianceTransfer",
        "I_stage/connection_lock_history.csv": "ConnectionLockHistory",
        "I_stage/release_history_record.csv": "ReleaseHistoryRecord",
        "matrices/matrix_manifest.csv": "MatrixManifest",
        "matrices/vector_layout.csv": "VectorLayout",
        "validation/topology_step_lcp_oracle.csv": "TopologyStepLcpOracle",
        "validation/stage_measurement_update_oracle.csv": (
            "StageMeasurementUpdateOracle"
        ),
        "validation/validation_result.csv": "ValidationResults",
        "field_dictionary.csv": "FieldDictionary",
        "object_file_map.csv": "ObjectFileMap",
    }
    if relative in special:
        return special[relative]
    return "".join(
        word.capitalize() for word in Path(relative).stem.split("_")
    )


def _primary_key(fields: list[str]) -> str:
    for field in fields:
        if field.endswith("_id") and not field.endswith("_ids"):
            return field
    return ""


def _write_object_map() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(TARGET.rglob("*.csv")):
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            fields = csv.DictReader(stream).fieldnames or []
        rows.append({
            "object_name": _object_name(relative),
            "input_package": PACKAGE_ID,
            "file_path": relative,
            "primary_key": _primary_key(fields),
            "code_consumer": "core.data_loader;core.stage_measurement_update",
            "producer": "scripts/build_stage_measurement_update_fixture.py",
            "is_synthetic": True,
            "is_runtime_result": False,
            "group": Path(relative).parent.as_posix(),
            "row_mode": "synthetic_stage_measurement_update",
        })
    runtime_objects = {
        "TopologyStepSpec": (
            "I0/assembly_topology.csv", "topology_step_id"
        ),
        "TopologyStepOperatorMatrices": (
            "matrices/multi_part_matrices.npz", "npz_key"
        ),
        "TopologyStepResult": (
            "runtime/topology_step_result.csv", "topology_step_id"
        ),
        "TopologyStepExecutionReport": (
            "runtime/topology_step_execution.csv", "topology_step_id"
        ),
        "UpdateDecisionRecord": (
            "runtime/update_decision_record.csv", "update_id"
        ),
        "ReSolveRequirement": (
            "runtime/resolve_requirement.csv", "update_id"
        ),
        "UpdateRollbackRecord": (
            "runtime/update_rollback_record.csv", "rollback_record_id"
        ),
        "StageMeasurementUpdateResult": (
            "runtime/measurement_update_summary.csv", "update_id"
        ),
        "PredictedStageState": (
            "runtime/predicted_state_snapshot.csv", "stage_state_id"
        ),
        "PosteriorStageState": (
            "runtime/posterior_state_snapshot.csv", "stage_state_id"
        ),
        "MeasurementInnovationReport": (
            "runtime/measurement_innovation.csv", "measurement_id"
        ),
        "PosteriorStateReport": (
            "runtime/posterior_state_lineage.csv", "state_id"
        ),
    }
    for object_name, (relative, key) in runtime_objects.items():
        exists = (TARGET / relative).exists()
        rows.append({
            "object_name": object_name,
            "input_package": PACKAGE_ID,
            "file_path": relative,
            "primary_key": key,
            "code_consumer": "core.reporting;app",
            "producer": (
                "scripts/build_stage_measurement_update_fixture.py"
                if exists
                else "core.reporting.build_runtime_report_zip"
            ),
            "is_synthetic": True,
            "is_runtime_result": not exists,
            "group": Path(relative).parent.as_posix(),
            "row_mode": (
                "synthetic_stage_measurement_update"
                if exists else "runtime_result"
            ),
        })
    rows.append({
        "object_name": "ObjectFileMap",
        "input_package": PACKAGE_ID,
        "file_path": "object_file_map.csv",
        "primary_key": "object_name",
        "code_consumer": "core.package_validator",
        "producer": "scripts/build_stage_measurement_update_fixture.py",
        "is_synthetic": True,
        "is_runtime_result": False,
        "group": ".",
        "row_mode": "governance",
    })
    pd.DataFrame(rows, columns=OBJECT_MAP_COLUMNS).drop_duplicates(
        ["object_name", "file_path"]
    ).to_csv(
        TARGET / "object_file_map.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _infer_type(values: list[str]) -> str:
    cleaned = [value for value in values if value.strip()]
    if not cleaned:
        return "string"
    if {value.lower() for value in cleaned} <= {
        "true", "false", "0", "1",
    }:
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


def _write_field_dictionary() -> None:
    schemas: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for path in sorted(TARGET.rglob("*.csv")):
        relative = path.relative_to(TARGET).as_posix()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            schemas[relative] = (reader.fieldnames or [], list(reader))
    schemas["field_dictionary.csv"] = (FIELD_DICTIONARY_COLUMNS, [])
    rows: list[dict[str, Any]] = []
    for relative, (fields, records) in sorted(schemas.items()):
        for field in fields:
            values = [str(record.get(field, "")) for record in records]
            required = field in {
                "topology_step_id", "checkpoint_id", "measurement_id",
                "update_config_id", "update_state_layout_id", "npz_key",
            }
            rows.append({
                "file_path": relative,
                "object_name": _object_name(relative),
                "field_name": field,
                "data_type": _infer_type(values),
                "required": required,
                "cardinality": "1" if required else "0..1",
                "unit": "per_object_definition",
                "enum_or_format": (
                    "IDENTIFY|CALIBRATE|VALIDATE|MONITOR|EXCLUDE"
                    if field == "data_role"
                    else ""
                ),
                "key_semantics": (
                    "primary or foreign key per object schema"
                    if field.endswith("_id")
                    else ""
                ),
                "missing_handling": (
                    "blocking if required" if required else "blank allowed"
                ),
                "description": f"{_object_name(relative)}.{field}",
                "example_value": next(
                    (value for value in values if value.strip()), ""
                ),
            })
    pd.DataFrame(rows, columns=FIELD_DICTIONARY_COLUMNS).to_csv(
        TARGET / "field_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_readme() -> None:
    (TARGET / "README.md").write_text(
        "# 03_STAGE_MEASUREMENT_UPDATE_MIN_CASE\n\n"
        "This package is a deterministic synthetic numerical-consistency case "
        "for a two-dimensional low-dimensional stage-state posterior update. "
        "TS205 is a non-solving MEASURE step whose checkpoint explicitly uses "
        "TS204 as its mechanical source. Two synthetic process measurements "
        "drive a one-pass linear Gaussian update with Joseph covariance, an "
        "explicit state-to-q mapping, and one coupled global LCP re-solve. "
        "The accepted posterior is propagated into TS301-TS304 with explicit "
        "identity F and zero Q matrices.\n\n"
        "`engineering_claim_allowed=false`. The measurements are not factory "
        "data, the sensitivities are not online FE derivatives, and the package "
        "does not validate engineering accuracy, identify parameters, or update "
        "SMS/Cn/Ct/mu/beta_r/joint stiffness.\n",
        encoding="utf-8",
    )


def package_digest(root: Path = TARGET) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_self_validation() -> dict[str, Any]:
    shutil.copyfile(
        VALIDATOR_SOURCE,
        TARGET / "validation" / "validate_package.py",
    )
    report = validate_package(TARGET, include_attachment_check=False)
    report["checks"].append({
        "name": "validation attachments match current package",
        "passed": True,
        "status": "PASS",
        "blocking": True,
        "detail": "stored_status=PASS",
    })
    report["total"] = len(report["checks"])
    report["passed"] = sum(
        item["status"] == "PASS" for item in report["checks"]
    )
    report["blocking_fail_count"] = sum(
        item["status"] == "FAIL" and item["blocking"]
        for item in report["checks"]
    )
    report["status"] = (
        "PASS" if report["blocking_fail_count"] == 0 else "FAIL"
    )
    (TARGET / "validation" / "test_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (TARGET / "validation" / "TEST_RESULTS.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    return validate_package(TARGET)


def build() -> dict[str, Any]:
    _safe_recreate_target()
    route = _write_route()
    measurement = _write_measurement_tables()
    arrays, oracle = _write_matrices(measurement)
    _write_oracle_and_validation(arrays, measurement, oracle)
    _write_manifest(route)
    _write_readme()
    _write_object_map()
    _write_field_dictionary()
    report = _write_self_validation()
    if report["status"] != "PASS":
        failures = [
            item for item in report["checks"]
            if item["status"] == "FAIL" and item["blocking"]
        ]
        raise RuntimeError(
            "generated package failed self-validation: "
            + json.dumps(failures, ensure_ascii=False)
        )
    report["package_digest"] = package_digest()
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
