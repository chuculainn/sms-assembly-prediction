from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.data_loader import SMSPackage
from tests.topology_fixture_factory import make_two_part_two_point_fixture


def make_non12d_measurement_fixture() -> SMSPackage:
    """Create a 2-point/1-interface fixture with a 1D posterior state."""
    package = make_two_part_two_point_fixture()
    package.manifest.update({
        "package_name": "TEST_NON12D_STAGE_MEASUREMENT",
        "measurement_data_nature":
            "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "engineering_claim_allowed": False,
    })
    route = package.raw_tables["I0/assembly_topology.csv"].copy()
    route.loc[
        route["topology_step_id"].astype(str).eq("EV_INSPECT"),
        "measurement_checkpoint_id",
    ] = ""
    package.raw_tables["I0/assembly_topology.csv"] = route
    package.raw_tables["I_meas/measurement_checkpoint.csv"] = pd.DataFrame([{
        "checkpoint_id": "CHECK_MID",
        "topology_id": "TINY_ROUTE",
        "topology_step_id": "EV_MEASURE",
        "source_topology_step_id": "EV_LOCATE",
        "measurement_set_id": "SET_MID",
        "update_config_id": "CFG_MID",
        "reference_state_id": "PARENT_RUNTIME_STATE",
        "missing_measurement_policy": "BLOCK",
        "rollback_policy": "ROLLBACK_TO_PREDICTED",
        "active_flag": True,
        "notes": "synthetic non-12D fixture",
    }])
    package.raw_tables[
        "I_meas/measurement_observation_map.csv"
    ] = pd.DataFrame([{
        "observation_id": "OBS_RIGHT_GAP",
        "checkpoint_id": "CHECK_MID",
        "measurement_id": "MEAS_RIGHT_GAP",
        "observation_order": 0,
        "observed_quantity": "GAP_G",
        "vector_source": "RUNTIME_STAGE_STATE",
        "global_index_optional": 1,
        "target_object_id_optional": "IF_ALPHA_BETA",
        "unit": "mm",
        "coordinate_system_id": "",
        "reference_state_id": "PARENT_RUNTIME_STATE",
        "sensitivity_row_index": 0,
        "quality_flag": "PASS",
    }])
    package.raw_tables[
        "I_meas/measurement_update_config.csv"
    ] = pd.DataFrame([{
        "update_config_id": "CFG_MID",
        "update_state_layout_id": "ETA_TINY_1",
        "prior_mean_matrix_id": "ETA_TINY_PRIOR",
        "prior_covariance_matrix_id": "P_TINY_PRIOR",
        "observation_jacobian_matrix_id": "H_TINY",
        "measurement_covariance_matrix_id_optional": "R_TINY",
        "state_to_q_mapping_rule":
            "G_Q_TINY__{checkpoint_id}__{operator_set_id}",
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
    }])
    package.raw_tables["I_meas/measurement_record.csv"] = pd.DataFrame([{
        "measurement_id": "MEAS_RIGHT_GAP",
        "measurement_set_id": "SET_MID",
        "sample_id": "SAMPLE_001",
        "object_type": "Interface",
        "object_id": "IF_ALPHA_BETA",
        "interface_id_optional": "IF_ALPHA_BETA",
        "stage_id": "MEASURE_STAGE",
        "measurement_type": "GAP_G",
        "location": "global_index=1",
        "direction": "normal",
        "coordinate_system_id": "",
        "value": 0.30,
        "unit": "mm",
        "standard_uncertainty": 0.10,
        "covariance_block_id_optional": "R_TINY",
        "sensor_or_device_id": "SYNTH_SENSOR",
        "update_target": "STAGE_STATE",
        "data_role": "CALIBRATE",
        "quality_flag": "PASS",
        "reference_state_id_optional": "PARENT_RUNTIME_STATE",
        "timestamp": "2026-07-25T00:00:00+08:00",
        "source": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
    }])
    package.raw_tables["I_stage/state_update_basis.csv"] = pd.DataFrame([{
        "update_state_layout_id": "ETA_TINY_1",
        "component_order": 0,
        "component_id": "BIAS_RIGHT_GAP",
        "component_type": "GAP_BIAS",
        "target_object_type": "Interface",
        "target_object_id": "IF_ALPHA_BETA",
        "unit": "mm",
        "description": "one-dimensional synthetic correction",
    }])
    package.raw_tables[
        "I_stage/stage_covariance_transfer.csv"
    ] = pd.DataFrame([
        {
            "covariance_transfer_id": f"TRANSFER_{step}",
            "update_state_layout_id": "ETA_TINY_1",
            "from_topology_step_id": previous,
            "to_topology_step_id": step,
            "state_jacobian_F_matrix_id": "F_TINY",
            "process_noise_Q_matrix_id": "Q_TINY",
            "quality_flag": "PASS",
        }
        for previous, step in (
            ("EV_MEASURE", "EV_JOIN"),
            ("EV_JOIN", "EV_RELEASE"),
            ("EV_RELEASE", "EV_INSPECT"),
        )
    ])
    additions = {
        "ETA_TINY_PRIOR": np.zeros(1),
        "P_TINY_PRIOR": np.array([[0.04]]),
        "H_TINY": np.array([[1.0]]),
        "R_TINY": np.array([[0.01]]),
        "F_TINY": np.eye(1),
        "Q_TINY": np.zeros((1, 1)),
        "G_Q_TINY__CHECK_MID__TINY_LOCATE":
            np.array([[0.0], [1.0]]),
        "G_Q_TINY__CHECK_MID__TINY_JOIN":
            np.array([[0.2], [0.8]]),
        "G_Q_TINY__CHECK_MID__TINY_RELEASE":
            np.array([[0.1], [0.6]]),
    }
    package.matrices.update(additions)
    manifest = package.raw_tables["matrices/matrix_manifest.csv"].copy()
    rows = []
    for key, value in additions.items():
        is_gq = key.startswith("G_Q_TINY")
        is_h = key == "H_TINY"
        is_r = key == "R_TINY"
        rows.append({
            "matrix_id": key,
            "npz_file": "in_memory.npz",
            "npz_key": key,
            "shape": json.dumps(list(value.shape)),
            "dtype": str(value.dtype),
            "row_layout_id_optional": (
                "VL_TWO_POINT"
                if is_gq
                else "OBS_TINY_1"
                if is_h or is_r
                else "ETA_TINY_1"
                if key in {
                    "ETA_TINY_PRIOR", "P_TINY_PRIOR",
                    "F_TINY", "Q_TINY",
                }
                else ""
            ),
            "column_layout_id_optional": (
                "ETA_TINY_1"
                if is_gq or is_h or key in {
                    "P_TINY_PRIOR", "F_TINY", "Q_TINY",
                }
                else "OBS_TINY_1"
                if is_r
                else ""
            ),
            "derivation_source": "synthetic non-12D fixture",
            "checkpoint_id_optional": (
                "CHECK_MID" if is_gq else ""
            ),
            "operator_set_id_optional": (
                key.rsplit("__", 1)[-1] if is_gq else ""
            ),
            "is_synthetic": True,
            "engineering_claim_allowed": False,
        })
    package.raw_tables["matrices/matrix_manifest.csv"] = pd.concat(
        [manifest, pd.DataFrame(rows)],
        ignore_index=True,
    )
    return package


def make_double_checkpoint_measurement_fixture() -> SMSPackage:
    """Create two sequential checkpoints; the second has zero innovation."""
    package = make_non12d_measurement_fixture()
    route = package.raw_tables["I0/assembly_topology.csv"].copy()
    first_index = route.index[
        route["topology_step_id"].astype(str).eq("EV_MEASURE")
    ][0]
    first = route.loc[first_index].copy()
    first_order = int(first["step_order"])
    route.loc[
        pd.to_numeric(route["step_order"], errors="coerce") > first_order,
        "step_order",
    ] = (
        pd.to_numeric(
            route.loc[
                pd.to_numeric(route["step_order"], errors="coerce")
                > first_order,
                "step_order",
            ],
            errors="coerce",
        )
        + 1
    )
    route.loc[first_index, "operator_set_id"] = "TINY_LOCATE"
    second = first.to_dict()
    second["topology_step_id"] = "EV_MEASURE_2"
    second["step_order"] = first_order + 1
    second["parent_topology_step_id"] = "EV_MEASURE"
    second["operation_type"] = "MEASURE"
    second["stage_id"] = "MEASURE_STAGE_2"
    second["measurement_checkpoint_id"] = "CHECK_SECOND"
    second["operator_set_id"] = "TINY_LOCATE"
    second["solve_required"] = False
    second["notes"] = "second sequential zero-innovation checkpoint"
    route.loc[
        route["topology_step_id"].astype(str).eq("EV_JOIN"),
        "parent_topology_step_id",
    ] = "EV_MEASURE_2"
    route = pd.concat(
        [route, pd.DataFrame([second])], ignore_index=True
    ).sort_values("step_order", kind="stable").reset_index(drop=True)
    package.raw_tables["I0/assembly_topology.csv"] = route

    checkpoints = package.raw_tables[
        "I_meas/measurement_checkpoint.csv"
    ].copy()
    checkpoints = pd.concat([
        checkpoints,
        pd.DataFrame([{
            "checkpoint_id": "CHECK_SECOND",
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_MEASURE_2",
            "source_topology_step_id": "EV_MEASURE",
            "measurement_set_id": "SET_SECOND",
            "update_config_id": "CFG_SECOND",
            "reference_state_id": "PARENT_RUNTIME_STATE",
            "missing_measurement_policy": "BLOCK",
            "rollback_policy": "ROLLBACK_TO_PREDICTED",
            "active_flag": True,
            "notes": "zero innovation after accepted first posterior",
        }]),
    ], ignore_index=True)
    package.raw_tables[
        "I_meas/measurement_checkpoint.csv"
    ] = checkpoints

    observations = package.raw_tables[
        "I_meas/measurement_observation_map.csv"
    ].copy()
    observations = pd.concat([
        observations,
        pd.DataFrame([{
            "observation_id": "OBS_SECOND_GAP",
            "checkpoint_id": "CHECK_SECOND",
            "measurement_id": "MEAS_SECOND_GAP",
            "observation_order": 0,
            "observed_quantity": "GAP_G",
            "vector_source": "RUNTIME_STAGE_STATE",
            "global_index_optional": 1,
            "target_object_id_optional": "IF_ALPHA_BETA",
            "unit": "mm",
            "coordinate_system_id": "",
            "reference_state_id": "PARENT_RUNTIME_STATE",
            "sensitivity_row_index": 0,
            "quality_flag": "PASS",
        }]),
    ], ignore_index=True)
    package.raw_tables[
        "I_meas/measurement_observation_map.csv"
    ] = observations

    configs = package.raw_tables[
        "I_meas/measurement_update_config.csv"
    ].copy()
    configs = pd.concat([
        configs,
        pd.DataFrame([{
            "update_config_id": "CFG_SECOND",
            "update_state_layout_id": "ETA_TINY_1",
            "prior_mean_matrix_id": "ETA_SECOND_PRIOR",
            "prior_covariance_matrix_id": "P_SECOND_PRIOR",
            "observation_jacobian_matrix_id": "H_SECOND",
            "measurement_covariance_matrix_id_optional": "R_SECOND",
            "state_to_q_mapping_rule":
                "G_Q_SECOND__{checkpoint_id}__{operator_set_id}",
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
        }]),
    ], ignore_index=True)
    package.raw_tables[
        "I_meas/measurement_update_config.csv"
    ] = configs

    measurements = package.raw_tables[
        "I_meas/measurement_record.csv"
    ].copy()
    measurements = pd.concat([
        measurements,
        pd.DataFrame([{
            "measurement_id": "MEAS_SECOND_GAP",
            "measurement_set_id": "SET_SECOND",
            "sample_id": "SAMPLE_001",
            "object_type": "Interface",
            "object_id": "IF_ALPHA_BETA",
            "interface_id_optional": "IF_ALPHA_BETA",
            "stage_id": "MEASURE_STAGE_2",
            "measurement_type": "GAP_G",
            "location": "global_index=1",
            "direction": "normal",
            "coordinate_system_id": "",
            "value": 0.2942857142857143,
            "unit": "mm",
            "standard_uncertainty": 0.10,
            "covariance_block_id_optional": "R_SECOND",
            "sensor_or_device_id": "SYNTH_SENSOR_2",
            "update_target": "STAGE_STATE",
            "data_role": "CALIBRATE",
            "quality_flag": "PASS",
            "reference_state_id_optional": "PARENT_RUNTIME_STATE",
            "timestamp": "2026-07-25T00:00:00+08:00",
            "source": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        }]),
    ], ignore_index=True)
    package.raw_tables["I_meas/measurement_record.csv"] = measurements

    additions = {
        "ETA_SECOND_PRIOR": np.zeros(1),
        "P_SECOND_PRIOR": np.array([[0.04]]),
        "H_SECOND": np.array([[1.0]]),
        "R_SECOND": np.array([[0.01]]),
        "G_Q_SECOND__CHECK_SECOND__TINY_LOCATE":
            np.array([[0.0], [1.0]]),
        "G_Q_SECOND__CHECK_SECOND__TINY_JOIN":
            np.array([[0.2], [0.8]]),
        "G_Q_SECOND__CHECK_SECOND__TINY_RELEASE":
            np.array([[0.1], [0.6]]),
    }
    package.matrices.update(additions)
    manifest = package.raw_tables[
        "matrices/matrix_manifest.csv"
    ].copy()
    rows = []
    for key, value in additions.items():
        is_gq = key.startswith("G_Q_SECOND")
        rows.append({
            "matrix_id": key,
            "npz_file": "in_memory.npz",
            "npz_key": key,
            "shape": json.dumps(list(value.shape)),
            "dtype": str(value.dtype),
            "row_layout_id_optional": (
                "VL_TWO_POINT"
                if is_gq
                else "OBS_SECOND_1"
                if key in {"H_SECOND", "R_SECOND"}
                else "ETA_TINY_1"
            ),
            "column_layout_id_optional": (
                "ETA_TINY_1"
                if is_gq or key in {
                    "P_SECOND_PRIOR", "H_SECOND",
                }
                else "OBS_SECOND_1"
                if key == "R_SECOND"
                else ""
            ),
            "derivation_source":
                "synthetic double-checkpoint fixture",
            "measurement_checkpoint_id_optional": (
                "CHECK_SECOND" if is_gq else ""
            ),
            "operator_set_id_optional": (
                key.rsplit("__", 1)[-1] if is_gq else ""
            ),
            "engineering_claim_allowed": False,
        })
    package.raw_tables["matrices/matrix_manifest.csv"] = pd.concat(
        [manifest, pd.DataFrame(rows)], ignore_index=True
    )
    return package
