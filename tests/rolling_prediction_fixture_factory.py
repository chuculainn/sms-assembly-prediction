from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from core.data_loader import SMSPackage
from tests.stage_measurement_fixture_factory import (
    make_non12d_measurement_fixture,
)


def make_non12d_rolling_prediction_fixture() -> SMSPackage:
    """Return a 3-contact fixture with a 1D future-part virtual SMS."""
    package = copy.deepcopy(make_non12d_measurement_fixture())
    package.manifest.update({
        "package_name": "TEST_NON12D_POSTERIOR_ROLLING",
        "virtual_sms_sample_nature":
            "EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY",
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
    })

    package.parts = pd.concat([
        package.parts,
        pd.DataFrame([{
            "part_id": "X_GAMMA", "part_name": "gamma"
        }]),
    ], ignore_index=True)
    package.interfaces = pd.concat([
        package.interfaces,
        pd.DataFrame([{
            "interface_id": "IF_BETA_GAMMA",
            "part_i": "X_BETA",
            "part_j": "X_GAMMA",
        }]),
    ], ignore_index=True)
    package.contact_points = pd.concat([
        package.contact_points,
        pd.DataFrame([{
            "contact_point_id": "CP_GAMMA",
            "contact_domain_id": "CD_BETA_GAMMA",
            "local_index": 0,
            "area_weight": 1.5,
        }]),
    ], ignore_index=True)
    package.raw_tables["I0/part.csv"] = package.parts.copy()
    package.raw_tables["I0/interface.csv"] = package.interfaces.copy()
    package.raw_tables["I_Gamma/contact_point.csv"] = (
        package.contact_points.copy()
    )
    domains = package.raw_tables["I_Gamma/contact_domain.csv"].copy()
    package.raw_tables["I_Gamma/contact_domain.csv"] = pd.concat([
        domains,
        pd.DataFrame([{
            "contact_domain_id": "CD_BETA_GAMMA",
            "interface_id": "IF_BETA_GAMMA",
        }]),
    ], ignore_index=True)
    layout = pd.DataFrame([
        {
            "vector_layout_id": "VL_THREE_CONTACT",
            "block_order": 0,
            "object_type": "Interface",
            "object_id": "IF_ALPHA_BETA",
            "contact_domain_id": "CD_ALPHA_BETA",
            "start_index": 0,
            "end_index": 1,
        },
        {
            "vector_layout_id": "VL_THREE_CONTACT",
            "block_order": 1,
            "object_type": "Interface",
            "object_id": "IF_BETA_GAMMA",
            "contact_domain_id": "CD_BETA_GAMMA",
            "start_index": 2,
            "end_index": 2,
        },
    ])
    package.raw_tables["matrices/vector_layout.csv"] = layout

    route = package.raw_tables["I0/assembly_topology.csv"].copy()
    join_mask = route["topology_step_id"].astype(str).eq("EV_JOIN")
    route.loc[join_mask, "added_part_ids"] = "X_GAMMA"
    route.loc[join_mask, "activated_interface_ids"] = "IF_BETA_GAMMA"
    package.raw_tables["I0/assembly_topology.csv"] = route

    contact_vector_keys = {
        key for key, value in package.matrices.items()
        if (
            key.startswith("Q_TINY_")
            or key.startswith("U_FREE_TINY_")
            or key.startswith("G_Q_TINY__")
            or key.startswith("W_STRUCT_TINY_")
            or key.startswith("W_TOTAL_TINY_")
            or key.startswith("CN_TINY_")
        )
        and value.shape[0] == 2
    }
    for key in sorted(contact_vector_keys):
        value = np.asarray(package.matrices[key], dtype=float)
        if value.ndim == 1:
            package.matrices[key] = np.r_[value, 0.0]
        elif key.startswith("G_Q_TINY__"):
            package.matrices[key] = np.vstack([value, np.zeros(
                (1, value.shape[1])
            )])
        else:
            expanded = np.zeros((3, 3), dtype=float)
            expanded[:2, :2] = value
            if key.startswith("W_STRUCT_"):
                expanded[2, 2] = 1.2
                expanded[0, 2] = expanded[2, 0] = 0.04
                expanded[1, 2] = expanded[2, 1] = -0.02
            elif key.startswith("CN_"):
                expanded[2, 2] = 0.15
            package.matrices[key] = expanded
    for operator in ("TINY_JOIN", "TINY_RELEASE"):
        package.matrices[f"W_TOTAL_{operator}"] = (
            package.matrices[f"W_STRUCT_{operator}"]
            + package.matrices[f"CN_{operator}"]
        )

    additions = {
        "G_SMS_GAMMA_JOIN": np.array([[0.0], [0.0], [-0.25]]),
        "G_SMS_GAMMA_RELEASE": np.array([[0.0], [0.0], [-0.15]]),
        "Q_FUTURE_ZERO_3": np.zeros(3),
        "J_INTERFACE_ALL": np.array([[0.0, 0.0, 0.5]]),
        "J_SMS_GAMMA": np.array([[0.2]]),
    }
    package.matrices.update(additions)
    manifest = package.raw_tables["matrices/matrix_manifest.csv"].copy()
    for index, row in manifest.iterrows():
        matrix_id = str(row.get("matrix_id", ""))
        if matrix_id in package.matrices:
            value = np.asarray(package.matrices[matrix_id])
            manifest.at[index, "shape"] = json.dumps(list(value.shape))
            if matrix_id in contact_vector_keys:
                manifest.at[index, "row_layout_id_optional"] = (
                    "VL_THREE_CONTACT"
                )
                if value.ndim == 2 and not matrix_id.startswith(
                    "G_Q_TINY__"
                ):
                    manifest.at[index, "column_layout_id_optional"] = (
                        "VL_THREE_CONTACT"
                    )
    new_manifest_rows = []
    for matrix_id, value in additions.items():
        is_sms_q = matrix_id.startswith("G_SMS_")
        is_direct = matrix_id == "J_SMS_GAMMA"
        is_process_zero = matrix_id == "Q_FUTURE_ZERO_3"
        new_manifest_rows.append({
            "matrix_id": matrix_id,
            "npz_file": "in_memory.npz",
            "npz_key": matrix_id,
            "shape": json.dumps(list(value.shape)),
            "dtype": str(value.dtype),
            "row_layout_id_optional": (
                "VL_THREE_CONTACT"
                if value.shape[0] == 3
                else "KCP_TINY"
            ),
            "column_layout_id_optional": (
                "SMS_GAMMA_1"
                if matrix_id.startswith("G_SMS_")
                or matrix_id == "J_SMS_GAMMA"
                else "VL_THREE_CONTACT"
                if matrix_id == "J_INTERFACE_ALL"
                else ""
            ),
            "derivation_source": "synthetic non-12D rolling fixture",
            "quality_flag": "PASS",
            "engineering_claim_allowed": False,
            "future_part_id_optional": (
                "X_GAMMA" if is_sms_q or is_direct else ""
            ),
            "reference_sms_id_optional": (
                "SMS_REF_GAMMA" if is_sms_q or is_direct else ""
            ),
            "mapping_semantics_optional": (
                "DELTA_FROM_OPERATOR_REFERENCE"
                if is_sms_q or is_direct else
                "EXPLICIT_ZERO" if is_process_zero else ""
            ),
            "mapping_role_optional": (
                "EFFECTIVE_MAPPING" if is_sms_q else
                "SMS_TO_KCP_DIRECT" if is_direct else
                "EXPLICIT_ZERO_NO_EFFECT" if is_process_zero else ""
            ),
        })
    package.raw_tables["matrices/matrix_manifest.csv"] = pd.concat([
        manifest, pd.DataFrame(new_manifest_rows)
    ], ignore_index=True)

    plan_id = "PLAN_TINY_POSTERIOR"
    sample_set_id = "SET_TINY_SMS"
    reference_id = "SMS_REF_GAMMA"
    package.raw_tables["I_pred/rolling_prediction_plan.csv"] = pd.DataFrame([{
        "rolling_plan_id": plan_id,
        "topology_id": "TINY_ROUTE",
        "source_checkpoint_id": "CHECK_MID",
        "source_topology_step_id": "EV_MEASURE",
        "source_state_policy": "REQUIRE_ACCEPTED_POSTERIOR",
        "source_posterior_state_id_optional": "",
        "prediction_start_step_id": "EV_JOIN",
        "prediction_end_step_id_optional": "EV_INSPECT",
        "virtual_sms_sample_set_id": sample_set_id,
        "future_part_ids": "X_GAMMA",
        "future_process_scenario_id": "SCENARIO_TINY_ZERO",
        "kcp_set_id": "KCP_TINY",
        "baseline_comparison_policy": "POSTERIOR_AND_PREDICTED_CUTOFF",
        "aggregation_policy":
            "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE",
        "failure_policy": "FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS",
        "active_flag": True,
        "engineering_claim_allowed": False,
        "quality_flag": "PASS",
        "notes": "non-12D deterministic fixture",
    }])
    package.raw_tables["I_pred/virtual_sms_sample_set.csv"] = pd.DataFrame([{
        "sample_set_id": sample_set_id,
        "sample_set_name": "tiny explicit set",
        "sample_nature": "EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY",
        "sample_count": 2,
        "sms_layout_id": "SMS_GAMMA_1",
        "reference_sms_id": reference_id,
        "generation_method": "DETERMINISTIC_EXPLICIT_VALUES",
        "probability_interpretation_allowed": False,
        "engineering_claim_allowed": False,
        "source": "in-memory test fixture",
        "quality_flag": "PASS",
        "notes": "",
    }])
    package.raw_tables["I_pred/virtual_sms_sample.csv"] = pd.DataFrame([
        {
            "virtual_sms_sample_id": "TINY_REF",
            "sample_set_id": sample_set_id,
            "part_id": "X_GAMMA",
            "sms_layout_id": "SMS_GAMMA_1",
            "reference_sms_id": reference_id,
            "coefficient_source": "REFERENCE_SMS",
            "sample_order": 0,
            "quality_flag": "PASS",
            "source": "explicit",
            "notes": "",
        },
        {
            "virtual_sms_sample_id": "TINY_OFFSET",
            "sample_set_id": sample_set_id,
            "part_id": "X_GAMMA",
            "sms_layout_id": "SMS_GAMMA_1",
            "reference_sms_id": reference_id,
            "coefficient_source": "EXPLICIT_VALUE",
            "sample_order": 1,
            "quality_flag": "PASS",
            "source": "explicit",
            "notes": "",
        },
    ])
    package.raw_tables["I_pred/virtual_sms_component.csv"] = pd.DataFrame([{
        "sms_layout_id": "SMS_GAMMA_1",
        "component_order": 0,
        "component_id": "GAMMA_MODE",
        "component_type": "SYNTHETIC_SHAPE_MODE",
        "part_id": "X_GAMMA",
        "unit": "1",
        "reference_state_id": reference_id,
        "quality_flag": "PASS",
        "description": "one explicit mode",
    }])
    package.raw_tables["I_pred/virtual_sms_coefficients.csv"] = pd.DataFrame([
        {
            "virtual_sms_sample_id": "TINY_REF",
            "part_id": "X_GAMMA",
            "component_id": "GAMMA_MODE",
            "component_order": 0,
            "value": 0.0,
            "unit": "1",
            "reference_sms_id": reference_id,
            "quality_flag": "PASS",
        },
        {
            "virtual_sms_sample_id": "TINY_OFFSET",
            "part_id": "X_GAMMA",
            "component_id": "GAMMA_MODE",
            "component_order": 0,
            "value": 0.4,
            "unit": "1",
            "reference_sms_id": reference_id,
            "quality_flag": "PASS",
        },
    ])
    package.raw_tables["I_pred/future_sms_assignment.csv"] = pd.DataFrame([
        {
            "assignment_id": f"ASSIGN_{sample_id}",
            "rolling_plan_id": plan_id,
            "virtual_sms_sample_id": sample_id,
            "part_id": "X_GAMMA",
            "first_effective_topology_step_id": "EV_JOIN",
            "last_effective_topology_step_id_optional": "EV_INSPECT",
            "mapping_semantics": "DELTA_FROM_OPERATOR_REFERENCE",
            "reference_sms_id": reference_id,
            "quality_flag": "PASS",
        }
        for sample_id in ("TINY_REF", "TINY_OFFSET")
    ])
    package.raw_tables["I_pred/sms_operator_mapping.csv"] = pd.DataFrame([
        {
            "mapping_id": f"MAP_{operator}",
            "rolling_plan_id": plan_id,
            "part_id": "X_GAMMA",
            "operator_set_id": operator,
            "matrix_id": matrix_id,
            "row_vector_layout_id": "VL_THREE_CONTACT",
            "column_sms_layout_id": "SMS_GAMMA_1",
            "mapping_semantics": "DELTA_FROM_OPERATOR_REFERENCE",
            "reference_sms_id": reference_id,
            "first_valid_step_id": step_id,
            "last_valid_step_id_optional": step_id,
            "mapping_role": "EFFECTIVE_MAPPING",
            "derivation_source": "synthetic non-12D rolling fixture",
            "quality_flag": "PASS",
        }
        for operator, matrix_id, step_id in (
            ("TINY_JOIN", "G_SMS_GAMMA_JOIN", "EV_JOIN"),
            ("TINY_RELEASE", "G_SMS_GAMMA_RELEASE", "EV_RELEASE"),
        )
    ])
    package.raw_tables["I_pred/future_process_scenario.csv"] = pd.DataFrame([{
        "future_process_scenario_id": "SCENARIO_TINY_ZERO",
        "scenario_type": "DETERMINISTIC_BASELINE",
        "q_correction_matrix_id_optional": "Q_FUTURE_ZERO_3",
        "parameter_override_allowed": False,
        "probability_weight_optional": "",
        "probability_interpretation_allowed": False,
        "quality_flag": "PASS",
    }])
    package.raw_tables["I_pred/rolling_kcp_config.csv"] = pd.DataFrame([{
        "rolling_plan_id": plan_id,
        "part_id": "X_GAMMA",
        "kcp_set_id": "KCP_TINY",
        "direct_sms_matrix_id_optional": "J_SMS_GAMMA",
        "aggregation_policy":
            "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE",
        "final_state_includes_direct_sms_geometry": False,
        "quality_flag": "PASS",
    }])

    package.kcp_kcm = pd.DataFrame([{
        "feature_id": "KCP_TINY_GAP",
        "feature_role": "KCP",
        "feature_type": "gap",
        "stage_id_optional": "RELEASE_STAGE",
        "nominal_value": 0.0,
        "lower_tol": -1.0,
        "upper_tol": 1.0,
        "expected_unit": "mm",
        "description": "tiny final gap",
        "extraction_matrix_id": "J_INTERFACE_ALL",
        "kcp_set_id": "KCP_TINY",
    }])
    package.raw_tables["prediction/kcp_prediction_result.csv"] = pd.DataFrame([{
        "kcp_ids": "KCP_TINY_GAP",
        "sms_contribution": "[0.0]",
        "process_contribution": "[0.0]",
        "rebound_contribution": "[0.0]",
        "slip_contribution_optional": "[]",
        "measurement_uncertainty_contribution_optional": "[]",
    }])
    return package
