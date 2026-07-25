from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_loader import SMSPackage, load_package


ROOT = Path(__file__).resolve().parents[1]


def make_two_part_two_point_fixture() -> SMSPackage:
    """Return an in-memory non-12D fixture with MEASURE, RELEASE and INSPECT."""
    package = copy.deepcopy(
        load_package(ROOT / "data" / "02_TOPOLOGY_STEP_MIN_CASE")
    )
    package.manifest = {
        "package_name": "TEST_TWO_PART_TWO_POINT",
        "schema_version": "V2.5_TOPOLOGY_STEP",
        "data_nature": "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE",
        "engineering_claim_allowed": False,
        "operator_mode": "PRECOMPUTED_TOPOLOGY_STEP_OPERATOR",
        "fixture_expectations": {"require_nonzero_cross_blocks": False},
    }
    package.package_type = "V25_MULTI_PART"
    package.parts = pd.DataFrame(
        [
            {"part_id": "X_ALPHA", "part_name": "alpha"},
            {"part_id": "X_BETA", "part_name": "beta"},
        ]
    )
    package.interfaces = pd.DataFrame(
        [
            {
                "interface_id": "IF_ALPHA_BETA",
                "part_i": "X_ALPHA",
                "part_j": "X_BETA",
            }
        ]
    )
    package.contact_points = pd.DataFrame(
        [
            {
                "contact_point_id": "CP_LEFT",
                "contact_domain_id": "CD_ALPHA_BETA",
                "local_index": 0,
                "area_weight": 1.0,
            },
            {
                "contact_point_id": "CP_RIGHT",
                "contact_domain_id": "CD_ALPHA_BETA",
                "local_index": 1,
                "area_weight": 2.0,
            },
        ]
    )
    route_rows = [
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_INIT",
            "step_order": 0,
            "parent_topology_step_id": "",
            "assembly_cycle_id": "C0",
            "operation_type": "INIT",
            "stage_id": "INIT_STAGE",
            "input_subassembly_id": "",
            "result_subassembly_id": "ASM_ALPHA",
            "added_part_ids": "X_ALPHA",
            "operator_set_id": "",
            "solve_required": False,
        },
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_LOCATE",
            "step_order": 10,
            "parent_topology_step_id": "EV_INIT",
            "assembly_cycle_id": "C1",
            "operation_type": "LOCATE",
            "stage_id": "LOCATE_STAGE",
            "input_subassembly_id": "ASM_ALPHA",
            "result_subassembly_id": "ASM_PAIR",
            "added_part_ids": "X_BETA",
            "activated_interface_ids": "IF_ALPHA_BETA",
            "operator_set_id": "TINY_LOCATE",
            "solve_required": True,
        },
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_MEASURE",
            "step_order": 20,
            "parent_topology_step_id": "EV_LOCATE",
            "assembly_cycle_id": "C1",
            "operation_type": "MEASURE",
            "stage_id": "MEASURE_STAGE",
            "input_subassembly_id": "ASM_PAIR",
            "result_subassembly_id": "ASM_PAIR",
            "operator_set_id": "",
            "solve_required": False,
            "measurement_checkpoint_id": "CHECK_MID",
        },
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_JOIN",
            "step_order": 30,
            "parent_topology_step_id": "EV_MEASURE",
            "assembly_cycle_id": "C1",
            "operation_type": "JOIN",
            "stage_id": "JOIN_STAGE",
            "input_subassembly_id": "ASM_PAIR",
            "result_subassembly_id": "ASM_PAIR",
            "activated_joint_ids": "J_KEEP;J_DROP",
            "operator_set_id": "TINY_JOIN",
            "solve_required": True,
        },
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_RELEASE",
            "step_order": 40,
            "parent_topology_step_id": "EV_JOIN",
            "assembly_cycle_id": "C1",
            "operation_type": "RELEASE",
            "stage_id": "RELEASE_STAGE",
            "input_subassembly_id": "ASM_PAIR",
            "result_subassembly_id": "ASM_PAIR",
            "operator_set_id": "TINY_RELEASE",
            "solve_required": True,
        },
        {
            "topology_id": "TINY_ROUTE",
            "topology_step_id": "EV_INSPECT",
            "step_order": 50,
            "parent_topology_step_id": "EV_RELEASE",
            "assembly_cycle_id": "C1",
            "operation_type": "INSPECT",
            "stage_id": "INSPECT_STAGE",
            "input_subassembly_id": "ASM_PAIR",
            "result_subassembly_id": "ASM_PAIR",
            "operator_set_id": "",
            "solve_required": False,
            "measurement_checkpoint_id": "CHECK_FINAL",
        },
    ]
    topology_columns = [
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
    route = pd.DataFrame(route_rows).reindex(columns=topology_columns, fill_value="")
    layout = pd.DataFrame(
        [
            {
                "vector_layout_id": "VL_TWO_POINT",
                "block_order": 0,
                "object_type": "Interface",
                "object_id": "IF_ALPHA_BETA",
                "contact_domain_id": "CD_ALPHA_BETA",
                "start_index": 0,
                "end_index": 1,
            }
        ]
    )
    joints = pd.DataFrame(
        [
            {
                "joint_id": "J_KEEP",
                "interface_id": "IF_ALPHA_BETA",
                "retention_rule": "RETAIN_THROUGH_RELEASE",
                "stiffness_matrix_id": "K_KEEP",
            },
            {
                "joint_id": "J_DROP",
                "interface_id": "IF_ALPHA_BETA",
                "retention_rule": "REMOVE_AT_RELEASE",
                "stiffness_matrix_id": "K_DROP",
            },
        ]
    )
    q_values = {
        "TINY_LOCATE": np.array([-1.0, 0.2]),
        "TINY_JOIN": np.array([-0.8, -0.4]),
        "TINY_RELEASE": np.array([-0.5, 0.1]),
    }
    structural = np.array([[2.0, 0.15], [0.15, 1.5]])
    compliance = np.diag([0.1, 0.2])
    matrices: dict[str, np.ndarray] = {}
    manifest_rows: list[dict[str, str]] = []
    for operator_id, q in q_values.items():
        values = {
            f"Q_{operator_id}": q,
            f"W_STRUCT_{operator_id}": structural,
            f"CN_{operator_id}": compliance,
            f"W_TOTAL_{operator_id}": structural + compliance,
            f"U_FREE_{operator_id}": q.copy(),
        }
        matrices.update(values)
        for key, value in values.items():
            manifest_rows.append(
                {
                    "matrix_id": key,
                    "npz_file": "in_memory.npz",
                    "npz_key": key,
                    "shape": json.dumps(list(value.shape)),
                    "dtype": str(value.dtype),
                    "row_layout_id_optional": "VL_TWO_POINT",
                    "column_layout_id_optional": (
                        "VL_TWO_POINT" if value.ndim == 2 else ""
                    ),
                }
            )
    package.matrices = matrices
    package.raw_tables.update(
        {
            "I0/part.csv": package.parts.copy(),
            "I0/interface.csv": package.interfaces.copy(),
            "I0/assembly_topology.csv": route,
            "I0/joint_definition.csv": joints,
            "I0/sms_prior.csv": pd.DataFrame(
                columns=[
                    "sms_prior_id",
                    "part_id",
                    "prior_type",
                    "basis_id",
                    "alpha_mean",
                    "alpha_covariance_id",
                    "source_sample_ids",
                ]
            ),
            "I_Gamma/contact_domain.csv": pd.DataFrame(
                [
                    {
                        "contact_domain_id": "CD_ALPHA_BETA",
                        "interface_id": "IF_ALPHA_BETA",
                    }
                ]
            ),
            "I_Gamma/contact_point.csv": package.contact_points.copy(),
            "I_stage/boundary_item.csv": pd.DataFrame(columns=["boundary_id"]),
            "I_stage/load_item.csv": pd.DataFrame(columns=["load_id", "load_type"]),
            "matrices/vector_layout.csv": layout,
            "matrices/matrix_manifest.csv": pd.DataFrame(manifest_rows),
        }
    )
    return package
