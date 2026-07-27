from __future__ import annotations

import copy
from io import BytesIO
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from core.data_loader import load_package
from core.multi_part import vector_layout
from core.reporting import (
    build_runtime_report_zip,
    measurement_update_report_tables,
)
from core.stage_measurement_update import (
    FROZEN_PARAMETER_TARGETS,
    StageMeasurementUpdateValidationError,
    extract_observation_vector,
    linear_gaussian_update,
    load_measurement_checkpoints,
    load_measurement_observations,
    load_measurement_update_configs,
    load_state_update_basis,
    validate_measurement_update_package,
    validate_runtime_measurement_override,
)
from core.stage_solver import run_all_stages
from core.topology_step import (
    TopologyStepValidationError,
    run_topology_steps,
    topology_step_execution_table,
    topology_step_state_lineage_table,
)
import core.stage_measurement_update as update_module
from scripts.cli_check import _runtime_exit_code
from tests.stage_measurement_fixture_factory import (
    make_double_checkpoint_measurement_fixture,
    make_non12d_measurement_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "03_STAGE_MEASUREMENT_UPDATE_MIN_CASE"
PACKAGE_02 = ROOT / "data" / "02_TOPOLOGY_STEP_MIN_CASE"
PYTHON = Path(sys.executable)


def _copy_package():
    return copy.deepcopy(load_package(PACKAGE))


def _first_update(result):
    return next(
        item["measurement_update"]
        for item in result.values()
        if item.get("measurement_update") is not None
    )


class StageMeasurementDataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)

    def test_checkpoint_config_and_observation_loaders(self) -> None:
        checkpoints = load_measurement_checkpoints(self.package)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].source_topology_step_id, "TS204")
        self.assertEqual(len(load_measurement_observations(
            self.package, checkpoints[0].checkpoint_id
        )), 2)
        self.assertEqual(len(load_measurement_update_configs(self.package)), 1)
        self.assertEqual(len(load_state_update_basis(
            self.package, "ETA_STAGE_AFTER_ABC_2"
        )), 2)

    def test_checkpoint_foreign_keys_and_unique_route_mapping(self) -> None:
        report = validate_measurement_update_package(self.package)
        checks = dict(zip(report["check_item"], report["status"]))
        self.assertEqual(checks["checkpoint步骤外键可解析"], "PASS")
        self.assertEqual(
            checks["checkpoint与topology_step唯一对应"], "PASS"
        )
        self.assertEqual(
            checks["source_topology_step位于checkpoint之前"], "PASS"
        )

    def test_calibrate_and_update_roles_are_eligible(self) -> None:
        for role in ("CALIBRATE", "UPDATE"):
            package = _copy_package()
            table = package.raw_tables[
                "I_meas/measurement_record.csv"
            ].copy()
            table["data_role"] = role
            package.raw_tables[
                "I_meas/measurement_record.csv"
            ] = table
            update = _first_update(run_topology_steps(package))
            self.assertTrue(update.posterior_accepted)
            self.assertEqual(
                update.decision_record.accepted_measurement_ids,
                update.measurement_ids,
            )

    def test_validate_identify_and_missing_role_do_not_update(self) -> None:
        for role in ("VALIDATE", "IDENTIFY"):
            package = _copy_package()
            table = package.raw_tables[
                "I_meas/measurement_record.csv"
            ].copy()
            table["data_role"] = role
            package.raw_tables[
                "I_meas/measurement_record.csv"
            ] = table
            update = _first_update(run_topology_steps(package))
            self.assertFalse(update.posterior_accepted)
            self.assertIsNone(update.rollback_record)
            self.assertEqual(update.trace["status"], "EVALUATION_ONLY")
        package = _copy_package()
        table = package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy()
        table["data_role"] = ""
        package.raw_tables[
            "I_meas/measurement_record.csv"
        ] = table
        with self.assertRaises(TopologyStepValidationError):
            run_topology_steps(package)

    def test_frozen_parameter_and_sms_targets_are_blocked(self) -> None:
        self.assertTrue({"CN", "SMS"} <= FROZEN_PARAMETER_TARGETS)
        for target in ("CN", "SMS", "JOINT_STIFFNESS"):
            package = _copy_package()
            table = package.raw_tables[
                "I_meas/measurement_record.csv"
            ].copy()
            table["update_target"] = target
            package.raw_tables[
                "I_meas/measurement_record.csv"
            ] = table
            update = _first_update(run_topology_steps(package))
            self.assertFalse(update.posterior_accepted)
            self.assertIn("FROZEN", update.rollback_record.failure_reason)

    def test_unit_reference_coordinate_and_quality_mismatch_roll_back(self) -> None:
        cases = (
            ("unit", "inch"),
            ("reference_state_id_optional", "WRONG_STATE"),
            ("coordinate_system_id", "WRONG_CS"),
            ("quality_flag", "FAIL"),
        )
        for field, value in cases:
            package = _copy_package()
            table = package.raw_tables[
                "I_meas/measurement_record.csv"
            ].copy()
            table.loc[table.index[0], field] = value
            package.raw_tables[
                "I_meas/measurement_record.csv"
            ] = table
            try:
                update = _first_update(run_topology_steps(package))
            except TopologyStepValidationError:
                continue
            self.assertFalse(update.posterior_accepted, field)
            self.assertTrue(update.rollback_record.failure_reason)

    def test_matrix_shapes_and_psd_are_checked(self) -> None:
        for key, bad in (
            ("P_PRIOR_MCP_AFTER_ABC", np.ones((3, 3))),
            ("H_OBS_MCP_AFTER_ABC", np.ones((2, 3))),
            ("R_MEAS_MCP_AFTER_ABC", np.array([[1.0, 2.0], [2.0, 1.0]])),
            (
                "G_Q_UPDATE__MCP_AFTER_ABC__OP_TS204",
                np.ones((11, 2)),
            ),
        ):
            package = _copy_package()
            package.matrices[key] = bad
            try:
                update = _first_update(run_topology_steps(package))
            except TopologyStepValidationError:
                continue
            self.assertFalse(update.posterior_accepted, key)

    def test_missing_future_gq_is_blocking_quality_failure(self) -> None:
        package = _copy_package()
        package.matrices.pop(
            "G_Q_UPDATE__MCP_AFTER_ABC__OP_TS304"
        )
        report = validate_measurement_update_package(package)
        row = report[
            report["check_item"].eq("checkpoint及后续步骤G_q完整")
        ].iloc[0]
        self.assertEqual(row["status"], "FAIL")

    def test_runtime_override_only_changes_value_and_uncertainty(self) -> None:
        original = self.package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy(deep=True)
        override = pd.DataFrame([
            {
                "checkpoint_id": "MCP_AFTER_ABC",
                "measurement_id": original.iloc[0]["measurement_id"],
                "value": float(original.iloc[0]["value"]) + 1e-4,
                "standard_uncertainty": 0.0025,
            },
            {
                "checkpoint_id": "MCP_AFTER_ABC",
                "measurement_id": original.iloc[1]["measurement_id"],
                "value": float(original.iloc[1]["value"]),
                "standard_uncertainty": 0.003,
            },
        ])
        validated = validate_runtime_measurement_override(
            self.package, override
        )
        result = run_topology_steps(
            self.package, measurement_override=validated
        )
        self.assertEqual(
            _first_update(result).measurement_source,
            "RUNTIME_OVERRIDE",
        )
        pd.testing.assert_frame_equal(
            original,
            self.package.raw_tables["I_meas/measurement_record.csv"],
        )

    def test_invalid_runtime_override_does_not_mutate_package(self) -> None:
        original = self.package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy(deep=True)
        invalid = pd.DataFrame([{
            "measurement_id": original.iloc[0]["measurement_id"],
            "value": 0.0,
            "data_role": "CALIBRATE",
        }])
        with self.assertRaises(StageMeasurementUpdateValidationError):
            validate_runtime_measurement_override(self.package, invalid)
        pd.testing.assert_frame_equal(
            original,
            self.package.raw_tables["I_meas/measurement_record.csv"],
        )


class StageMeasurementMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.result = run_topology_steps(cls.package)
        cls.update = _first_update(cls.result)
        cls.oracle = pd.read_csv(
            PACKAGE / "validation" / "stage_measurement_update_oracle.csv",
            encoding="utf-8-sig",
        ).iloc[0]

    def test_eta_and_covariance_match_independent_oracle(self) -> None:
        tolerance = float(self.oracle["posterior_tolerance"])
        np.testing.assert_allclose(
            self.update.eta_posterior,
            json.loads(self.oracle["eta_posterior"]),
            rtol=0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            self.update.P_posterior,
            json.loads(self.oracle["P_posterior"]),
            rtol=0,
            atol=tolerance,
        )

    def test_q_lambda_and_gap_match_independent_oracle(self) -> None:
        tolerance = float(self.oracle["posterior_tolerance"])
        lcp_tolerance = float(self.oracle["lcp_tolerance"])
        np.testing.assert_allclose(
            self.update.q_posterior,
            json.loads(self.oracle["q_posterior"]),
            atol=tolerance,
        )
        active = np.asarray(
            json.loads(self.oracle["active_indices"]), dtype=int
        )
        np.testing.assert_allclose(
            self.update.lambda_full[active],
            json.loads(self.oracle["lambda_posterior_active"]),
            atol=lcp_tolerance,
        )
        np.testing.assert_allclose(
            self.update.gap_full[active],
            json.loads(self.oracle["gap_posterior_active"]),
            atol=lcp_tolerance,
        )

    def test_joseph_covariance_is_symmetric_psd_and_reduces_trace(self) -> None:
        self.assertEqual(
            self.update.trace["covariance_form"], "JOSEPH"
        )
        np.testing.assert_allclose(
            self.update.P_posterior,
            self.update.P_posterior.T,
            atol=1e-14,
        )
        self.assertGreaterEqual(
            np.linalg.eigvalsh(self.update.P_posterior).min(), -1e-14
        )
        self.assertLess(
            np.trace(self.update.P_posterior),
            np.trace(self.update.P_prior),
        )

    def test_nis_and_residual_reduction_match_oracle(self) -> None:
        self.assertAlmostEqual(
            self.update.nis, float(self.oracle["nis"]), places=5
        )
        self.assertGreater(
            self.update.trace["prior_residual_norm"], 0.0
        )
        self.assertLess(
            self.update.trace["posterior_residual_norm"],
            self.update.trace["prior_residual_norm"],
        )

    def test_full_covariance_block_changes_result(self) -> None:
        eta = np.zeros(2)
        P = np.array([[4e-4, 3e-5], [3e-5, 2.25e-4]])
        H = self.package.matrices["H_OBS_MCP_AFTER_ABC"]
        residual = self.update.innovation
        full = linear_gaussian_update(
            eta,
            P,
            H,
            self.package.matrices["R_MEAS_MCP_AFTER_ABC"],
            residual,
            np.zeros_like(residual),
        )
        diagonal = linear_gaussian_update(
            eta,
            P,
            H,
            np.diag([4e-6, 9e-6]),
            residual,
            np.zeros_like(residual),
        )
        self.assertGreater(
            np.linalg.norm(
                full["eta_posterior"] - diagonal["eta_posterior"]
            ),
            1e-10,
        )

    def test_no_explicit_matrix_inverse_is_used(self) -> None:
        source = inspect.getsource(linear_gaussian_update)
        self.assertNotIn("np.linalg.inv", source)
        self.assertNotIn("numpy.linalg.inv", source)
        self.assertIn("np.linalg.solve", source)


class StageMeasurementIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.result = run_topology_steps(cls.package)
        cls.update = _first_update(cls.result)

    def test_predicted_and_posterior_states_are_both_retained(self) -> None:
        step = self.result[self.update.topology_step_id]
        self.assertEqual(
            step["predicted_stage_state"].state_role, "PREDICTED"
        )
        self.assertEqual(
            step["posterior_stage_state"].state_role, "POSTERIOR"
        )
        self.assertEqual(
            step["posterior_stage_state"].posterior_parent_state_id,
            step["predicted_stage_state"].stage_state_id,
        )

    def test_q_correction_and_single_global_lcp(self) -> None:
        Gq = self.package.matrices[self.update.trace["G_q_key"]]
        np.testing.assert_allclose(
            self.update.q_correction_posterior,
            Gq @ self.update.eta_posterior,
        )
        original = update_module.solve_lcp_active_set
        with patch.object(
            update_module, "solve_lcp_active_set", wraps=original
        ) as solver:
            update = _first_update(run_topology_steps(self.package))
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(update.resolve_lcp_call_count, 1)
        self.assertTrue(update.trace["cross_interface_blocks_retained"])

    def test_complementarity_and_cross_interface_blocks_are_retained(self) -> None:
        self.assertLessEqual(
            self.update.physical_residuals[
                "complementarity_residual"
            ],
            1e-8,
        )
        source = self.result[self.update.source_topology_step_id]
        self.assertGreater(
            np.linalg.norm(source["W_struct"][0:3, 3:6]), 0.0
        )

    def test_next_step_inherits_posterior_and_applies_gq(self) -> None:
        keys = list(self.result)
        position = keys.index(self.update.topology_step_id)
        next_step = self.result[keys[position + 1]]
        self.assertEqual(
            next_step["predicted_stage_state"].parent_stage_state_id,
            self.update.posterior_state_id,
        )
        self.assertGreater(
            float(next_step["contact_trace"]["q_correction_norm"]),
            0.0,
        )

    def test_posterior_path_differs_and_disabled_path_matches_prior(self) -> None:
        disabled = run_topology_steps(
            self.package, measurement_update_enabled=False
        )
        self.assertFalse(np.allclose(
            self.result["TS301"]["q"], disabled["TS301"]["q"]
        ))
        baseline = run_topology_steps(load_package(PACKAGE_02))
        disabled_02 = run_topology_steps(
            load_package(PACKAGE_02),
            measurement_update_enabled=False,
        )
        for key in baseline:
            np.testing.assert_allclose(
                baseline[key]["q"], disabled_02[key]["q"]
            )

    def test_covariance_and_eta_propagate_with_explicit_fq(self) -> None:
        for key in ("TS301", "TS302", "TS303", "TS304"):
            state = self.result[key]["predicted_stage_state"]
            np.testing.assert_allclose(
                state.state_correction_vector,
                self.update.eta_posterior,
            )
            np.testing.assert_allclose(
                state.state_covariance,
                self.update.P_posterior,
            )
            self.assertEqual(
                state.covariance_source,
                "PACKAGE_STAGE_COVARIANCE_TRANSFER",
            )

    def test_lock_release_part_and_parameter_history_are_preserved(self) -> None:
        checkpoint = self.result[self.update.topology_step_id]
        predicted = checkpoint["predicted_stage_state"]
        posterior = checkpoint["posterior_stage_state"]
        self.assertEqual(
            predicted.connection_lock_history_ids,
            posterior.connection_lock_history_ids,
        )
        self.assertEqual(
            predicted.release_history_ids, posterior.release_history_ids
        )
        self.assertEqual(
            predicted.joint_lock_state, posterior.joint_lock_state
        )
        self.assertEqual(
            predicted.part_state.keys(), posterior.part_state.keys()
        )
        self.assertTrue(self.update.trace["parameter_frozen"])
        self.assertTrue(self.update.trace["sms_frozen"])

    def test_lineage_and_execution_reports_contain_dual_state_fields(self) -> None:
        lineage = topology_step_state_lineage_table(self.result)
        self.assertTrue(
            {"PREDICTED", "POSTERIOR"} <= set(lineage["state_role"])
        )
        execution = topology_step_execution_table(self.result)
        required = {
            "measurement_checkpoint_id", "predicted_state_id",
            "posterior_state_id", "effective_state_id",
            "measurement_update_status", "posterior_accepted",
            "covariance_trace", "state_covariance_trace",
            "state_correction_norm",
            "rollback_record_id",
        }
        self.assertTrue(required <= set(execution.columns))

    def test_nis_failure_rolls_back_without_replacing_effective_state(self) -> None:
        package = _copy_package()
        config = package.raw_tables[
            "I_meas/measurement_update_config.csv"
        ].copy()
        config["nis_threshold"] = 1e-12
        package.raw_tables[
            "I_meas/measurement_update_config.csv"
        ] = config
        result = run_topology_steps(package)
        update = _first_update(result)
        step = result[update.topology_step_id]
        self.assertFalse(update.posterior_accepted)
        self.assertEqual(
            step["stage_state"].stage_state_id,
            step["predicted_stage_state"].stage_state_id,
        )
        self.assertEqual(
            update.trace["status"], "POSTERIOR_REJECTED_ROLLBACK"
        )

    def test_lcp_failure_rolls_back(self) -> None:
        with patch.object(
            update_module,
            "solve_lcp_active_set",
            side_effect=RuntimeError("synthetic lcp failure"),
        ):
            result = run_topology_steps(self.package)
        update = _first_update(result)
        self.assertFalse(update.posterior_accepted)
        self.assertIn(
            "synthetic lcp failure",
            update.rollback_record.failure_reason,
        )

    def test_non12d_nonstandard_ids_fixture_passes(self) -> None:
        package = make_non12d_measurement_fixture()
        result = run_topology_steps(package)
        update = result["EV_MEASURE"]["measurement_update"]
        self.assertTrue(update.posterior_accepted)
        self.assertEqual(len(package.contact_points), 2)
        self.assertEqual(len(update.eta_posterior), 1)
        self.assertEqual(
            result["EV_JOIN"][
                "predicted_stage_state"
            ].parent_stage_state_id,
            update.posterior_state_id,
        )


class StageMeasurementIndependentAuditRepairTests(unittest.TestCase):
    """Regression tests mapped one-for-one to the independent audit list."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.result = run_topology_steps(cls.package)
        cls.update = _first_update(cls.result)
        cls.oracle = pd.read_csv(
            PACKAGE / "validation" / "stage_measurement_update_oracle.csv",
            encoding="utf-8-sig",
        ).iloc[0]

    @staticmethod
    def _run_with_roles(*roles: str):
        package = _copy_package()
        measurements = package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy()
        observations = package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ].copy()
        for index, role in enumerate(roles[:2]):
            measurements.loc[measurements.index[index], "data_role"] = role
        if len(roles) == 3:
            measurement = measurements.iloc[1].copy()
            measurement["measurement_id"] = "MEAS_IDENTIFY_AUDIT"
            measurement["data_role"] = roles[2]
            measurements = pd.concat(
                [measurements, measurement.to_frame().T],
                ignore_index=True,
            )
            observation = observations.iloc[1].copy()
            observation["observation_id"] = "OBS_IDENTIFY_AUDIT"
            observation["measurement_id"] = "MEAS_IDENTIFY_AUDIT"
            observation["observation_order"] = 2
            observation["sensitivity_row_index"] = 2
            observations = pd.concat(
                [observations, observation.to_frame().T],
                ignore_index=True,
            )
            H = package.matrices["H_OBS_MCP_AFTER_ABC"]
            R = package.matrices["R_MEAS_MCP_AFTER_ABC"]
            package.matrices["H_OBS_MCP_AFTER_ABC"] = np.vstack(
                [H, H[1]]
            )
            expanded_R = np.zeros((3, 3), dtype=float)
            expanded_R[:2, :2] = R
            expanded_R[2, 2] = R[1, 1]
            package.matrices["R_MEAS_MCP_AFTER_ABC"] = expanded_R
            manifest = package.raw_tables[
                "matrices/matrix_manifest.csv"
            ].copy()
            manifest.loc[
                manifest["matrix_id"].astype(str).eq(
                    "H_OBS_MCP_AFTER_ABC"
                ),
                "shape",
            ] = "[3, 2]"
            manifest.loc[
                manifest["matrix_id"].astype(str).eq(
                    "R_MEAS_MCP_AFTER_ABC"
                ),
                "shape",
            ] = "[3, 3]"
            package.raw_tables[
                "matrices/matrix_manifest.csv"
            ] = manifest
        package.raw_tables[
            "I_meas/measurement_record.csv"
        ] = measurements
        package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ] = observations
        return package, _first_update(run_topology_steps(package))

    @staticmethod
    def _assert_governance_blocked(testcase, package) -> None:
        try:
            update = _first_update(run_topology_steps(package))
        except TopologyStepValidationError:
            return
        testcase.assertFalse(update.posterior_accepted)
        testcase.assertIsNotNone(update.rollback_record)

    def test_audit_01_prior_observation_uses_unified_extractor(self) -> None:
        observations = [
            item for item in load_measurement_observations(
                self.package, self.update.checkpoint_id
            )
            if item.measurement_id in self.update.measurement_ids
        ]
        source = self.result[self.update.source_topology_step_id]
        np.testing.assert_allclose(
            extract_observation_vector(
                source, observations, vector_layout(self.package)
            ),
            self.update.z_predicted_prior_physical,
        )

    def test_audit_02_post_lcp_observation_uses_same_extractor(self) -> None:
        observations = [
            item for item in load_measurement_observations(
                self.package, self.update.checkpoint_id
            )
            if item.measurement_id in self.update.measurement_ids
        ]
        post_state = {
            "vector_layout_id":
                self.result[self.update.source_topology_step_id][
                    "stage_state"
                ].vector_layout_id,
            "active_interface_ids":
                self.result[self.update.source_topology_step_id][
                    "stage_state"
                ].active_interface_ids,
            "gap_full": self.update.gap_full,
            "lambda_full": self.update.lambda_full,
            "pressure": self.update.pressure,
            "local_compression": self.update.local_compression,
        }
        np.testing.assert_allclose(
            extract_observation_vector(
                post_state, observations, vector_layout(self.package)
            ),
            self.update.z_predicted_posterior_physical,
        )

    def test_audit_03_linearized_and_physical_results_are_separate(self) -> None:
        record = self.update.to_record()
        self.assertIn("z_predicted_posterior_linearized", record)
        self.assertIn("z_predicted_posterior_physical", record)
        self.assertGreater(
            np.linalg.norm(self.update.linearization_error), 0.0
        )

    def test_audit_04_acceptance_uses_physical_residual(self) -> None:
        self.assertEqual(
            self.update.trace["acceptance_basis"],
            "POST_LCP_PHYSICAL_OBSERVATION",
        )
        self.assertAlmostEqual(
            self.update.trace["posterior_residual_norm"],
            np.linalg.norm(self.update.residual_posterior_physical),
        )

    def test_audit_05_linear_improvement_physical_worsening_rolls_back(self) -> None:
        package = _copy_package()
        package.matrices["H_OBS_MCP_AFTER_ABC"] *= -1.0
        update = _first_update(run_topology_steps(package))
        self.assertLess(
            update.trace["posterior_linearized_residual_norm"],
            update.trace["prior_residual_norm"],
        )
        self.assertGreater(
            update.trace["posterior_residual_norm"],
            update.trace["prior_residual_norm"],
        )
        self.assertFalse(update.posterior_accepted)

    def test_audit_06_package_03_physical_residual_improves_and_accepts(self) -> None:
        self.assertTrue(self.update.posterior_accepted)
        self.assertTrue(self.update.physical_residual_improved)
        self.assertLess(
            np.linalg.norm(self.update.residual_posterior_physical),
            np.linalg.norm(self.update.residual_prior_physical),
        )

    def test_audit_07_weighted_physical_residual_improves(self) -> None:
        self.assertLess(
            self.update.weighted_residual_posterior_physical,
            self.update.weighted_residual_prior_physical,
        )

    def test_audit_08_oracle_checks_actual_post_lcp_observation(self) -> None:
        np.testing.assert_allclose(
            self.update.z_predicted_posterior_physical,
            json.loads(self.oracle["z_predicted_posterior_physical"]),
            atol=float(self.oracle["posterior_tolerance"]),
        )

    def test_audit_09_generator_h_is_independent_physical_fd(self) -> None:
        manifest = self.package.raw_tables[
            "matrices/matrix_manifest.csv"
        ]
        row = manifest[
            manifest["matrix_id"].astype(str).eq(
                "H_OBS_MCP_AFTER_ABC"
            )
        ].iloc[0]
        self.assertIn(
            "INDEPENDENT_GLOBAL_LCP_CENTRAL_FINITE_DIFFERENCE",
            str(row["derivation_source"]),
        )

    def test_audit_10_h_records_active_set_stability(self) -> None:
        self.assertTrue(bool(self.oracle["active_set_stable"]))
        self.assertEqual(
            json.loads(self.oracle["active_set_prior"]),
            json.loads(self.oracle["active_set_true"]),
        )

    def test_audit_11_second_zero_innovation_does_not_change_q(self) -> None:
        result = run_topology_steps(
            make_double_checkpoint_measurement_fixture()
        )
        second = result["EV_MEASURE_2"]["measurement_update"]
        self.assertLess(np.linalg.norm(second.eta_increment), 1e-12)
        self.assertLess(np.linalg.norm(second.q_update_increment), 1e-12)
        np.testing.assert_allclose(
            second.q_effective_after_update,
            second.q_source_effective,
            atol=1e-12,
        )

    def test_audit_12_double_checkpoint_lineage_is_correct(self) -> None:
        result = run_topology_steps(
            make_double_checkpoint_measurement_fixture()
        )
        first = result["EV_MEASURE"]["measurement_update"]
        second_step = result["EV_MEASURE_2"]
        second = second_step["measurement_update"]
        self.assertEqual(
            second_step["predicted_stage_state"].parent_stage_state_id,
            first.posterior_state_id,
        )
        self.assertEqual(
            second.posterior_state.posterior_parent_state_id,
            second.predicted_state_id,
        )

    def test_audit_13_downstream_step_applies_current_eta_once(self) -> None:
        package = make_double_checkpoint_measurement_fixture()
        result = run_topology_steps(package)
        downstream = result["EV_JOIN"]
        G_q = package.matrices[
            downstream["contact_trace"]["G_q_key"]
        ]
        eta = downstream[
            "predicted_stage_state"
        ].state_correction_vector
        np.testing.assert_allclose(
            downstream["q"],
            downstream["q_base"] + G_q @ eta,
            atol=1e-12,
        )

    def test_audit_14_sms_mutation_during_update_is_detected(self) -> None:
        package = _copy_package()
        original = update_module.solve_lcp_active_set

        def mutating_solver(*args, **kwargs):
            solution = original(*args, **kwargs)
            table = package.raw_tables["I0/sms_prior.csv"].copy()
            table.iloc[0, 0] = f"{table.iloc[0, 0]}_MUTATED"
            package.raw_tables["I0/sms_prior.csv"] = table
            return solution

        with patch.object(
            update_module,
            "solve_lcp_active_set",
            side_effect=mutating_solver,
        ):
            update = _first_update(run_topology_steps(package))
        self.assertFalse(update.posterior_accepted)
        self.assertIn(
            "TABLE::I0/sms_prior.csv",
            update.trace["changed_frozen_objects"],
        )

    def test_audit_15_parameter_mutation_during_update_is_detected(self) -> None:
        package = _copy_package()
        original = update_module.solve_lcp_active_set

        def mutating_solver(*args, **kwargs):
            solution = original(*args, **kwargs)
            table = package.raw_tables["I0/material.csv"].copy()
            table.iloc[0, 0] = f"{table.iloc[0, 0]}_MUTATED"
            package.raw_tables["I0/material.csv"] = table
            return solution

        with patch.object(
            update_module,
            "solve_lcp_active_set",
            side_effect=mutating_solver,
        ):
            update = _first_update(run_topology_steps(package))
        self.assertFalse(update.posterior_accepted)
        self.assertIn(
            "TABLE::I0/material.csv",
            update.trace["changed_frozen_objects"],
        )

    def test_audit_16_validate_does_not_enter_kalman_update(self) -> None:
        _, update = self._run_with_roles("CALIBRATE", "VALIDATE")
        self.assertEqual(update.kalman_gain.shape[1], 1)
        self.assertEqual(len(update.measurement_ids), 1)

    def test_audit_17_validate_does_not_block_calibrate(self) -> None:
        _, update = self._run_with_roles("CALIBRATE", "VALIDATE")
        self.assertTrue(update.posterior_accepted)
        self.assertEqual(len(update.evaluation_measurement_ids), 1)

    def test_audit_18_validate_has_prior_and_post_physical_residuals(self) -> None:
        _, update = self._run_with_roles("CALIBRATE", "VALIDATE")
        record = next(
            row for row in update.observation_records
            if row["observation_class"] == "EVALUATION_ONLY"
        )
        self.assertIn("prior_residual_physical", record)
        self.assertIn("posterior_residual_physical", record)

    def test_audit_19_identify_is_skipped(self) -> None:
        _, update = self._run_with_roles(
            "CALIBRATE", "VALIDATE", "IDENTIFY"
        )
        self.assertEqual(len(update.skipped_measurement_ids), 1)
        self.assertTrue(any(
            row["observation_class"] == "SKIPPED_IDENTIFY"
            for row in update.observation_records
        ))

    def test_audit_20_validate_only_is_not_update_or_rollback(self) -> None:
        _, update = self._run_with_roles("VALIDATE", "VALIDATE")
        self.assertEqual(update.trace["status"], "EVALUATION_ONLY")
        self.assertEqual(update.resolve_lcp_call_count, 0)
        self.assertFalse(update.posterior_accepted)
        self.assertIsNone(update.rollback_record)

    def test_audit_21_sample_id_mismatch_is_blocking(self) -> None:
        package = _copy_package()
        table = package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy()
        table.loc[table.index[0], "sample_id"] = "WRONG_SAMPLE"
        package.raw_tables[
            "I_meas/measurement_record.csv"
        ] = table
        self._assert_governance_blocked(self, package)

    def test_audit_22_reference_state_mismatch_is_blocking(self) -> None:
        package = _copy_package()
        table = package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ].copy()
        table.loc[
            table.index[0], "reference_state_id"
        ] = "WRONG_REFERENCE"
        package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ] = table
        self._assert_governance_blocked(self, package)

    def test_audit_23_unknown_vector_source_is_blocking(self) -> None:
        package = _copy_package()
        table = package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ].copy()
        table.loc[table.index[0], "vector_source"] = "UNKNOWN_VECTOR"
        package.raw_tables[
            "I_meas/measurement_observation_map.csv"
        ] = table
        self._assert_governance_blocked(self, package)

    def test_audit_24_stage_checkpoint_mismatch_is_blocking(self) -> None:
        package = _copy_package()
        table = package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy()
        table.loc[table.index[0], "stage_id"] = "WRONG_STAGE"
        package.raw_tables[
            "I_meas/measurement_record.csv"
        ] = table
        self._assert_governance_blocked(self, package)

    def test_audit_25_missing_declared_covariance_is_blocking(self) -> None:
        package = _copy_package()
        package.matrices.pop("R_MEAS_MCP_AFTER_ABC")
        self._assert_governance_blocked(self, package)

    def test_audit_26_explicit_diagonal_covariance_fallback_is_allowed(self) -> None:
        package = _copy_package()
        config = package.raw_tables[
            "I_meas/measurement_update_config.csv"
        ].copy()
        config["measurement_covariance_matrix_id_optional"] = ""
        config["allow_diagonal_covariance_fallback"] = True
        package.raw_tables[
            "I_meas/measurement_update_config.csv"
        ] = config
        measurements = package.raw_tables[
            "I_meas/measurement_record.csv"
        ].copy()
        measurements["covariance_block_id_optional"] = ""
        package.raw_tables[
            "I_meas/measurement_record.csv"
        ] = measurements
        update = _first_update(run_topology_steps(package))
        self.assertEqual(
            update.measurement_covariance_source,
            "DIAGONAL_STANDARD_UNCERTAINTY",
        )

    def test_audit_27_full_covariance_source_is_reported(self) -> None:
        self.assertEqual(
            self.update.measurement_covariance_source,
            "FULL_COVARIANCE_BLOCK",
        )

    def test_audit_28_mixed_units_use_weighted_residual(self) -> None:
        measurements = self.package.raw_tables[
            "I_meas/measurement_record.csv"
        ]
        self.assertGreater(len(set(measurements["unit"].astype(str))), 1)
        self.assertNotAlmostEqual(
            self.update.weighted_residual_prior_physical,
            np.linalg.norm(self.update.residual_prior_physical) ** 2,
        )

    def test_audit_29_ui_exposes_linearized_and_physical_results(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        for token in (
            "z_predicted_posterior_linearized",
            "z_predicted_posterior_physical",
            "physical_residual_improved",
            "acceptance_basis",
        ):
            self.assertIn(token, source)

    def test_audit_30_cli_reports_physical_residual_gate(self) -> None:
        completed = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "cli_check.py"), str(PACKAGE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "POSTERIOR_PHYSICAL_RESIDUAL_GATE=PASS",
            completed.stdout,
        )

    def test_audit_31_package_validator_checks_physical_residual(self) -> None:
        completed = subprocess.run(
            [
                str(PYTHON),
                str(
                    ROOT
                    / "scripts"
                    / "stage_measurement_update_package_validator.py"
                ),
                str(PACKAGE),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"passed": 38', completed.stdout)
        self.assertIn(
            "post-LCP physical residual improvement",
            completed.stdout,
        )

    def test_audit_32_old_02_package_does_not_regress(self) -> None:
        package = load_package(PACKAGE_02)
        enabled = run_topology_steps(package)
        disabled = run_topology_steps(
            load_package(PACKAGE_02),
            measurement_update_enabled=False,
        )
        self.assertEqual(enabled.keys(), disabled.keys())
        for key in enabled:
            np.testing.assert_allclose(
                enabled[key]["q"], disabled[key]["q"]
            )

    def test_audit_33_legacy_package_does_not_regress(self) -> None:
        package = load_package(ROOT / "data" / "E1_manual_input_9pt")
        result = run_all_stages(package)
        self.assertEqual(
            set(result),
            set(package.stage_plan["stage_id"].astype(str)),
        )

    def test_audit_34_nonformal_fixture_has_no_fixed_ids_or_12d_limit(self) -> None:
        package = make_non12d_measurement_fixture()
        update = _first_update(run_topology_steps(package))
        self.assertEqual(len(package.contact_points), 2)
        self.assertEqual(update.eta_posterior.size, 1)
        self.assertNotEqual(update.checkpoint_id, "MCP_AFTER_ABC")
        self.assertNotEqual(update.topology_step_id, "TS205")

    def test_audit_35_oracle_does_not_call_production_runner(self) -> None:
        source = (
            ROOT / "scripts" / "build_stage_measurement_update_fixture.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_stage_measurement_update", source)
        self.assertNotIn("run_topology_steps", source)
        self.assertFalse(bool(self.oracle["production_update_function_used"]))
        self.assertFalse(bool(self.oracle["production_topology_runner_used"]))


class StageMeasurementReportCliUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.result = run_all_stages(cls.package)

    def test_report_contains_all_required_measurement_artifacts(self) -> None:
        payload = build_runtime_report_zip(
            self.package, self.result, pd.DataFrame(), []
        )
        required = {
            "measurement_checkpoint_table.csv",
            "measurement_update_summary.csv",
            "measurement_innovation.csv",
            "measurement_observation_map.csv",
            "update_decision_record.csv",
            "resolve_requirement.csv",
            "posterior_state_snapshot.csv",
            "predicted_posterior_comparison.csv",
            "stage_covariance_trace.csv",
            "update_rollback_record.csv",
            "posterior_state_lineage.csv",
            "measurement_update_trace.json",
        }
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertTrue(required <= set(archive.namelist()))
            trace = json.loads(
                archive.read("measurement_update_trace.json")
            )
        self.assertEqual(len(trace), 1)
        self.assertTrue(trace[0]["trace"]["parameter_frozen"])

    def test_report_tables_have_measurement_source_and_truth_boundary(self) -> None:
        tables, _ = measurement_update_report_tables(
            self.package, self.result
        )
        summary = tables["measurement_update_summary.csv"]
        self.assertEqual(summary.iloc[0]["measurement_source"], "PACKAGE")
        self.assertFalse(
            bool(summary.iloc[0]["engineering_claim_allowed"])
        )

    def test_cli_03_machine_readable_summary(self) -> None:
        completed = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "cli_check.py"), str(PACKAGE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for line in (
            "FINAL_STATUS=PASS",
            "MEASUREMENT_CHECKPOINT_COUNT=1",
            "MEASUREMENT_UPDATE_ATTEMPT_COUNT=1",
            "POSTERIOR_ACCEPTED_COUNT=1",
            "POSTERIOR_ROLLBACK_COUNT=0",
            "MEASUREMENT_UPDATE_FAIL_COUNT=0",
        ):
            self.assertIn(line, completed.stdout)

    def test_cli_posterior_runtime_failure_maps_to_exit_two(self) -> None:
        self.assertEqual(
            _runtime_exit_code(
                0,
                "PASS",
                "PASS",
                measurement_update_fail_count=1,
            ),
            ("FAIL", 2),
        )

    def test_package_local_validator_exit_zero(self) -> None:
        completed = subprocess.run(
            [
                str(PYTHON),
                str(PACKAGE / "validation" / "validate_package.py"),
                str(PACKAGE),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"status": "PASS"', completed.stdout)

    def test_oracle_declares_no_production_update_or_runner(self) -> None:
        oracle = pd.read_csv(
            PACKAGE / "validation" / "stage_measurement_update_oracle.csv",
            encoding="utf-8-sig",
        )
        self.assertFalse(
            oracle["production_update_function_used"].astype(bool).any()
        )
        self.assertFalse(
            oracle["production_topology_runner_used"].astype(bool).any()
        )

    def test_core_does_not_contain_fixture_specific_ids(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "core" / "stage_measurement_update.py",
                ROOT / "core" / "topology_step.py",
            )
        )
        for token in ("TS205", "MCP_AFTER_ABC", "G_AB", "G_BC"):
            self.assertNotIn(token, corpus)

    def test_streamlit_page_15_and_no_checkpoint_message(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(
            str(ROOT / "app.py"), default_timeout=90
        ).run()
        selector = next(
            box for box in app.selectbox
            if any(PACKAGE.name in str(value) for value in box.options)
        )
        option_03 = next(
            value for value in selector.options if PACKAGE.name in value
        )
        selector.set_value(option_03).run(timeout=90)
        navigation = next(
            radio for radio in app.radio
            if radio.key == "main_page_navigation"
        )
        navigation.set_value(navigation.options[-2]).run(timeout=90)
        self.assertFalse(app.exception)
        self.assertTrue(any(
            box.label == "checkpoint 选择" for box in app.selectbox
        ))
        self.assertTrue(any(
            "POSTERIOR_ACCEPTED" in str(item.value)
            for item in app.success
        ))

        selector = next(
            box for box in app.selectbox
            if any(PACKAGE_02.name in str(value) for value in box.options)
        )
        option_02 = next(
            value for value in selector.options
            if PACKAGE_02.name in value
        )
        selector.set_value(option_02).run(timeout=90)
        navigation = next(
            radio for radio in app.radio
            if radio.key == "main_page_navigation"
        )
        navigation.set_value(navigation.options[-2]).run(timeout=90)
        self.assertFalse(app.exception)
        self.assertTrue(any(
            "未配置阶段实测后验更新" in str(item.value)
            for item in app.info
        ))


if __name__ == "__main__":
    unittest.main()
