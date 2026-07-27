from __future__ import annotations

import copy
from io import BytesIO
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from core.data_loader import load_package
from core.reporting import (
    build_rolling_prediction_report_zip,
    rolling_prediction_report_tables,
)
from core.rolling_prediction import (
    EMPIRICAL_FRACTION_LABEL,
    RollingPredictionRuntimeError,
    RollingPredictionValidationError,
    has_rolling_prediction_plans,
    load_rolling_prediction_plans,
    load_virtual_sms_components,
    load_virtual_sms_sample_sets,
    load_virtual_sms_samples,
    run_posterior_virtual_sms_rolling_prediction,
    stable_hash,
    stable_package_file_hash,
    validate_rolling_prediction_package,
)
from core.topology_step import run_topology_steps
import core.rolling_prediction as rolling_module
import core.topology_step as topology_module
from scripts.build_posterior_virtual_sms_rolling_fixture import (
    build as build_fixture,
    package_digest,
)
from scripts.cli_check import (
    _runtime_exit_code,
    _unique_plan_sample_count,
)
from scripts.posterior_virtual_sms_rolling_package_validator import (
    validate_package as validate_rolling_fixture,
)
from tests.rolling_prediction_fixture_factory import (
    make_non12d_rolling_prediction_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE"
PACKAGE_03 = ROOT / "data" / "03_STAGE_MEASUREMENT_UPDATE_MIN_CASE"
PLAN_ID = "ROLL_PLAN_ABC_TO_ABCD"


class PosteriorVirtualSMSRollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.topology = run_topology_steps(cls.package)
        cls.result = run_posterior_virtual_sms_rolling_prediction(
            cls.package, cls.topology, PLAN_ID
        )

    def test_plan_source_and_legacy_compatibility(self) -> None:
        plans = load_rolling_prediction_plans(self.package)
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.source_checkpoint_id, "MCP_AFTER_ABC")
        self.assertEqual(plan.source_topology_step_id, "TS205")
        self.assertEqual(plan.prediction_start_step_id, "TS301")
        self.assertNotIn(
            plan.future_part_ids[0],
            self.result.source_posterior_state.active_part_ids,
        )
        self.assertTrue(
            self.result.source_posterior_state.posterior_accepted
        )
        self.assertEqual(
            self.result.source_posterior_state.state_role, "POSTERIOR"
        )
        self.assertEqual(
            self.result.sample_results[0].effective_source_state_id,
            self.result.source_posterior_state.stage_state_id,
        )
        old = load_package(PACKAGE_03)
        self.assertFalse(has_rolling_prediction_plans(old))
        self.assertTrue(validate_rolling_prediction_package(old).empty)

    def test_source_historical_and_package_hashes_are_immutable(self) -> None:
        trace = self.result.trace
        self.assertEqual(trace["immutability_status"], "PASS")
        self.assertEqual(
            trace["source_state_hashes_before"],
            trace["source_state_hashes_after"],
        )
        self.assertEqual(
            trace["source_state_hashes_before"]["posterior"],
            stable_hash(self.result.source_posterior_state),
        )
        self.assertIn("historical_topology",
                      trace["source_state_hashes_before"])
        self.assertIn("package", trace["source_state_hashes_before"])

    def test_invalid_source_posterior_blocks_formal_run(self) -> None:
        topology = copy.deepcopy(self.topology)
        source = topology["TS205"]["posterior_stage_state"]
        source.posterior_accepted = False
        with self.assertRaises(RollingPredictionValidationError):
            run_posterior_virtual_sms_rolling_prediction(
                self.package, topology, PLAN_ID
            )

    def test_sample_library_component_order_reference_and_delta(self) -> None:
        sample_sets = load_virtual_sms_sample_sets(self.package)
        samples = load_virtual_sms_samples(self.package)
        components = load_virtual_sms_components(self.package)
        self.assertEqual(sample_sets[0].sample_count, 5)
        self.assertEqual(len(samples), 5)
        self.assertEqual(
            [item.component_order for item in components], [0, 1]
        )
        reference = next(
            item for item in self.result.sample_results
            if item.virtual_sms_sample_id == "VSMS_REF"
        )
        offset = next(
            item for item in self.result.sample_results
            if item.virtual_sms_sample_id == "VSMS_M1_POS"
        )
        self.assertEqual(reference.sms_coefficients["P_D"], [0.0, 0.0])
        self.assertEqual(reference.sms_delta_coefficients["P_D"], [0.0, 0.0])
        self.assertEqual(offset.sms_delta_coefficients["P_D"], [1.0, 0.0])
        self.assertFalse(sample_sets[0].probability_interpretation_allowed)
        self.assertFalse(sample_sets[0].engineering_claim_allowed)

    def test_virtual_sms_applies_once_only_on_future_steps(self) -> None:
        reference = next(
            item for item in self.result.sample_results
            if item.virtual_sms_sample_id == "VSMS_REF"
        )
        offset = next(
            item for item in self.result.sample_results
            if item.virtual_sms_sample_id == "VSMS_M1_POS"
        )
        self.assertTrue(all(
            np.allclose(step["q_virtual_sms_correction"], 0.0)
            for step in reference.step_results.values()
        ))
        self.assertTrue(all(
            np.linalg.norm(step["q_virtual_sms_correction"]) > 0.0
            for step in offset.step_results.values()
            if step["topology_step_spec"].solve_required
        ))
        traces = offset.trace["sms_application_trace"]
        keys = [
            (
                row["part_id"], row["topology_step_id"],
                row["virtual_sms_sample_id"],
            )
            for row in traces
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(row["application_count"] == 1 for row in traces))
        self.assertTrue(all(row["part_id"] == "P_D" for row in traces))
        self.assertTrue(all(
            row["topology_step_id"].startswith("TS3") for row in traces
        ))

    def test_mapping_missing_and_shape_layout_mismatch_block(self) -> None:
        missing = copy.deepcopy(self.package)
        table = missing.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ].copy()
        missing.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ] = table.iloc[1:].reset_index(drop=True)
        gates = validate_rolling_prediction_package(missing)
        self.assertTrue(gates["status"].eq("FAIL").any())

        mismatch = copy.deepcopy(self.package)
        matrix_id = str(table.iloc[0]["matrix_id"])
        mismatch.matrices[matrix_id] = np.zeros((11, 2))
        gates = validate_rolling_prediction_package(mismatch)
        self.assertTrue(gates["status"].eq("FAIL").any())

    def test_explicit_zero_mapping_is_valid(self) -> None:
        package = copy.deepcopy(self.package)
        mappings = package.raw_tables["I_pred/sms_operator_mapping.csv"]
        for matrix_id in mappings["matrix_id"].astype(str):
            package.matrices[matrix_id] = np.zeros_like(
                package.matrices[matrix_id]
            )
            package.raw_tables[
                "matrices/matrix_manifest.csv"
            ].loc[
                package.raw_tables[
                    "matrices/matrix_manifest.csv"
                ]["matrix_id"].astype(str).eq(matrix_id),
                "mapping_role_optional",
            ] = "EXPLICIT_ZERO_NO_EFFECT"
        package.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ]["mapping_role"] = "EXPLICIT_ZERO_NO_EFFECT"
        gates = validate_rolling_prediction_package(package, self.topology)
        self.assertFalse(gates["status"].eq("FAIL").any())
        run = run_posterior_virtual_sms_rolling_prediction(
            package, self.topology, PLAN_ID, ["VSMS_M1_POS"]
        )
        self.assertTrue(all(
            np.allclose(step["q_virtual_sms_correction"], 0.0)
            for step in run.sample_results[0].step_results.values()
        ))
        self.assertTrue(all(
            row["mapping_role"] == "EXPLICIT_ZERO_NO_EFFECT"
            for row in run.sample_results[0].trace[
                "sms_application_trace"
            ]
        ))

    def test_q_decomposition_and_operator_base_semantics(self) -> None:
        for sample in self.result.sample_results:
            previous_base = None
            for step in sample.step_results.values():
                reconstructed = (
                    step["q_operator_base"]
                    + step["q_posterior_state_correction"]
                    + step["q_virtual_sms_correction"]
                    + step["q_future_process_correction"]
                )
                np.testing.assert_allclose(step["q"], reconstructed)
                self.assertAlmostEqual(
                    np.linalg.norm(
                        step["q_posterior_state_correction"]
                    ),
                    step["contact_trace"][
                        "q_posterior_state_correction_norm"
                    ],
                )
                if previous_base is not None:
                    self.assertFalse(
                        np.shares_memory(step["q_operator_base"], previous_base)
                    )
                previous_base = step["q_operator_base"]

    def test_topology_branch_lineage_global_lcp_and_physics(self) -> None:
        final_ids = set()
        for sample in self.result.sample_results:
            self.assertEqual(
                list(sample.step_results), ["TS301", "TS302", "TS303", "TS304"]
            )
            first = sample.step_results["TS301"]["stage_state"]
            self.assertEqual(
                first.parent_stage_state_id,
                sample.effective_source_state_id,
            )
            final_ids.add(sample.final_state_id)
            for step in sample.step_results.values():
                self.assertEqual(step["lcp_call_count"], 1)
                self.assertLessEqual(
                    step["solution"].residuals[
                        "complementarity_residual"
                    ],
                    1e-8,
                )
                self.assertTrue(np.all(step["lambda_full"] >= -1e-9))
                active = step["active_index_mask"]
                self.assertEqual(
                    step["W_total"][np.ix_(active, active)].shape,
                    step["W_active"].shape,
                )
            cross = sample.trace["q_decomposition"][0]["cross_block_norms"]
            self.assertTrue(cross)
            self.assertGreater(max(cross.values()), 0.0)
            self.assertTrue(
                sample.step_results["TS303"]["connection_lock_history"]
            )
            self.assertTrue(
                sample.step_results["TS304"]["release_history"]
            )
            self.assertTrue(
                sample.step_results["TS304"][
                    "stage_state"
                ].joint_lock_state
            )
        self.assertEqual(len(final_ids), 5)

    def test_order_invariance_single_batch_and_repeatability(self) -> None:
        reversed_run = run_posterior_virtual_sms_rolling_prediction(
            self.package,
            self.topology,
            PLAN_ID,
            list(reversed([
                item.virtual_sms_sample_id
                for item in self.result.sample_results
            ])),
        )
        left = self.result.kcp_predictions.set_index(
            ["virtual_sms_sample_id", "kcp_id"]
        )["predicted_value"].sort_index()
        right = reversed_run.kcp_predictions.set_index(
            ["virtual_sms_sample_id", "kcp_id"]
        )["predicted_value"].sort_index()
        np.testing.assert_allclose(left, right, atol=1e-12)
        one = run_posterior_virtual_sms_rolling_prediction(
            self.package, self.topology, PLAN_ID, ["VSMS_COMBO"]
        )
        expected = left.loc["VSMS_COMBO"].sort_index()
        actual = one.kcp_predictions.set_index("kcp_id")[
            "predicted_value"
        ].sort_index()
        np.testing.assert_allclose(expected, actual, atol=1e-12)
        self.assertEqual(
            stable_hash(one.kcp_predictions),
            stable_hash(run_posterior_virtual_sms_rolling_prediction(
                self.package, self.topology, PLAN_ID, ["VSMS_COMBO"]
            ).kcp_predictions),
        )

    def test_sample_failure_is_isolated(self) -> None:
        production_branch = rolling_module._run_sample_branch

        def isolated_failure(*args, **kwargs):
            sample_id = args[5]
            source_role = args[7]
            if sample_id == "VSMS_M1_NEG" and source_role == "POSTERIOR":
                raise RuntimeError("synthetic isolated branch failure")
            return production_branch(*args, **kwargs)

        with patch.object(
            rolling_module, "_run_sample_branch",
            side_effect=isolated_failure,
        ):
            run = run_posterior_virtual_sms_rolling_prediction(
                self.package, self.topology, PLAN_ID
            )
        self.assertEqual(len(run.sample_failures), 1)
        self.assertEqual(
            run.sample_failures[0].virtual_sms_sample_id, "VSMS_M1_NEG"
        )
        self.assertEqual(len(run.sample_results), 4)
        self.assertEqual(run.quality_status, "FAIL")
        self.assertEqual(
            run.status_summary["formal_failure_count"], 1
        )

    def test_kcp_ledger_oracle_and_baseline_comparison(self) -> None:
        self.assertEqual(len(self.result.kcp_predictions), 15)
        self.assertEqual(len(self.result.baseline_comparison), 15)
        self.assertTrue(
            self.result.kcp_predictions["double_count_status"].eq(
                "PASS"
            ).all()
        )
        self.assertTrue(
            self.result.kcp_predictions["quality_status"].eq("PASS").all()
        )
        self.assertTrue(
            self.result.kcp_predictions["aggregation_policy"].eq(
                "ONE_SMS_PER_PART_PLUS_ONE_INCREMENT_PER_INTERFACE_STAGE"
            ).all()
        )
        self.assertTrue(
            self.result.kcp_predictions["tolerance_status"].isin(
                ["WITHIN_OR_NOT_DEFINED", "EXCEED"]
            ).all()
        )
        oracle = pd.read_csv(
            PACKAGE / "validation" / "rolling_kcp_oracle.csv",
            encoding="utf-8-sig",
        )
        posterior_oracle = oracle[
            oracle["source_state_role"].astype(str).eq("POSTERIOR")
        ]
        merged = posterior_oracle.merge(
            self.result.kcp_predictions,
            on=["virtual_sms_sample_id", "kcp_id"],
            suffixes=("_oracle", "_actual"),
        )
        np.testing.assert_allclose(
            merged["predicted_value_oracle"],
            merged["predicted_value_actual"],
            atol=5e-8,
        )
        self.assertTrue(
            oracle["production_kcp_entry_used"]
            .astype(str).str.lower().eq("false").all()
        )

    def test_descriptive_summary_truth_and_contact_modes(self) -> None:
        summary = self.result.descriptive_summary
        table = summary.kcp_summary
        self.assertEqual(summary.sample_count, 5)
        self.assertEqual(summary.success_count, 5)
        self.assertEqual(summary.failure_count, 0)
        self.assertFalse(summary.probability_interpretation_allowed)
        self.assertFalse(summary.engineering_claim_allowed)
        self.assertTrue(table["fraction_semantics"].eq(
            EMPIRICAL_FRACTION_LABEL
        ).all())
        self.assertTrue(table[
            "probability_interpretation_allowed"
        ].eq(False).all())
        self.assertFalse(any(
            "pf" in column.lower() for column in table.columns
        ))
        self.assertEqual(int(table["count"].sum()), 15)
        self.assertFalse(self.result.contact_mode_summary.empty)
        expected = json.loads(
            (PACKAGE / "validation" / "rolling_expected_summary.json")
            .read_text(encoding="utf-8")
        )
        for kcp_id, values in expected["kcp_summary"].items():
            row = table.set_index("kcp_id").loc[kcp_id]
            self.assertAlmostEqual(
                float(row["descriptive_mean"]), values["mean"], places=10
            )
            self.assertAlmostEqual(
                float(row["empirical_p95"]), values["p95"], places=10
            )

    def test_report_contains_all_required_rolling_artifacts(self) -> None:
        payload = build_rolling_prediction_report_zip(
            self.package, self.result
        )
        names = set(zipfile.ZipFile(BytesIO(payload)).namelist())
        expected = {
            "rolling_prediction_plan.csv",
            "virtual_sms_sample_manifest.csv",
            "virtual_sms_coefficients.csv",
            "future_sms_assignment.csv",
            "sms_operator_mapping.csv",
            "rolling_sample_summary.csv",
            "rolling_step_execution.csv",
            "rolling_sms_application_trace.csv",
            "rolling_state_lineage.csv",
            "rolling_kcp_predictions.csv",
            "rolling_contribution_ledger.csv",
            "rolling_kcp_summary.csv",
            "rolling_baseline_comparison.csv",
            "rolling_contact_mode_summary.csv",
            "rolling_sample_failure.csv",
            "rolling_quality_gate.csv",
            "rolling_prediction_trace.json",
        }
        self.assertEqual(names, expected)

    def test_fixture_governance_and_local_validator(self) -> None:
        report = validate_rolling_fixture(PACKAGE)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["passed"], report["total"])
        self.assertEqual(report["blocking_fail_count"], 0)
        manifest = pd.read_csv(
            PACKAGE / "matrices" / "matrix_manifest.csv",
            encoding="utf-8-sig",
        )
        with np.load(
            PACKAGE / "matrices" / "multi_part_matrices.npz"
        ) as archive:
            self.assertEqual(
                set(manifest["npz_key"].astype(str)), set(archive.files)
            )
        objects = pd.read_csv(
            PACKAGE / "object_file_map.csv", encoding="utf-8-sig"
        )
        fields = pd.read_csv(
            PACKAGE / "field_dictionary.csv", encoding="utf-8-sig"
        )
        self.assertIn("RollingPredictionPlan",
                      set(objects["object_name"]))
        self.assertIn(
            "I_pred/rolling_prediction_plan.csv",
            set(fields["file_path"]),
        )

    def test_generator_is_reproducible(self) -> None:
        first = build_fixture()
        second = build_fixture()
        self.assertEqual(first["validation_status"], "PASS")
        self.assertEqual(second["validation_status"], "PASS")
        self.assertEqual(
            first["package_digest"], second["package_digest"]
        )
        self.assertEqual(
            package_digest(PACKAGE), second["package_digest"]
        )

    def test_cli_summary_and_runtime_failure_exit_contract(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cli_check.py"),
             str(PACKAGE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for line in (
            "ROLLING_PLAN_COUNT=1",
            "ROLLING_RUN_COUNT=1",
            "ROLLING_SAMPLE_COUNT=5",
            "ROLLING_SUCCESS_COUNT=5",
            "ROLLING_FAILURE_COUNT=0",
            "ROLLING_KCP_RESULT_COUNT=15",
            "ROLLING_REFERENCE_SAMPLE_PASS_COUNT=1",
            "ROLLING_PHYSICAL_FAIL_COUNT=0",
            "ROLLING_DOUBLE_COUNT_FAIL_COUNT=0",
            "ROLLING_PROBABILITY_INTERPRETATION_ALLOWED=false",
            "ROLLING_FINAL_STATUS=PASS",
        ):
            self.assertIn(line, completed.stdout)
        old = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cli_check.py"),
             str(PACKAGE_03)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(old.returncode, 0, old.stderr)
        self.assertIn("ROLLING_PLAN_COUNT=0", old.stdout)
        self.assertIn("ROLLING_FINAL_STATUS=NOT_APPLICABLE", old.stdout)
        self.assertEqual(
            _runtime_exit_code(0, "PASS", "PASS",
                               rolling_failure_count=1),
            ("FAIL", 2),
        )

    def test_no_formal_fixture_ids_are_hardcoded_in_core(self) -> None:
        source = inspect.getsource(rolling_module) + inspect.getsource(
            topology_module.run_topology_steps_from_state
        )
        for fixture_id in (
            "ROLL_PLAN_ABC_TO_ABCD",
            "VSMS_REF",
            "MCP_AFTER_ABC",
            "TS301",
            "P_D",
            "VL_CONTACT_ALL_12",
        ):
            self.assertNotIn(fixture_id, source)

    def test_non12d_in_memory_fixture_runs_formal_entry(self) -> None:
        package = make_non12d_rolling_prediction_fixture()
        self.assertEqual(len(package.contact_points), 3)
        topology = run_topology_steps(package)
        gates = validate_rolling_prediction_package(package, topology)
        self.assertFalse(gates["status"].eq("FAIL").any())
        result = run_posterior_virtual_sms_rolling_prediction(
            package, topology, "PLAN_TINY_POSTERIOR"
        )
        self.assertEqual(len(result.sample_results), 2)
        self.assertFalse(result.sample_failures)
        offset = next(
            item for item in result.sample_results
            if item.virtual_sms_sample_id == "TINY_OFFSET"
        )
        self.assertTrue(all(
            step["q"].shape == (3,)
            for step in offset.step_results.values()
        ))
        self.assertGreater(
            offset.kcp_prediction_result.iloc[0]["predicted_value"], 0.0
        )

    def test_streamlit_page_16_plan_sample_and_legacy_message(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(
            str(ROOT / "app.py"), default_timeout=90
        ).run()
        package_selector = next(
            box for box in app.selectbox
            if any(PACKAGE.name in str(option)
                   for option in box.options)
        )
        option_04 = next(
            option for option in package_selector.options
            if PACKAGE.name in str(option)
        )
        package_selector.set_value(option_04).run(timeout=90)
        navigation = next(
            radio for radio in app.radio
            if radio.key == "main_page_navigation"
        )
        self.assertEqual(len(navigation.options), 16)
        navigation.set_value(navigation.options[-1]).run(timeout=90)
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any(
            box.label == "rolling plan 选择" for box in app.selectbox
        ))
        self.assertTrue(any(
            item.label == "virtual SMS 样本选择"
            for item in app.multiselect
        ))
        rendered = "\n".join(
            str(item.value)
            for collection in (
                app.warning, app.caption, app.subheader, app.markdown
            )
            for item in collection
        )
        self.assertIn("不是 Monte Carlo", rendered)
        self.assertIn("失效概率", rendered)
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("下载后验虚拟 SMS 滚动预测报告 ZIP", source)

        package_selector = next(
            box for box in app.selectbox
            if any(PACKAGE_03.name in str(option)
                   for option in box.options)
        )
        option_03 = next(
            option for option in package_selector.options
            if PACKAGE_03.name in str(option)
        )
        package_selector.set_value(option_03).run(timeout=90)
        navigation = next(
            radio for radio in app.radio
            if radio.key == "main_page_navigation"
        )
        navigation.set_value(navigation.options[-1]).run(timeout=90)
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any(
            "当前数据包未配置后验状态驱动的虚拟 SMS 滚动预测"
            in str(item.value)
            for item in app.info
        ))
        navigation.set_value(navigation.options[-2]).run(timeout=90)
        self.assertFalse(app.exception, app.exception)
        self.assertFalse(any(
            item.value == "后验状态驱动的虚拟 SMS 滚动预测"
            for item in app.subheader
        ))


class PosteriorVirtualSMSRollingAuditRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_package(PACKAGE)
        cls.topology = run_topology_steps(cls.package)
        cls.result = run_posterior_virtual_sms_rolling_prediction(
            cls.package, cls.topology, PLAN_ID
        )

    @staticmethod
    def _blocking_failures(package) -> pd.DataFrame:
        gates = validate_rolling_prediction_package(package)
        return gates[
            gates["status"].astype(str).eq("FAIL")
            & gates["blocking"].astype(bool)
        ]

    def test_audit_01_authoritative_status_is_shared_everywhere(self) -> None:
        result = self.result
        self.assertEqual(result.quality_status, "PASS")
        self.assertEqual(
            result.quality_status,
            result.descriptive_summary.quality_status,
        )
        self.assertEqual(
            result.quality_status, result.trace["quality_status"]
        )
        self.assertEqual(
            result.quality_status,
            result.status_summary["run_quality_status"],
        )
        self.assertFalse(result.quality_gates["status"].eq("FAIL").any())
        tables, trace = rolling_prediction_report_tables(
            self.package, result
        )
        self.assertEqual(trace["run_quality_status"], "PASS")
        self.assertTrue(
            tables["rolling_sample_summary.csv"][
                "run_quality_status"
            ].eq("PASS").all()
        )
        self.assertEqual(
            result.status_summary["formal_sample_count"], 5
        )
        self.assertEqual(
            result.status_summary["baseline_attempt_count"], 5
        )

    def test_audit_02_duplicate_kcp_propagates_sample_run_report_cli(self) -> None:
        package = copy.deepcopy(self.package)
        configs = package.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ].copy()
        package.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ] = pd.concat([configs, configs.iloc[[0]]], ignore_index=True)
        gates = validate_rolling_prediction_package(
            package, self.topology
        )
        duplicate_gate = gates[
            gates["check_item"].astype(str).str.contains(
                "KCP contribution uniqueness"
            )
        ].iloc[0]
        self.assertEqual(duplicate_gate["status"], "FAIL")
        self.assertFalse(bool(duplicate_gate["blocking"]))
        result = run_posterior_virtual_sms_rolling_prediction(
            package, self.topology, PLAN_ID
        )
        self.assertEqual(result.quality_status, "FAIL")
        self.assertEqual(result.trace["quality_status"], "FAIL")
        self.assertEqual(len(result.sample_failures), 5)
        self.assertTrue(all(
            item.quality_status == "FAIL"
            and item.double_count_fail_count > 0
            and item.kcp_prediction_result[
                "double_count_status"
            ].eq("FAIL").all()
            and item.kcp_prediction_result[
                "quality_status"
            ].eq("FAIL").all()
            for item in result.sample_failures
        ))
        self.assertGreater(
            result.status_summary["double_count_fail_count"], 0
        )
        self.assertTrue(
            result.quality_gates["status"].eq("FAIL").any()
        )
        tables, trace = rolling_prediction_report_tables(package, result)
        self.assertEqual(trace["run_quality_status"], "FAIL")
        self.assertTrue(
            tables["rolling_sample_failure.csv"][
                "run_quality_status"
            ].eq("FAIL").all()
        )
        self.assertTrue(
            tables["rolling_sample_summary.csv"][
                "sample_quality_status"
            ].eq("FAIL").all()
        )
        self.assertTrue(
            tables["rolling_kcp_predictions.csv"][
                "sample_quality_status"
            ].eq("FAIL").all()
        )
        self.assertGreater(
            int(tables["rolling_sample_failure.csv"][
                "double_count_fail_count"
            ].sum()),
            0,
        )
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn(
            'if rolling_result.quality_status == "PASS"', app_source
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / PACKAGE.name
            shutil.copytree(PACKAGE, target)
            cli_configs = pd.read_csv(
                target / "I_pred" / "rolling_kcp_config.csv",
                encoding="utf-8-sig",
            )
            pd.concat(
                [cli_configs, cli_configs.iloc[[0]]],
                ignore_index=True,
            ).to_csv(
                target / "I_pred" / "rolling_kcp_config.csv",
                index=False,
                encoding="utf-8-sig",
            )
            (target / "validation" / "validate_package.py").unlink()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "cli_check.py"),
                    str(target),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("ROLLING_FINAL_STATUS=FAIL", completed.stdout)
        line = next(
            item for item in completed.stdout.splitlines()
            if item.startswith("ROLLING_DOUBLE_COUNT_FAIL_COUNT=")
        )
        self.assertGreater(int(line.split("=", 1)[1]), 0)

    def test_audit_03_assignment_interval_faults_are_blocking(self) -> None:
        mutations = (
            (
                "bad first",
                "first_effective_topology_step_id",
                "BAD_STEP",
            ),
            (
                "bad last",
                "last_effective_topology_step_id_optional",
                "BAD_STEP",
            ),
            (
                "first after last",
                "first_effective_topology_step_id",
                "TS304",
            ),
            (
                "first differs from part add",
                "first_effective_topology_step_id",
                "TS302",
            ),
        )
        for name, column, value in mutations:
            with self.subTest(name=name):
                package = copy.deepcopy(self.package)
                table = package.raw_tables[
                    "I_pred/future_sms_assignment.csv"
                ].copy()
                table.loc[0, column] = value
                if name == "first after last":
                    table.loc[
                        0, "last_effective_topology_step_id_optional"
                    ] = "TS301"
                package.raw_tables[
                    "I_pred/future_sms_assignment.csv"
                ] = table
                gates = validate_rolling_prediction_package(
                    package, self.topology
                )
                failures = gates[
                    gates["status"].astype(str).eq("FAIL")
                    & gates["blocking"].astype(bool)
                ]
                self.assertTrue(
                    failures["check_item"].astype(str).str.contains(
                        "assignment"
                    ).any(),
                    failures.to_string(index=False),
                )

    def test_audit_04_bad_step_cli_is_blocking_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / PACKAGE.name
            shutil.copytree(PACKAGE, target)
            assignments = pd.read_csv(
                target / "I_pred" / "future_sms_assignment.csv",
                encoding="utf-8-sig",
            )
            assignments[
                "last_effective_topology_step_id_optional"
            ] = "BAD_STEP"
            assignments.to_csv(
                target / "I_pred" / "future_sms_assignment.csv",
                index=False,
                encoding="utf-8-sig",
            )
            (target / "validation" / "validate_package.py").unlink()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "cli_check.py"),
                    str(target),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("ROLLING_FINAL_STATUS=FAIL", completed.stdout)
        self.assertIn("FINAL_STATUS=FAIL", completed.stdout)

    def test_audit_05_application_matrix_missing_duplicate_unexpected(self) -> None:
        production = rolling_module._rolling_contexts

        def run_fault(kind: str):
            def faulty(*args, **kwargs):
                contexts = production(*args, **kwargs)
                first = next(iter(contexts.values()))
                if kind == "missing":
                    first["sms_application_trace"] = []
                elif kind == "duplicate":
                    first["sms_application_trace"][0][
                        "application_count"
                    ] = 2
                else:
                    extra = copy.deepcopy(
                        first["sms_application_trace"][0]
                    )
                    extra["topology_step_id"] = "PRE_ACTIVATION_STEP"
                    first["sms_application_trace"].append(extra)
                return contexts

            with patch.object(
                rolling_module, "_rolling_contexts", side_effect=faulty
            ):
                return run_posterior_virtual_sms_rolling_prediction(
                    self.package,
                    self.topology,
                    PLAN_ID,
                    ["VSMS_M1_POS"],
                )

        for kind in ("missing", "duplicate", "unexpected"):
            with self.subTest(kind=kind):
                result = run_fault(kind)
                self.assertEqual(result.quality_status, "FAIL")
                self.assertEqual(
                    result.status_summary["formal_sample_count"], 1
                )
                self.assertEqual(
                    result.status_summary["formal_failure_count"], 1
                )
                self.assertGreater(
                    result.status_summary["application_fail_count"], 0
                )
                self.assertEqual(
                    result.sample_failures[0].source_state_role,
                    "POSTERIOR",
                )

    def test_audit_06_normal_application_matrix_has_twenty_rows(self) -> None:
        traces = [
            row
            for item in self.result.sample_results
            for row in item.trace["sms_application_trace"]
        ]
        self.assertEqual(len(traces), 20)
        self.assertTrue(all(
            row["application_count"] == 1 for row in traces
        ))
        self.assertTrue(all(
            item.trace["application_completeness"]["status"] == "PASS"
            and item.trace["application_completeness"][
                "expected_count"
            ] == 4
            for item in self.result.sample_results
        ))

    def test_audit_07_mapping_role_and_matrix_norm_are_governed(self) -> None:
        mappings = self.package.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ]
        matrix_id = str(mappings.iloc[0]["matrix_id"])
        cases = []
        effective_zero = copy.deepcopy(self.package)
        effective_zero.matrices[matrix_id] = np.zeros_like(
            effective_zero.matrices[matrix_id]
        )
        cases.append(("effective zero", effective_zero))
        explicit_nonzero = copy.deepcopy(self.package)
        explicit_nonzero.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ].loc[0, "mapping_role"] = "EXPLICIT_ZERO_NO_EFFECT"
        explicit_nonzero.raw_tables[
            "matrices/matrix_manifest.csv"
        ].loc[
            explicit_nonzero.raw_tables[
                "matrices/matrix_manifest.csv"
            ]["matrix_id"].astype(str).eq(matrix_id),
            "mapping_role_optional",
        ] = "EXPLICIT_ZERO_NO_EFFECT"
        cases.append(("explicit nonzero", explicit_nonzero))
        unknown_role = copy.deepcopy(self.package)
        unknown_role.raw_tables[
            "I_pred/sms_operator_mapping.csv"
        ].loc[0, "mapping_role"] = "UNKNOWN_ROLE"
        cases.append(("unknown role", unknown_role))
        for name, package in cases:
            with self.subTest(name=name):
                failures = self._blocking_failures(package)
                self.assertTrue(
                    failures["check_item"].astype(str).str.contains(
                        "G_SMS"
                    ).any()
                )

    def test_audit_08_policy_quality_reference_and_unit_faults(self) -> None:
        cases = (
            (
                "invalid KCP set",
                "I_pred/rolling_prediction_plan.csv",
                "kcp_set_id",
                "NO_SUCH_SET",
                "KCP set",
            ),
            (
                "invalid aggregation",
                "I_pred/rolling_prediction_plan.csv",
                "aggregation_policy",
                "UNSUPPORTED_POLICY",
                "aggregation policy",
            ),
            (
                "invalid baseline",
                "I_pred/rolling_prediction_plan.csv",
                "baseline_comparison_policy",
                "UNKNOWN_BASELINE",
                "baseline policy",
            ),
            (
                "invalid failure",
                "I_pred/rolling_prediction_plan.csv",
                "failure_policy",
                "UNKNOWN",
                "failure policy",
            ),
            (
                "plan quality",
                "I_pred/rolling_prediction_plan.csv",
                "quality_flag",
                "FAIL",
                "plan quality",
            ),
            (
                "sample quality",
                "I_pred/virtual_sms_sample.csv",
                "quality_flag",
                "FAIL",
                "sample row quality",
            ),
            (
                "component quality",
                "I_pred/virtual_sms_component.csv",
                "quality_flag",
                "FAIL",
                "component quality",
            ),
            (
                "coefficient reference",
                "I_pred/virtual_sms_coefficients.csv",
                "reference_sms_id",
                "WRONG_REF",
                "coefficient unit/reference/quality",
            ),
            (
                "coefficient unit",
                "I_pred/virtual_sms_coefficients.csv",
                "unit",
                "mm",
                "coefficient unit/reference/quality",
            ),
            (
                "assignment quality",
                "I_pred/future_sms_assignment.csv",
                "quality_flag",
                "FAIL",
                "assignment",
            ),
            (
                "mapping reference",
                "I_pred/sms_operator_mapping.csv",
                "reference_sms_id",
                "WRONG_REF",
                "G_SMS",
            ),
            (
                "scenario quality",
                "I_pred/future_process_scenario.csv",
                "quality_flag",
                "FAIL",
                "scenario quality",
            ),
            (
                "KCP quality",
                "I_pred/rolling_kcp_config.csv",
                "quality_flag",
                "FAIL",
                "KCP config",
            ),
        )
        for name, table_name, column, value, expected_gate in cases:
            with self.subTest(name=name):
                package = copy.deepcopy(self.package)
                package.raw_tables[table_name][column] = (
                    package.raw_tables[table_name][column].astype(object)
                )
                package.raw_tables[table_name].loc[0, column] = value
                gates = validate_rolling_prediction_package(
                    package, self.topology
                )
                failures = gates[
                    gates["status"].astype(str).eq("FAIL")
                    & gates["blocking"].astype(bool)
                ]
                self.assertTrue(
                    failures["check_item"].astype(str).str.contains(
                        expected_gate, regex=False
                    ).any(),
                    failures.to_string(index=False),
                )

    def test_audit_09_predicted_only_failure_has_separate_counts(self) -> None:
        production = rolling_module._run_sample_branch

        def predicted_failure(*args, **kwargs):
            if args[5] == "VSMS_M1_POS" and args[7] == "PREDICTED":
                raise RollingPredictionRuntimeError(
                    "injected predicted-only failure"
                )
            return production(*args, **kwargs)

        with patch.object(
            rolling_module,
            "_run_sample_branch",
            side_effect=predicted_failure,
        ):
            result = run_posterior_virtual_sms_rolling_prediction(
                self.package, self.topology, PLAN_ID
            )
        summary = result.descriptive_summary
        self.assertEqual(summary.sample_count, 5)
        self.assertEqual(summary.success_count, 5)
        self.assertEqual(summary.failure_count, 0)
        self.assertEqual(summary.baseline_attempt_count, 5)
        self.assertEqual(summary.baseline_success_count, 4)
        self.assertEqual(summary.baseline_failure_count, 1)
        self.assertEqual(result.quality_status, "FAIL")
        self.assertEqual(
            result.predicted_baseline_failures[0].source_state_role,
            "PREDICTED",
        )
        failed_rows = result.baseline_comparison[
            result.baseline_comparison[
                "virtual_sms_sample_id"
            ].eq("VSMS_M1_POS")
        ]
        self.assertTrue(
            failed_rows["comparison_status"].eq(
                "PREDICTED_FAILED"
            ).all()
        )
        self.assertTrue(
            failed_rows["posterior_minus_predicted"].isna().all()
        )

    def test_audit_10_posterior_only_policy_skips_baseline(self) -> None:
        package = copy.deepcopy(self.package)
        package.raw_tables[
            "I_pred/rolling_prediction_plan.csv"
        ].loc[0, "baseline_comparison_policy"] = "POSTERIOR_ONLY"
        gates = validate_rolling_prediction_package(
            package, self.topology
        )
        self.assertFalse(
            (
                gates["status"].eq("FAIL")
                & gates["blocking"].astype(bool)
            ).any(),
            gates.to_string(index=False),
        )
        result = run_posterior_virtual_sms_rolling_prediction(
            package, self.topology, PLAN_ID
        )
        self.assertEqual(result.quality_status, "PASS")
        self.assertEqual(
            result.status_summary["baseline_attempt_count"], 0
        )
        self.assertEqual(
            result.status_summary["baseline_quality_status"],
            "NOT_APPLICABLE",
        )
        self.assertFalse(result.predicted_baseline_results)
        self.assertFalse(result.predicted_baseline_failures)
        self.assertEqual(
            result.status_summary["baseline_success_count"], 0
        )
        self.assertEqual(
            result.status_summary["baseline_failure_count"], 0
        )
        self.assertTrue(result.baseline_comparison.empty)
        self.assertGreater(len(result.baseline_comparison.columns), 0)
        self.assertNotIn(
            "PREDICTED_FAILED",
            set(result.baseline_comparison.get(
                "comparison_status", pd.Series(dtype=str)
            ).astype(str)),
        )
        tables, _ = rolling_prediction_report_tables(package, result)
        self.assertTrue(
            tables["rolling_baseline_comparison.csv"].empty
        )
        self.assertGreater(
            len(tables["rolling_baseline_comparison.csv"].columns), 0
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / PACKAGE.name
            shutil.copytree(PACKAGE, target)
            plan_path = (
                target / "I_pred" / "rolling_prediction_plan.csv"
            )
            plan_table = pd.read_csv(
                plan_path, encoding="utf-8-sig"
            )
            plan_table.loc[
                0, "baseline_comparison_policy"
            ] = "POSTERIOR_ONLY"
            plan_table.to_csv(
                plan_path, index=False, encoding="utf-8-sig"
            )
            (target / "validation" / "validate_package.py").unlink()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "cli_check.py"),
                    str(target),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "ROLLING_BASELINE_QUALITY_STATUS=NOT_APPLICABLE",
            completed.stdout,
        )
        self.assertIn(
            "ROLLING_BASELINE_ATTEMPT_COUNT=0",
            completed.stdout,
        )
        self.assertNotIn("PREDICTED_FAILED", completed.stdout)

    def test_audit_11_complete_topology_hash_detects_start_mutation(self) -> None:
        topology = copy.deepcopy(self.topology)
        production = rolling_module._run_sample_branch

        def mutate_start(*args, **kwargs):
            topology_result = args[1]
            plan = args[3]
            topology_result[plan.prediction_start_step_id][
                "audit_injected_mutation"
            ] = True
            return production(*args, **kwargs)

        with patch.object(
            rolling_module, "_run_sample_branch", side_effect=mutate_start
        ):
            result = run_posterior_virtual_sms_rolling_prediction(
                self.package, topology, PLAN_ID, ["VSMS_REF"]
            )
        self.assertEqual(result.quality_status, "FAIL")
        self.assertEqual(result.trace["immutability_status"], "FAIL")
        self.assertIn(
            "TS301", result.trace["immutability_changed_objects"]
        )
        self.assertEqual(
            result.status_summary["immutability_fail_count"], 1
        )
        self.assertEqual(
            _runtime_exit_code(
                0,
                "PASS",
                "PASS",
                rolling_immutability_fail_count=1,
            ),
            ("FAIL", 2),
        )

    def test_audit_12_cli_counts_unique_samples_for_multiple_parts(self) -> None:
        table = pd.DataFrame([
            {
                "sample_set_id": "SET",
                "virtual_sms_sample_id": f"S{sample}",
                "part_id": part,
            }
            for sample in range(5)
            for part in ("PART_LEFT", "PART_RIGHT")
        ])
        count, complete = _unique_plan_sample_count(
            table, "SET", ("PART_LEFT", "PART_RIGHT")
        )
        self.assertEqual(count, 5)
        self.assertTrue(complete)
        count, complete = _unique_plan_sample_count(
            table.iloc[:-1], "SET", ("PART_LEFT", "PART_RIGHT")
        )
        self.assertEqual(count, 5)
        self.assertFalse(complete)

    def test_audit_13_package_file_hash_uses_raw_bytes(self) -> None:
        current = stable_package_file_hash(PACKAGE)
        self.assertEqual(current, stable_package_file_hash(PACKAGE / "."))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.csv").write_bytes(b"x,y\\n1,2\\n")
            (root / "nested").mkdir()
            target = root / "nested" / "b.bin"
            target.write_bytes(b"\x00\x01\x02")
            before = stable_package_file_hash(root)
            self.assertEqual(before, stable_package_file_hash(root / "."))
            target.write_bytes(b"\x00\x01\x03")
            after = stable_package_file_hash(root)
        self.assertNotEqual(before, after)

    def test_final_01_direct_sms_boolean_changes_actual_kcp_path(self) -> None:
        sample_id = "VSMS_M1_POS"
        add_result = run_posterior_virtual_sms_rolling_prediction(
            self.package, self.topology, PLAN_ID, [sample_id]
        )
        included_package = copy.deepcopy(self.package)
        included_package.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ]["final_state_includes_direct_sms_geometry"] = True
        gates = validate_rolling_prediction_package(
            included_package, self.topology
        )
        self.assertFalse(
            (
                gates["status"].eq("FAIL")
                & gates["blocking"].astype(bool)
            ).any(),
            gates.to_string(index=False),
        )
        included_result = run_posterior_virtual_sms_rolling_prediction(
            included_package, self.topology, PLAN_ID, [sample_id]
        )
        add = add_result.sample_results[0]
        included = included_result.sample_results[0]
        add_kcp = add.kcp_prediction_result.set_index("kcp_id")
        included_kcp = included.kcp_prediction_result.set_index("kcp_id")
        np.testing.assert_allclose(
            add_kcp["predicted_value"].to_numpy(float)
            - included_kcp["predicted_value"].to_numpy(float),
            add_kcp[
                "virtual_sms_direct_contribution"
            ].to_numpy(float),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            included_kcp[
                "virtual_sms_direct_contribution"
            ].to_numpy(float),
            0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            add_kcp[
                "virtual_sms_direct_contribution_candidate"
            ].to_numpy(float),
            included_kcp[
                "virtual_sms_direct_contribution_candidate"
            ].to_numpy(float),
            atol=1e-12,
        )
        self.assertIn(
            "FUTURE_SMS_DIRECT_GEOMETRY",
            set(add.contribution_ledger["source_class"].astype(str)),
        )
        self.assertNotIn(
            "FUTURE_SMS_DIRECT_GEOMETRY",
            set(included.contribution_ledger["source_class"].astype(str)),
        )
        self.assertFalse(
            add_result.status_summary[
                "final_state_includes_direct_sms_geometry"
            ]
        )
        self.assertTrue(
            included_result.status_summary[
                "final_state_includes_direct_sms_geometry"
            ]
        )
        self.assertEqual(
            included_result.status_summary[
                "direct_sms_aggregation_action"
            ],
            "DIRECT_SMS_ALREADY_INCLUDED_IN_FINAL_STATE",
        )

    def test_final_02_direct_sms_invalid_and_conflicting_config_blocks(
        self,
    ) -> None:
        invalid = copy.deepcopy(self.package)
        invalid.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ]["final_state_includes_direct_sms_geometry"] = (
            invalid.raw_tables[
                "I_pred/rolling_kcp_config.csv"
            ]["final_state_includes_direct_sms_geometry"].astype(object)
        )
        invalid.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ].loc[0, "final_state_includes_direct_sms_geometry"] = "yes"
        failures = self._blocking_failures(invalid)
        self.assertTrue(
            failures["check_item"].astype(str).str.contains(
                "direct geometry semantics", regex=False
            ).any(),
            failures.to_string(index=False),
        )

        conflict = copy.deepcopy(self.package)
        configs = conflict.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ].copy()
        conflicting = configs.iloc[[0]].copy()
        conflicting[
            "final_state_includes_direct_sms_geometry"
        ] = True
        conflict.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ] = pd.concat([configs, conflicting], ignore_index=True)
        failures = self._blocking_failures(conflict)
        self.assertTrue(
            failures["check_item"].astype(str).str.contains(
                "direct geometry semantics", regex=False
            ).any(),
            failures.to_string(index=False),
        )

        bad_aggregation = copy.deepcopy(self.package)
        bad_aggregation.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ].loc[0, "aggregation_policy"] = "UNSUPPORTED"
        failures = self._blocking_failures(bad_aggregation)
        self.assertTrue(
            failures["check_item"].astype(str).str.contains(
                "aggregation compatibility", regex=False
            ).any(),
            failures.to_string(index=False),
        )

    def test_final_03_source_checkpoint_and_runtime_linkage_faults_block(
        self,
    ) -> None:
        static_faults = {
            "checkpoint step": (
                "I_meas/measurement_checkpoint.csv",
                "topology_step_id",
                "TS204",
            ),
            "checkpoint topology": (
                "I_meas/measurement_checkpoint.csv",
                "topology_id",
                "WRONG_TOPOLOGY",
            ),
            "plan posterior id": (
                "I_pred/rolling_prediction_plan.csv",
                "source_posterior_state_id_optional",
                "WRONG_POSTERIOR",
            ),
        }
        for name, (table_name, column, value) in static_faults.items():
            with self.subTest(name=name):
                package = copy.deepcopy(self.package)
                package.raw_tables[table_name][column] = (
                    package.raw_tables[table_name][column].astype(object)
                )
                package.raw_tables[table_name].loc[0, column] = value
                gates = validate_rolling_prediction_package(
                    package, self.topology
                )
                failures = gates[
                    gates["status"].astype(str).eq("FAIL")
                    & gates["blocking"].astype(bool)
                ]
                self.assertTrue(
                    failures["check_item"].astype(str).str.contains(
                        "source checkpoint/topology linkage",
                        regex=False,
                    ).any(),
                    failures.to_string(index=False),
                )

        duplicate_checkpoint = copy.deepcopy(self.package)
        checkpoints = duplicate_checkpoint.raw_tables[
            "I_meas/measurement_checkpoint.csv"
        ].copy()
        duplicate_checkpoint.raw_tables[
            "I_meas/measurement_checkpoint.csv"
        ] = pd.concat(
            [checkpoints, checkpoints.iloc[[0]]], ignore_index=True
        )
        failures = self._blocking_failures(duplicate_checkpoint)
        self.assertTrue(
            failures["check_item"].astype(str).str.contains(
                "source checkpoint", regex=False
            ).any(),
            failures.to_string(index=False),
        )

        mutators = {
            "update checkpoint": lambda source: setattr(
                source["measurement_update"],
                "checkpoint_id",
                "WRONG_CHECKPOINT",
            ),
            "update rejected": lambda source: setattr(
                source["measurement_update"],
                "posterior_accepted",
                False,
            ),
            "posterior id": lambda source: setattr(
                source["measurement_update"],
                "posterior_state_id",
                "WRONG_POSTERIOR",
            ),
            "actual role": lambda source: setattr(
                source["stage_state"], "state_role", "PREDICTED"
            ),
            "actual checkpoint": lambda source: setattr(
                source["stage_state"],
                "source_checkpoint_id",
                "WRONG_CHECKPOINT",
            ),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                topology = copy.deepcopy(self.topology)
                mutate(topology["TS205"])
                gates = validate_rolling_prediction_package(
                    self.package, topology
                )
                linkage = gates[
                    gates["check_item"].astype(str).str.contains(
                        "source checkpoint/topology linkage",
                        regex=False,
                    )
                ].iloc[0]
                self.assertEqual(linkage["status"], "FAIL")
                self.assertTrue(bool(linkage["blocking"]))
                with self.assertRaises(
                    RollingPredictionValidationError
                ):
                    run_posterior_virtual_sms_rolling_prediction(
                        self.package, topology, PLAN_ID, ["VSMS_REF"]
                    )
        self.assertEqual(
            self.result.trace["source_linkage_status"], "PASS"
        )
        self.assertEqual(
            self.result.trace["actual_source_state_id"],
            self.result.source_posterior_state.stage_state_id,
        )

    def test_final_04_checkpoint_step_fault_cli_exit_one(self) -> None:
        faults = {
            "checkpoint step": (
                "I_meas/measurement_checkpoint.csv",
                "topology_step_id",
                "TS204",
            ),
            "invalid direct boolean": (
                "I_pred/rolling_kcp_config.csv",
                "final_state_includes_direct_sms_geometry",
                "INVALID",
            ),
        }
        for name, (relative, column, value) in faults.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp:
                    target = Path(temp) / PACKAGE.name
                    shutil.copytree(PACKAGE, target)
                    table_path = target / Path(relative)
                    table = pd.read_csv(
                        table_path, encoding="utf-8-sig"
                    )
                    table[column] = table[column].astype(object)
                    table.loc[0, column] = value
                    table.to_csv(
                        table_path,
                        index=False,
                        encoding="utf-8-sig",
                    )
                    (
                        target
                        / "validation"
                        / "validate_package.py"
                    ).unlink()
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "cli_check.py"),
                            str(target),
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                self.assertEqual(
                    completed.returncode, 1, completed.stderr
                )
                self.assertIn("FINAL_STATUS=FAIL", completed.stdout)
                self.assertIn(
                    "ROLLING_FINAL_STATUS=FAIL", completed.stdout
                )

    def test_final_05_streamlit_real_all_failure_and_posterior_only(
        self,
    ) -> None:
        from streamlit.testing.v1 import AppTest

        def render(package):
            with patch(
                "core.data_loader.load_package",
                return_value=package,
            ):
                app = AppTest.from_file(
                    str(ROOT / "app.py"), default_timeout=90
                ).run()
                navigation = next(
                    radio for radio in app.radio
                    if radio.key == "main_page_navigation"
                )
                navigation.set_value(
                    navigation.options[-1]
                ).run(timeout=90)
            return app

        duplicate = copy.deepcopy(self.package)
        configs = duplicate.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ].copy()
        duplicate.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ] = pd.concat([configs, configs.iloc[[0]]], ignore_index=True)
        failed_app = render(duplicate)
        self.assertFalse(failed_app.exception, failed_app.exception)
        self.assertIn(
            "0/5",
            {str(item.value) for item in failed_app.metric},
        )
        failure_rendered = "\n".join(
            str(item.value)
            for collection in (
                failed_app.error,
                failed_app.info,
                failed_app.markdown,
            )
            for item in collection
        )
        self.assertIn("FAIL", failure_rendered)
        self.assertIn("没有可信的 KCP", failure_rendered)
        self.assertIn("没有贡献账本记录", failure_rendered)
        self.assertIn("不生成描述性 KCP 统计", failure_rendered)
        self.assertTrue(any(
            isinstance(item.value, pd.DataFrame)
            and "failure_reason" in item.value.columns
            and len(item.value) >= 5
            for item in failed_app.dataframe
        ))

        posterior_only = copy.deepcopy(self.package)
        posterior_only.raw_tables[
            "I_pred/rolling_prediction_plan.csv"
        ].loc[0, "baseline_comparison_policy"] = "POSTERIOR_ONLY"
        posterior_app = render(posterior_only)
        self.assertFalse(posterior_app.exception, posterior_app.exception)
        posterior_rendered = "\n".join(
            str(item.value)
            for collection in (
                posterior_app.info,
                posterior_app.error,
                posterior_app.markdown,
            )
            for item in collection
        )
        self.assertIn("NOT_APPLICABLE", posterior_rendered)
        self.assertNotIn("PREDICTED_FAILED", posterior_rendered)

        included = copy.deepcopy(self.package)
        included.raw_tables[
            "I_pred/rolling_kcp_config.csv"
        ]["final_state_includes_direct_sms_geometry"] = True
        included_app = render(included)
        self.assertFalse(included_app.exception, included_app.exception)
        included_rendered = "\n".join(
            str(item.value)
            for collection in (
                included_app.caption,
                included_app.info,
                included_app.error,
            )
            for item in collection
        )
        self.assertIn(
            "final_state_includes_direct_sms_geometry=True",
            included_rendered,
        )
        self.assertIn(
            "DIRECT_SMS_ALREADY_INCLUDED_IN_FINAL_STATE",
            included_rendered,
        )

    def test_final_06_coefficient_delta_has_single_assignment(self) -> None:
        source = inspect.getsource(rolling_module._coefficient_vectors)
        self.assertEqual(
            source.count("deltas: dict[str, np.ndarray] = {}"), 1
        )


if __name__ == "__main__":
    unittest.main()
