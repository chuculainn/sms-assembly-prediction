from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.data_loader import load_package
from core.fallback import evaluate_validity_and_fallback
from core.physical_consistency import stage_physical_consistency
from core.stage_solver import run_all_stages
from core.topology_step import (
    TopologyStepValidationError,
    run_topology_steps,
    topology_step_execution_table,
    validate_topology_steps,
)
import core.topology_step as topology_step_module
from scripts.cli_check import _blocking_fail_count, _runtime_exit_code
from tests.topology_fixture_factory import make_two_part_two_point_fixture


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data" / "02_TOPOLOGY_STEP_MIN_CASE"
PYTHON = Path(sys.executable)
TOPOLOGY_FIELDS = {
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
}


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class PackageSelfValidationTests(unittest.TestCase):
    def test_package_local_validator_returns_zero_and_all_checks_pass(self) -> None:
        completed = subprocess.run(
            [str(PYTHON), str(PACKAGE / "validation" / "validate_package.py"), str(PACKAGE)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["blocking_fail_count"], 0)
        self.assertTrue(all(item["status"] == "PASS" for item in report["checks"]))

    def test_validator_source_has_no_old_fixture_ids(self) -> None:
        source = (PACKAGE / "validation" / "validate_package.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "P_PANEL_A",
            "P_RIB_B",
            "P_SPAR_C",
            "P_BRACKET_D",
            "G_PANEL_RIB",
            "CD_PANEL_RIB",
        ):
            self.assertNotIn(forbidden, source)

    def test_field_dictionary_covers_actual_topology_step_schema(self) -> None:
        dictionary = pd.read_csv(PACKAGE / "field_dictionary.csv", encoding="utf-8-sig")
        topology = dictionary[
            dictionary["file_path"].eq("I0/assembly_topology.csv")
        ]
        self.assertTrue(TOPOLOGY_FIELDS <= set(topology["field_name"]))
        self.assertTrue(
            {
                "data_type",
                "required",
                "cardinality",
                "unit",
                "enum_or_format",
                "key_semantics",
                "missing_handling",
                "description",
            }
            <= set(dictionary.columns)
        )

    def test_field_dictionary_covers_every_csv_header(self) -> None:
        dictionary = pd.read_csv(
            PACKAGE / "field_dictionary.csv", encoding="utf-8-sig", dtype=str
        ).fillna("")
        recorded = set(zip(dictionary["file_path"], dictionary["field_name"]))
        missing = []
        for path in PACKAGE.rglob("*.csv"):
            relative = path.relative_to(PACKAGE).as_posix()
            for field in pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns:
                if (relative, field) not in recorded:
                    missing.append((relative, field))
        self.assertEqual(missing, [])

    def test_object_file_map_registers_oracle_and_runtime_objects(self) -> None:
        mapping = pd.read_csv(PACKAGE / "object_file_map.csv", encoding="utf-8-sig")
        names = set(mapping["object_name"])
        self.assertTrue(
            {
                "TopologyStepSpec",
                "TopologyStepResult",
                "ConnectionLockHistory",
                "ReleaseHistoryRecord",
                "TopologyStepLcpOracle",
                "TopologyStepOperatorMatrices",
                "TopologyStepExecutionReport",
                "ValidationResults",
            }
            <= names
        )
        runtime_paths = set(
            mapping.loc[
                mapping["is_runtime_result"].astype(str).str.lower().eq("true"),
                "file_path",
            ]
        )
        self.assertTrue(
            {
                "runtime/topology_step_execution.csv",
                "runtime/topology_step_validation.csv",
                "runtime/active_subassembly_history.csv",
                "runtime/topology_step_state_lineage.csv",
                "runtime/topology_step_operator_usage.csv",
                "runtime/topology_step_contact_summary.csv",
                "runtime/connection_lock_history.csv",
                "runtime/release_history.csv",
            }
            <= runtime_paths
        )

    def test_matrix_manifest_exactly_matches_npz_keys(self) -> None:
        matrix_manifest = pd.read_csv(
            PACKAGE / "matrices" / "matrix_manifest.csv", encoding="utf-8-sig"
        )
        with np.load(
            PACKAGE / "matrices" / "multi_part_matrices.npz", allow_pickle=False
        ) as archive:
            keys = set(archive.files)
        self.assertEqual(len(matrix_manifest), len(keys))
        self.assertEqual(set(matrix_manifest["npz_key"]), keys)
        self.assertEqual(len(keys), 156)

    def test_matrix_manifest_shape_dtype_and_layout_match(self) -> None:
        matrix_manifest = pd.read_csv(
            PACKAGE / "matrices" / "matrix_manifest.csv",
            encoding="utf-8-sig",
            dtype=str,
        ).fillna("")
        with np.load(
            PACKAGE / "matrices" / "multi_part_matrices.npz", allow_pickle=False
        ) as archive:
            for row in matrix_manifest.to_dict("records"):
                array = archive[row["npz_key"]]
                self.assertEqual(tuple(json.loads(row["shape"])), array.shape)
                self.assertEqual(row["dtype"], str(array.dtype))
                if array.shape[0] == 12:
                    self.assertTrue(row["row_layout_id_optional"])
                if array.ndim == 2 and array.shape[1] == 12:
                    self.assertTrue(row["column_layout_id_optional"])

    def test_test_results_match_real_validator_statistics(self) -> None:
        stored = json.loads(
            (PACKAGE / "validation" / "test_results.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stored["status"], "PASS")
        self.assertEqual(stored["passed"], stored["total"])
        self.assertEqual(stored["blocking_fail_count"], 0)
        self.assertEqual(stored["matrix_manifest_count"], 156)
        self.assertEqual(stored["npz_key_count"], 156)

    def test_markdown_matches_json_summary(self) -> None:
        stored = json.loads(
            (PACKAGE / "validation" / "test_results.json").read_text(encoding="utf-8")
        )
        markdown = (PACKAGE / "validation" / "TEST_RESULTS.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(f"Status: **{stored['status']}**", markdown)
        self.assertIn(f"Checks: {stored['passed']}/{stored['total']} PASS", markdown)
        self.assertEqual(markdown.count("| PASS | true |"), stored["passed"])

    def test_run_log_points_to_current_package_and_generator(self) -> None:
        row = pd.read_csv(
            PACKAGE / "validation" / "run_log.csv", encoding="utf-8-sig"
        ).iloc[0]
        self.assertEqual(row["input_package_id"], PACKAGE.name)
        self.assertEqual(
            row["operator_or_script"], "scripts/build_topology_step_fixture.py"
        )
        self.assertEqual(int(row["matrix_manifest_count"]), 156)
        self.assertEqual(int(row["npz_key_count"]), 156)

    def test_quality_gate_targets_current_package(self) -> None:
        row = pd.read_csv(
            PACKAGE / "validation" / "quality_gate.csv", encoding="utf-8-sig"
        ).iloc[0]
        self.assertEqual(row["target_object_ids"], PACKAGE.name)
        self.assertEqual(row["pass_fail"], "PASS")
        self.assertIn("MATRIX_MANIFEST", row["check_items"])

    def test_validation_result_is_topology_step_oracle(self) -> None:
        row = pd.read_csv(
            PACKAGE / "validation" / "validation_result.csv", encoding="utf-8-sig"
        ).iloc[0]
        oracle = pd.read_csv(
            PACKAGE / "validation" / "topology_step_lcp_oracle.csv",
            encoding="utf-8-sig",
        )
        self.assertEqual(row["reference_type"], "SYNTHETIC_INDEPENDENT_LCP_ORACLE")
        self.assertEqual(
            set(str(row["validation_sample_ids"]).split(";")),
            set(oracle["topology_step_id"]),
        )
        self.assertEqual(row["pass_fail"], "PASS")

    def test_generator_rebuild_is_reproducible(self) -> None:
        first = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "build_topology_step_fixture.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        first_digest = _tree_digest(PACKAGE)
        second = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "build_topology_step_fixture.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first_digest, _tree_digest(PACKAGE))

    def test_blocking_package_validation_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            broken = Path(temporary) / PACKAGE.name
            shutil.copytree(PACKAGE, broken)
            interfaces = pd.read_csv(
                broken / "I0" / "interface.csv", encoding="utf-8-sig"
            )
            interfaces.loc[0, "part_i"] = "MISSING_PART"
            interfaces.to_csv(
                broken / "I0" / "interface.csv",
                index=False,
                encoding="utf-8-sig",
            )
            completed = subprocess.run(
                [
                    str(PYTHON),
                    str(broken / "validation" / "validate_package.py"),
                    str(broken),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
        self.assertNotEqual(completed.returncode, 0)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["blocking_fail_count"], 0)


class MechanicalStateAndRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = make_two_part_two_point_fixture()
        cls.result = run_topology_steps(cls.package)

    def test_second_fixture_is_two_part_one_interface_two_dimensional(self) -> None:
        self.assertEqual(len(self.package.parts), 2)
        self.assertEqual(len(self.package.interfaces), 1)
        self.assertEqual(len(self.package.contact_points), 2)
        self.assertTrue(
            np.asarray(self.result["EV_LOCATE"]["lambda_full"]).shape == (2,)
        )

    def test_init_not_required_has_empty_state_semantics(self) -> None:
        init = self.result["EV_INIT"]
        state = init["stage_state"]
        self.assertEqual(init["solve_status"], "NOT_REQUIRED")
        self.assertEqual(state.mechanical_state_action, "INITIALIZE_EMPTY")
        self.assertEqual(state.not_required_reason, "NO_PARENT_MECHANICAL_STATE")
        self.assertIsNone(state.parent_stage_state_id)
        np.testing.assert_array_equal(init["lambda_full"], np.zeros(2))
        self.assertTrue(np.isnan(init["gap_full"]).all())

    def test_middle_measure_inherits_all_mechanical_vectors(self) -> None:
        parent = self.result["EV_LOCATE"]
        measure = self.result["EV_MEASURE"]
        for name in (
            "lambda_full",
            "gap_full",
            "pressure",
            "local_compression",
            "active_index_mask",
        ):
            np.testing.assert_allclose(measure[name], parent[name], equal_nan=True)
        np.testing.assert_allclose(
            measure["stage_state"].contact_structural_response,
            parent["stage_state"].contact_structural_response,
        )
        self.assertEqual(
            measure["stage_state"].active_interface_ids,
            parent["stage_state"].active_interface_ids,
        )

    def test_middle_measure_creates_child_state_without_lcp(self) -> None:
        original = topology_step_module.solve_lcp_active_set
        with patch("core.topology_step.solve_lcp_active_set", wraps=original) as mocked:
            result = run_topology_steps(make_two_part_two_point_fixture())
        self.assertEqual(mocked.call_count, 3)
        measure = result["EV_MEASURE"]
        self.assertEqual(measure["lcp_call_count"], 0)
        self.assertEqual(
            measure["stage_state"].mechanical_state_action,
            "INHERIT_PARENT_UNCHANGED",
        )
        self.assertEqual(
            measure["stage_state"].parent_stage_state_id,
            result["EV_LOCATE"]["stage_state"].stage_state_id,
        )
        self.assertNotEqual(
            measure["stage_state"].stage_state_id,
            result["EV_LOCATE"]["stage_state"].stage_state_id,
        )

    def test_non_solving_mechanical_mutation_is_blocked(self) -> None:
        package = make_two_part_two_point_fixture()
        route = package.raw_tables["I0/assembly_topology.csv"].copy()
        route.loc[
            route["topology_step_id"].eq("EV_MEASURE"),
            "deactivated_interface_ids",
        ] = "IF_ALPHA_BETA"
        package.raw_tables["I0/assembly_topology.csv"] = route
        report = validate_topology_steps(package)
        row = report[report["check_item"].eq("非求解步骤不改变机械活动集合")].iloc[0]
        self.assertEqual(row["status"], "FAIL")
        with self.assertRaises(TopologyStepValidationError):
            run_topology_steps(package)

    def test_release_retains_keep_joint_and_removes_drop_joint(self) -> None:
        release = self.result["EV_RELEASE"]
        self.assertEqual(release["stage_state"].active_joint_ids, ["J_KEEP"])
        history = release["release_history"][0]
        self.assertEqual(history["retained_joint_ids"], "J_KEEP")
        self.assertEqual(history["removed_joint_ids"], "J_DROP")
        self.assertEqual(history["active_joint_ids_after_step"], "J_KEEP")

    def test_release_history_matches_actual_active_joint_set(self) -> None:
        release = self.result["EV_RELEASE"]
        history = release["release_history"][0]
        self.assertEqual(
            set(filter(None, history["active_joint_ids_after_step"].split(";"))),
            set(release["stage_state"].active_joint_ids),
        )
        self.assertEqual(
            release["stage_state"].joint_lock_state[
                "removed_joint_ids_at_release"
            ],
            ["J_DROP"],
        )

    def test_following_inspect_inherits_release_joint_set_and_response(self) -> None:
        release = self.result["EV_RELEASE"]
        inspect = self.result["EV_INSPECT"]
        self.assertEqual(inspect["stage_state"].active_joint_ids, ["J_KEEP"])
        for name in ("lambda_full", "gap_full", "pressure", "local_compression"):
            np.testing.assert_allclose(inspect[name], release[name], equal_nan=True)
        self.assertEqual(inspect["lcp_call_count"], 0)

    def test_explicit_deactivation_overrides_retain_rule(self) -> None:
        package = make_two_part_two_point_fixture()
        route = package.raw_tables["I0/assembly_topology.csv"].copy()
        route.loc[
            route["topology_step_id"].eq("EV_RELEASE"),
            "deactivated_joint_ids",
        ] = "J_KEEP"
        package.raw_tables["I0/assembly_topology.csv"] = route
        result = run_topology_steps(package)
        release = result["EV_RELEASE"]
        self.assertEqual(release["stage_state"].active_joint_ids, [])
        self.assertEqual(
            set(release["release_history"][0]["removed_joint_ids"].split(";")),
            {"J_KEEP", "J_DROP"},
        )

    def test_unknown_retention_rule_is_blocking(self) -> None:
        package = make_two_part_two_point_fixture()
        joints = package.raw_tables["I0/joint_definition.csv"].copy()
        joints.loc[joints["joint_id"].eq("J_DROP"), "retention_rule"] = "UNKNOWN_RULE"
        package.raw_tables["I0/joint_definition.csv"] = joints
        report = validate_topology_steps(package)
        row = report[
            report["check_item"].eq("JOIN和RELEASE retention_rule可解析")
        ].iloc[0]
        self.assertEqual(row["status"], "FAIL")
        with self.assertRaises(TopologyStepValidationError):
            run_topology_steps(package)

    def test_fallback_treats_not_required_as_valid_state(self) -> None:
        fallback = evaluate_validity_and_fallback(self.package, self.result)
        state_rows = fallback[fallback["check_item"].str.startswith("机械状态:")]
        self.assertEqual(set(state_rows["status"]), {"PASS"})
        self.assertTrue(
            state_rows["detail"].str.contains("mechanical_state_action=").all()
        )
        self.assertNotIn("FAIL", set(fallback["status"]))

    def test_physical_consistency_treats_not_required_as_valid_state(self) -> None:
        summary, detail = stage_physical_consistency(self.result)
        for step_id in ("EV_INIT", "EV_MEASURE", "EV_INSPECT"):
            self.assertEqual(
                summary.loc[summary["stage_id"].eq(step_id), "physics_status"].iloc[0],
                "PASS",
            )
            row = detail[
                detail["stage_id"].eq(step_id)
                & detail["check_item"].eq("mechanical_solve")
            ].iloc[0]
            self.assertEqual(row["status"], "PASS")
            self.assertIn("NOT_REQUIRED", row["detail"])


class CliAndParameterSemanticsTests(unittest.TestCase):
    def test_cli_blocking_failure_maps_to_exit_one(self) -> None:
        checks = pd.DataFrame(
            [{"status": "FAIL", "blocking": True}, {"status": "PASS", "blocking": False}]
        )
        self.assertEqual(_blocking_fail_count(checks), 1)

    def test_cli_solver_failure_maps_to_exit_two(self) -> None:
        self.assertEqual(_runtime_exit_code(1, "PASS", "PASS"), ("FAIL", 2))
        self.assertEqual(_runtime_exit_code(0, "FAIL", "PASS"), ("FAIL", 2))
        self.assertEqual(_runtime_exit_code(0, "PASS", "FAIL"), ("FAIL", 2))

    def test_cli_success_has_machine_readable_summary(self) -> None:
        completed = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "cli_check.py"), str(PACKAGE)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("FINAL_STATUS=PASS", completed.stdout)
        self.assertIn("BLOCKING_FAIL_COUNT=0", completed.stdout)
        self.assertIn("PHYSICAL_FAIL_COUNT=0", completed.stdout)

    def test_topology_runtime_multipliers_are_not_effective(self) -> None:
        package = load_package(PACKAGE)
        baseline = run_all_stages(package)
        changed = run_all_stages(
            package, sms_scale=2.2, closure_scale=0.35, cn_scale=3.5
        )
        for step_id in baseline:
            np.testing.assert_allclose(
                baseline[step_id]["lambda_full"],
                changed[step_id]["lambda_full"],
                equal_nan=True,
            )
            np.testing.assert_allclose(
                baseline[step_id]["gap_full"],
                changed[step_id]["gap_full"],
                equal_nan=True,
            )
            self.assertFalse(changed[step_id]["runtime_parameter_effective"])
            self.assertEqual(
                changed[step_id]["runtime_scales"],
                {"sms_scale": 1.0, "closure_scale": 1.0, "cn_scale": 1.0},
            )

    def test_topology_execution_report_marks_parameters_ineffective(self) -> None:
        result = run_all_stages(
            load_package(PACKAGE), sms_scale=2.0, closure_scale=2.0, cn_scale=2.0
        )
        table = topology_step_execution_table(result)
        self.assertTrue((table["parameter_effective"] == False).all())  # noqa: E712
        self.assertTrue(
            table["parameter_mode"]
            .eq("PRECOMPUTED_TOPOLOGY_STEP_OPERATOR_DISABLED")
            .all()
        )

    def test_legacy_runtime_multiplier_behavior_is_preserved(self) -> None:
        package = load_package(ROOT / "data" / "01_DEFAULT_MIN_CASE")
        baseline = run_all_stages(package)
        changed = run_all_stages(package, cn_scale=2.0)
        self.assertTrue(
            any(
                not np.allclose(
                    baseline[step_id]["W_total"], changed[step_id]["W_total"]
                )
                for step_id in baseline
            )
        )
        self.assertTrue(
            all(step["runtime_parameter_effective"] for step in changed.values())
        )
        self.assertTrue(
            all(step["runtime_scales"]["cn_scale"] == 2.0 for step in changed.values())
        )

    def test_app_disables_topology_multiplier_controls_but_not_legacy(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        legacy_slider = next(
            slider for slider in app.slider if slider.label == "SMS 形貌倍率"
        )
        self.assertFalse(legacy_slider.disabled)
        selector = next(
            box for box in app.selectbox if box.label == "标准输入包目录"
        )
        topology_option = next(
            option for option in selector.options if PACKAGE.name in option
        )
        selector.set_value(topology_option).run(timeout=60)
        for label in ("SMS 形貌倍率", "阶段闭合/载荷倍率", "局部界面柔度 Cn 倍率"):
            self.assertTrue(
                next(slider for slider in app.slider if slider.label == label).disabled
            )
        self.assertTrue(
            any(
                "预计算 topology_step 算子模式" in item.value
                for item in app.info
            )
        )

    def test_topology_mc_page_does_not_offer_fake_multiplier_run(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        selector = next(
            box for box in app.selectbox if box.label == "标准输入包目录"
        )
        topology_option = next(
            option for option in selector.options if PACKAGE.name in option
        )
        selector.set_value(topology_option).run(timeout=60)
        navigation = next(
            radio for radio in app.radio if radio.key == "main_page_navigation"
        )
        navigation.set_value("⑪ Monte Carlo与敏感性").run(timeout=60)
        self.assertFalse(
            any(button.label == "运行 Monte Carlo" for button in app.button)
        )
        self.assertTrue(
            any(
                "不对该模式执行" in item.value
                for item in app.info
            )
        )

    def test_general_executor_and_validator_have_no_fixture_ids(self) -> None:
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                ROOT / "core" / "topology_step.py",
                ROOT / "scripts" / "topology_step_package_validator.py",
            )
        )
        for forbidden in (
            "TS301",
            "G_AD",
            "G_DB",
            "P_A",
            "P_B",
            "range(13)",
            "== 12",
        ):
            self.assertNotIn(forbidden, corpus)


if __name__ == "__main__":
    unittest.main()
