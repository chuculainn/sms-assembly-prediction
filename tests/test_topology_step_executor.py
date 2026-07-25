from __future__ import annotations

import copy
from dataclasses import fields, replace
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from core.data_loader import load_package
from core.kcp import compare_validation, extract_kcp
from core.package_validator import data_truthfulness_statement, has_blocking_failures, validate_package_detailed
from core.reporting import build_runtime_report_zip
from core.stage_solver import run_all_stages, run_stage
from core.topology_step import (
    LEGACY_FALLBACK_REASON,
    LEGACY_OPERATOR,
    PRECOMPUTED_OPERATOR,
    TopologyStepSpec,
    TopologyStepValidationError,
    load_topology_steps,
    run_topology_steps,
    topology_step_execution_table,
    validate_topology_steps,
)
import core.topology_step as topology_step_module


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXTURE = DATA / "02_TOPOLOGY_STEP_MIN_CASE"


class TopologyStepExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pkg = load_package(FIXTURE)
        cls.specs = load_topology_steps(cls.pkg)
        cls.result = run_topology_steps(cls.pkg)

    @staticmethod
    def _check(report, text: str):
        rows = report[report["check_item"].astype(str).str.contains(text, regex=False)]
        if rows.empty:
            raise AssertionError(f"missing check: {text}")
        return rows.iloc[0]

    def test_topology_step_spec_has_contract_fields(self) -> None:
        actual = {field.name for field in fields(TopologyStepSpec)}
        expected = {
            "topology_id", "topology_step_id", "step_order", "parent_topology_step_id",
            "assembly_cycle_id", "operation_type", "stage_id", "input_subassembly_id",
            "result_subassembly_id", "added_part_ids", "removed_part_ids",
            "activated_interface_ids", "deactivated_interface_ids", "activated_boundary_ids",
            "deactivated_boundary_ids", "activated_load_ids", "removed_load_ids",
            "activated_joint_ids", "deactivated_joint_ids", "operator_set_id",
            "solve_required", "reference_state_id", "measurement_checkpoint_id", "notes",
        }
        self.assertTrue(expected <= actual)

    def test_topology_step_ids_are_unique(self) -> None:
        report = validate_topology_steps(self.pkg, self.specs)
        self.assertEqual(self._check(report, "topology_step_id唯一")["status"], "PASS")
        broken = self.specs + [replace(self.specs[-1], step_order=999)]
        self.assertEqual(self._check(validate_topology_steps(self.pkg, broken), "topology_step_id唯一")["status"], "FAIL")

    def test_step_order_is_stably_sorted_and_duplicate_is_blocked(self) -> None:
        self.assertEqual([s.topology_step_id for s in self.specs], [
            "TS000", "TS101", "TS102", "TS103", "TS104", "TS201", "TS202",
            "TS203", "TS204", "TS301", "TS302", "TS303", "TS304",
        ])
        broken = list(self.specs)
        broken[2] = replace(broken[2], step_order=broken[1].step_order)
        row = self._check(validate_topology_steps(self.pkg, broken), "step_order不重复")
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["blocking"])

    def test_parent_step_must_resolve(self) -> None:
        broken = list(self.specs)
        broken[1] = replace(broken[1], parent_topology_step_id="TS_MISSING")
        row = self._check(validate_topology_steps(self.pkg, broken), "parent_topology_step_id可解析")
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["blocking"])

    def test_parent_chain_must_be_acyclic(self) -> None:
        broken = list(self.specs)
        broken[0] = replace(broken[0], parent_topology_step_id="TS304")
        row = self._check(validate_topology_steps(self.pkg, broken), "父步骤链无环")
        self.assertEqual(row["status"], "FAIL")

    def test_future_parent_reference_is_blocked(self) -> None:
        broken = list(self.specs)
        broken[1] = replace(broken[1], parent_topology_step_id="TS102")
        row = self._check(validate_topology_steps(self.pkg, broken), "不引用未来步骤")
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["blocking"])

    def test_added_part_must_exist(self) -> None:
        broken = list(self.specs)
        broken[1] = replace(broken[1], added_part_ids=("P_UNKNOWN",))
        row = self._check(validate_topology_steps(self.pkg, broken), "added_part_ids零件存在")
        self.assertEqual(row["status"], "FAIL")

    def test_active_interface_endpoints_must_be_members(self) -> None:
        broken = list(self.specs)
        broken[1] = replace(broken[1], added_part_ids=())
        row = self._check(validate_topology_steps(self.pkg, broken), "活动接口两端属于当前子装配体")
        self.assertEqual(row["status"], "FAIL")

    def test_ts301_activates_two_interfaces_in_one_step(self) -> None:
        spec = next(s for s in self.specs if s.topology_step_id == "TS301")
        self.assertEqual(set(spec.activated_interface_ids), {"G_AD", "G_DB"})
        self.assertEqual(self.result["TS301"]["lcp_call_count"], 1)
        self.assertEqual(set(self.result["TS301"]["active_interface_ids"]), {"G_AB", "G_BC", "G_AD", "G_DB"})

    def test_all_solved_steps_make_exactly_one_global_lcp_call(self) -> None:
        original = topology_step_module.solve_lcp_active_set
        with patch("core.topology_step.solve_lcp_active_set", wraps=original) as mocked:
            result = run_topology_steps(self.pkg)
        self.assertEqual(mocked.call_count, sum(spec.solve_required for spec in self.specs))
        self.assertEqual(result["TS000"]["lcp_call_count"], 0)
        self.assertTrue(all(step["lcp_call_count"] == 1 for key, step in result.items() if key != "TS000"))

    def test_solver_call_dimension_proves_no_per_interface_split(self) -> None:
        dimensions: list[int] = []
        original = topology_step_module.solve_lcp_active_set

        def recording_solver(q, W, **kwargs):
            dimensions.append(len(q))
            return original(q, W, **kwargs)

        with patch("core.topology_step.solve_lcp_active_set", side_effect=recording_solver):
            run_topology_steps(self.pkg)
        solved_ids = [spec.topology_step_id for spec in self.specs if spec.solve_required]
        self.assertEqual(dict(zip(solved_ids, dimensions))["TS301"], 12)
        self.assertEqual(len(dimensions), len(solved_ids))

    def test_active_submatrix_keeps_cross_interface_blocks(self) -> None:
        step = self.result["TS301"]
        indices = np.flatnonzero(step["active_index_mask"])
        np.testing.assert_allclose(step["W_active"], step["W_total"][np.ix_(indices, indices)])
        self.assertGreater(np.linalg.norm(step["W_active"][6:9, 9:12]), 0.0)

    def test_distinct_interfaces_are_not_collapsed_by_endpoint_or_activation(self) -> None:
        step = self.result["TS301"]
        self.assertEqual(len(step["active_interface_ids"]), 4)
        self.assertEqual(len(set(step["active_interface_ids"])), 4)
        self.assertIn("G_AD", step["active_interface_ids"])
        self.assertIn("G_DB", step["active_interface_ids"])

    def test_subassembly_membership_changes_by_step(self) -> None:
        expected = {
            "TS000": ["P_A"], "TS101": ["P_A", "P_B"],
            "TS201": ["P_A", "P_B", "P_C"],
            "TS301": ["P_A", "P_B", "P_C", "P_D"],
        }
        for step_id, parts in expected.items():
            self.assertEqual(self.result[step_id]["stage_state"].active_part_ids, parts)

    def test_deactivated_interface_is_excluded_from_lcp(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        route = pkg.raw_tables["I0/assembly_topology.csv"].copy()
        route["deactivated_interface_ids"] = route["deactivated_interface_ids"].astype(object)
        route.loc[route["topology_step_id"].eq("TS302"), "deactivated_interface_ids"] = "G_BC"
        pkg.raw_tables["I0/assembly_topology.csv"] = route
        result = run_topology_steps(pkg)
        step = result["TS302"]
        self.assertNotIn("G_BC", step["active_interface_ids"])
        self.assertEqual(step["W_active"].shape, (9, 9))
        self.assertTrue(np.isnan(step["gap_full"][3:6]).all())
        self.assertTrue((step["lambda_full"][3:6] == 0.0).all())

    def test_active_index_mask_matches_vector_layout(self) -> None:
        self.assertEqual(int(self.result["TS101"]["active_index_mask"].sum()), 3)
        self.assertEqual(int(self.result["TS201"]["active_index_mask"].sum()), 6)
        self.assertEqual(int(self.result["TS301"]["active_index_mask"].sum()), 12)

    def test_init_generates_not_required_state(self) -> None:
        init = self.result["TS000"]
        self.assertEqual(init["solve_status"], "NOT_REQUIRED")
        self.assertEqual(init["solution"].convergence_status, "NOT_REQUIRED")
        self.assertEqual(init["stage_state"].parent_stage_state_id, None)

    def test_each_join_generates_one_unique_lock_history(self) -> None:
        join_steps = ["TS103", "TS203", "TS303"]
        identifiers = []
        for step_id in join_steps:
            records = self.result[step_id]["connection_lock_history"]
            self.assertEqual(len(records), 1)
            identifiers.append(records[0]["lock_history_id"])
        self.assertEqual(len(set(identifiers)), len(identifiers))
        self.assertEqual(self.result["TS303"]["connection_lock_history"][0]["joint_ids"], "JNT_AD;JNT_DB")

    def test_release_inherits_all_prior_join_locks(self) -> None:
        self.assertEqual(len(self.result["TS104"]["stage_state"].connection_lock_history_ids), 1)
        self.assertEqual(len(self.result["TS204"]["stage_state"].connection_lock_history_ids), 2)
        self.assertEqual(len(self.result["TS304"]["stage_state"].connection_lock_history_ids), 3)
        release = self.result["TS304"]["release_history"][0]
        self.assertEqual(len(release["lock_history_ids"].split(";")), 3)
        self.assertEqual(set(release["retained_joint_ids"].split(";")), {"JNT_AB", "JNT_BC", "JNT_AD", "JNT_DB"})

    def test_repeated_stage_types_execute_in_multiple_cycles(self) -> None:
        operations = [spec.operation_type for spec in self.specs]
        self.assertEqual(operations.count("LOCATE"), 3)
        self.assertEqual(operations.count("CLAMP"), 3)
        self.assertEqual(operations.count("JOIN"), 3)
        self.assertEqual(operations.count("RELEASE"), 3)
        self.assertEqual(len(self.result), 13)

    def test_state_parent_chain_replays_to_ts000_without_overwrite(self) -> None:
        state_ids = [step["stage_state"].stage_state_id for step in self.result.values()]
        self.assertEqual(len(state_ids), len(set(state_ids)))
        current = "TS304"
        replay = []
        while current:
            replay.append(current)
            current = self.result[current]["stage_state"].parent_topology_step_id
        self.assertEqual(replay[-1], "TS000")
        self.assertEqual(len(replay), 13)

    def test_new_parts_use_sms_once_and_existing_parts_inherit(self) -> None:
        final_parts = self.result["TS304"]["stage_state"].part_state
        self.assertEqual(set(final_parts), {"P_A", "P_B", "P_C", "P_D"})
        for state in final_parts.values():
            self.assertEqual(state["initialization_source"], "ASSEMBLY_BEFORE_SMS")
            self.assertEqual(state["initialization_count"], 1)
            self.assertTrue(state["sms_prior_found"])
            self.assertTrue(state["sms_prior_id"])
            self.assertTrue(state["sms_basis_id"])
            self.assertTrue(state["parent_part_state_id"])

    def test_legacy_four_stage_route_is_generated(self) -> None:
        legacy = load_package(DATA / "01_DEFAULT_MIN_CASE")
        specs = load_topology_steps(legacy)
        self.assertEqual([s.topology_step_id for s in specs], [
            "LEGACY_TS_LOCATE", "LEGACY_TS_CLAMP", "LEGACY_TS_JOIN", "LEGACY_TS_RELEASE",
        ])
        self.assertTrue(all(s.adapter_source == "LEGACY_STAGE_ADAPTER" for s in specs))

    def test_legacy_wrapper_is_numerically_equivalent_to_run_stage(self) -> None:
        legacy = load_package(DATA / "01_DEFAULT_MIN_CASE")
        wrapped = run_all_stages(legacy)
        for stage_id, step in wrapped.items():
            direct = run_stage(legacy, stage_id)
            np.testing.assert_allclose(step["solution"].lambda_n, direct["solution"].lambda_n, rtol=0, atol=1e-12)
            np.testing.assert_allclose(step["solution"].gap_g, direct["solution"].gap_g, rtol=0, atol=1e-12)

    def test_missing_operator_set_is_blocking(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.matrices.pop("Q_OP_TS101")
        report = validate_topology_steps(pkg)
        self.assertEqual(self._check(report, "operator_set_id可解析")["status"], "FAIL")
        with self.assertRaises(TopologyStepValidationError):
            run_topology_steps(pkg)

    def test_operator_and_manifest_shape_mismatch_is_blocking(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.matrices["Q_OP_TS101"] = pkg.matrices["Q_OP_TS101"][:-1]
        row = self._check(validate_topology_steps(pkg), "MatrixManifest与VectorLayout维度一致")
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["blocking"])

    def test_legacy_fallback_is_explicitly_marked(self) -> None:
        legacy = load_package(DATA / "01_DEFAULT_MIN_CASE")
        result = run_topology_steps(legacy)
        for step in result.values():
            self.assertEqual(step["operator_source"], LEGACY_OPERATOR)
            self.assertTrue(step["fallback_flag"])
            self.assertEqual(step["fallback_reason"], LEGACY_FALLBACK_REASON)

    def test_real_steps_record_precomputed_operator_source(self) -> None:
        for step_id, step in self.result.items():
            self.assertEqual(step["operator_source"], PRECOMPUTED_OPERATOR)
            self.assertFalse(step["fallback_flag"])
            if step_id != "TS000":
                self.assertTrue(step["operator_set_id"])

    def test_synthetic_fixture_disallows_engineering_claims(self) -> None:
        self.assertIs(self.pkg.manifest["engineering_claim_allowed"], False)
        self.assertIn("不代表真实工程预测结果", data_truthfulness_statement(self.pkg))

    def test_independent_lcp_oracle_matches_every_solved_step(self) -> None:
        oracle = self.pkg.raw_tables["validation/topology_step_lcp_oracle.csv"]
        for row in oracle.to_dict("records"):
            step = self.result[str(row["topology_step_id"])]
            expected_lambda = np.asarray(json.loads(row["lambda_active"]), dtype=float)
            expected_gap = np.asarray(json.loads(row["gap_active"]), dtype=float)
            # Keep the existing fixture-oracle acceptance tolerance used by the
            # solver contract; the active-set solver intentionally regularizes
            # the active linear system by 1e-12.
            np.testing.assert_allclose(step["lambda_active"], expected_lambda, rtol=0, atol=1e-7)
            np.testing.assert_allclose(step["gap_active"], expected_gap, rtol=0, atol=1e-7)

    def test_detailed_validator_allows_fixture(self) -> None:
        report = validate_package_detailed(self.pkg)
        self.assertFalse(has_blocking_failures(report), report[report["status"].eq("FAIL")].to_string(index=False))

    def test_report_zip_contains_all_topology_step_contract_files(self) -> None:
        kcp = extract_kcp(self.pkg, self.result)
        validation = compare_validation(kcp, self.pkg.validation_kcp)
        payload = build_runtime_report_zip(self.pkg, self.result, validation, [])
        expected = {
            "topology_step_execution.csv", "topology_step_validation.csv",
            "active_subassembly_history.csv", "topology_step_state_lineage.csv",
            "topology_step_operator_usage.csv", "topology_step_contact_summary.csv",
            "connection_lock_history.csv", "release_history.csv",
        }
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertTrue(expected <= set(archive.namelist()), expected - set(archive.namelist()))

    def test_execution_table_reports_each_step_lcp_count(self) -> None:
        table = topology_step_execution_table(self.result).set_index("topology_step_id")
        self.assertEqual(table.loc["TS000", "lcp_call_count"], 0)
        self.assertTrue((table.drop(index="TS000")["lcp_call_count"] == 1).all())

    def test_page_selects_topology_step_and_shows_route_audit(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("topology_step 选择", source)
        self.assertIn("topology_step 工艺路线表", source)
        self.assertIn("确定性工艺路线时间轴", source)
        self.assertIn("JOIN 锁定历史", source)
        self.assertIn("RELEASE 继承记录", source)

    def test_streamlit_page_selects_real_topology_step(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        package_selector = next(box for box in app.selectbox if box.label == "标准输入包目录")
        fixture_option = next(option for option in package_selector.options if "02_TOPOLOGY_STEP_MIN_CASE" in option)
        package_selector.set_value(fixture_option).run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        navigation = next(radio for radio in app.radio if radio.key == "main_page_navigation")
        navigation.set_value("⑦ topology_step / 四阶段兼容接触求解").run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        solve_selector = next(box for box in app.selectbox if box.label == "查看 topology_step 接触点结果")
        solve_ts301 = next(option for option in solve_selector.options if option.startswith("TS301 - LOCATE"))
        solve_selector.set_value(solve_ts301).run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        navigation = next(radio for radio in app.radio if radio.key == "main_page_navigation")
        navigation.set_value("⑬ 装配拓扑、阶段路径与状态传递").run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        step_selector = next(box for box in app.selectbox if box.label == "topology_step 选择")
        self.assertTrue(step_selector.options[0].startswith("TS000 - INIT"))
        self.assertTrue(step_selector.options[-1].startswith("TS304 - RELEASE"))
        ts301_option = next(option for option in step_selector.options if option.startswith("TS301 - LOCATE"))
        step_selector.set_value(ts301_option).run(timeout=60)
        self.assertFalse(app.exception, app.exception)

    def test_executor_has_no_fixture_part_interface_or_step_count_constants(self) -> None:
        source = (ROOT / "core" / "topology_step.py").read_text(encoding="utf-8")
        for forbidden in ("P_A", "P_B", "P_C", "P_D", "G_AB", "G_BC", "G_AD", "G_DB", "range(13)"):
            self.assertNotIn(forbidden, source)

    def test_new_fixture_cli_passes_with_current_python311(self) -> None:
        self.assertEqual(sys.version_info[:2], (3, 11))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cli_check.py"), str(FIXTURE)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
        self.assertIn("TS304", completed.stdout)


if __name__ == "__main__":
    unittest.main()
