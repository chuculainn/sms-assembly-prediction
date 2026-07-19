from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path
import unittest
import warnings
import zipfile

import numpy as np
import pandas as pd

from core.data_loader import load_package
from core.kcp import compare_validation, extract_kcp
from core.multi_part import (
    assembly_graph,
    assembly_path_summary,
    contribution_ledger_summary,
    coupling_ablation_comparison,
    coupling_block_summary,
    kcp_contribution_path,
    topology_summary,
    vector_layout,
)
from core.package_validator import (
    data_truthfulness_statement,
    has_blocking_failures,
    validate_package_detailed,
)
from core.reporting import build_runtime_report_zip
from core.stage_solver import run_all_stages
from core.stage_state import stage_transition_runtime_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class MultiPartRound2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pkg = load_package(DATA / "01_DEFAULT_MIN_CASE_4_PART")
        cls.result = run_all_stages(cls.pkg)

    def test_detailed_validator_passes_bundled_multi_part_case(self) -> None:
        report = validate_package_detailed(self.pkg)
        self.assertFalse(has_blocking_failures(report), report[report["status"].eq("FAIL")].to_string(index=False))
        self.assertFalse(report[report["status"].eq("FAIL")].any().any())
        self.assertTrue({"PASS", "WARN", "FAIL"}.issuperset(set(report["status"])))

    def test_detailed_validator_preserves_all_bundled_package_families(self) -> None:
        for package_dir in sorted(DATA.iterdir()):
            if not package_dir.is_dir():
                continue
            package = load_package(package_dir)
            report = validate_package_detailed(package)
            self.assertFalse(
                has_blocking_failures(report),
                f"{package_dir.name}\n{report[report['status'].eq('FAIL')].to_string(index=False)}",
            )

    def test_validator_reports_file_field_object_and_suggestion(self) -> None:
        report = validate_package_detailed(self.pkg)
        for column in ("file", "field", "object_id", "suggestion", "blocking"):
            self.assertIn(column, report.columns)

    def test_arbitrary_n_m_topology_is_data_driven(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.parts = pd.DataFrame([
            {"part_id": f"P{i}", "role_in_assembly": "flexible" if i == 1 else "member"}
            for i in range(5)
        ])
        pkg.interfaces = pd.DataFrame([
            {"interface_id": "G01", "part_i": "P0", "part_j": "P1"},
            {"interface_id": "G12", "part_i": "P1", "part_j": "P2"},
            {"interface_id": "G13", "part_i": "P1", "part_j": "P3"},
            {"interface_id": "G30", "part_i": "P3", "part_j": "P0"},
            {"interface_id": "G24", "part_i": "P2", "part_j": "P4"},
        ])
        topo = topology_summary(pkg)
        self.assertEqual(topo["part_count"], 5)
        self.assertEqual(topo["interface_count"], 5)
        self.assertTrue(topo["connected"])
        self.assertTrue(topo["has_serial_path"])
        self.assertTrue(topo["has_closed_or_parallel_path"])
        self.assertIn("P1", topo["shared_flexible_part_ids"])

    def test_parallel_interfaces_between_same_parts_are_lossless(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.parts = pd.DataFrame([
            {"part_id": "A"}, {"part_id": "B"}, {"part_id": "C"},
        ])
        pkg.interfaces = pd.DataFrame([
            {"interface_id": "G_AB_1", "part_i": "A", "part_j": "B"},
            {"interface_id": "G_AB_2", "part_i": "A", "part_j": "B"},
            {"interface_id": "G_BC", "part_i": "B", "part_j": "C"},
        ])
        graph = assembly_graph(pkg)
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertEqual(set(graph.get_edge_data("A", "B")), {"G_AB_1", "G_AB_2"})
        self.assertEqual(topology_summary(pkg)["interface_count"], 3)
        displayed_interfaces = ";".join(assembly_path_summary(pkg)["interface_ids"].astype(str))
        self.assertIn("G_AB_1", displayed_interfaces)
        self.assertIn("G_AB_2", displayed_interfaces)

    def test_vector_layout_covers_all_points_without_overlap(self) -> None:
        layout = vector_layout(self.pkg)
        covered = [
            index
            for _, row in layout.iterrows()
            for index in range(int(row["start_index"]), int(row["end_index"]) + 1)
        ]
        self.assertEqual(covered, list(range(len(self.pkg.contact_points))))

    def test_layout_overlap_is_blocking(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        layout = pkg.raw_tables["matrices/vector_layout.csv"].copy()
        layout.loc[1, "start_index"] = layout.loc[0, "end_index"]
        pkg.raw_tables["matrices/vector_layout.csv"] = layout
        report = validate_package_detailed(pkg)
        row = report[report["check_item"].eq("VectorLayout连续无重叠且全覆盖")].iloc[0]
        self.assertEqual(row["status"], "FAIL")
        self.assertTrue(row["blocking"])

    def test_invalid_interface_foreign_key_blocks_solve(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        broken = pkg.raw_tables["I0/interface.csv"].copy()
        broken.loc[0, "part_i"] = "P_DOES_NOT_EXIST"
        pkg.raw_tables["I0/interface.csv"] = broken
        report = validate_package_detailed(pkg)
        self.assertTrue(has_blocking_failures(report))
        self.assertTrue((report["check_item"].eq("外键:part_i") & report["status"].eq("FAIL")).any())

    def test_serial_parallel_and_cycle_paths_are_reported(self) -> None:
        paths = assembly_path_summary(self.pkg)
        self.assertIn("SERIAL", set(paths["path_type"]))
        self.assertIn("PARALLEL", set(paths["path_type"]))
        self.assertIn("CYCLE", set(paths["path_type"]))

    def test_all_six_four_part_cross_blocks_are_retained_with_metrics(self) -> None:
        for stage_id in self.result:
            blocks = coupling_block_summary(self.pkg, stage_id)
            cross = blocks[blocks["cross_interface"]]
            self.assertEqual(len(cross), 6)
            self.assertFalse(cross["zero_block"].any())
            self.assertTrue(np.isfinite(cross["relative_coupling_strength"]).all())
            self.assertTrue((cross["max_absolute_value"] > 0).all())

    def test_coupling_ablation_changes_solution_and_is_diagnostic(self) -> None:
        comparison = coupling_ablation_comparison(self.pkg, self.result, "S_JOIN_03", threshold=0.0)
        summary = comparison["summary"].iloc[0]
        self.assertGreater(summary["lambda_max_absolute_change_N"], 0.0)
        self.assertTrue(summary["warning_flag"])
        self.assertTrue((comparison["kcp_comparison"]["result_role"] == "DIAGNOSTIC_NOT_FORMAL_ENGINEERING_RESULT").all())

    def test_runtime_stage_parent_chain_is_continuous(self) -> None:
        runtime = stage_transition_runtime_table(self.result)
        self.assertEqual(len(runtime), len(self.pkg.stage_plan))
        self.assertEqual(runtime.iloc[0]["parent_stage_state_id"], "")
        for previous, current in zip(runtime.to_dict("records"), runtime.to_dict("records")[1:]):
            self.assertEqual(current["parent_stage_state_id"], previous["stage_state_id"])

    def test_stage_state_names_contact_response_without_changing_values(self) -> None:
        first_result = next(iter(self.result.values()))
        state = first_result["stage_state"]
        expected = first_result["W_struct"] @ first_result["solution"].lambda_n
        np.testing.assert_allclose(state.contact_structural_response, expected)
        record = state.to_record()
        self.assertIn("contact_structural_response_norm_mm", record)
        self.assertNotIn("displacement_norm_mm", record)
        self.assertEqual(
            record["response_type"],
            "CONTACT_STRUCTURAL_FLEXIBILITY_RESPONSE_W_STRUCT_TIMES_LAMBDA",
        )

    def test_locate_runtime_state_records_initial_boundaries_and_loads(self) -> None:
        first = next(iter(self.result.values()))["stage_state"]
        self.assertGreater(len(first.boundary_state["active"]), 0)
        self.assertEqual(first.boundary_state["activated"], first.boundary_state["active"])
        self.assertGreater(len(first.load_state["active"]), 0)
        self.assertEqual(first.load_state["activated"], first.load_state["active"])

    def test_release_inherits_join_lock_history(self) -> None:
        states = {stage["stage_state"].stage_type: stage["stage_state"] for stage in self.result.values()}
        self.assertEqual(
            states["RELEASE"].joint_lock_state["lock_history_id"],
            states["JOIN"].joint_lock_state["lock_history_id"],
        )

    def test_duplicate_contribution_key_is_rejected(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        records = pkg.raw_tables["prediction/contribution_record.csv"].copy()
        pkg.raw_tables["prediction/contribution_record.csv"] = pd.concat([records, records.iloc[[0]]], ignore_index=True)
        ledger = contribution_ledger_summary(pkg)
        self.assertEqual(ledger["status"], "FAIL")
        self.assertEqual(ledger["duplicate_count"], 1)

    def test_kcp_contribution_path_filters_other_kcp_sources(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.parts = pd.DataFrame([
            {"part_id": "A"}, {"part_id": "B"}, {"part_id": "C"},
        ])
        pkg.interfaces = pd.DataFrame([
            {"interface_id": "G1", "part_i": "A", "part_j": "C"},
            {"interface_id": "G2", "part_i": "B", "part_j": "C"},
        ])
        pkg.kcp_kcm = pd.DataFrame([
            {"feature_id": "KCP_A", "target_part_or_interface": "A"},
            {"feature_id": "KCP_B", "target_part_or_interface": "B"},
        ])
        pkg.raw_tables["prediction/contribution_record.csv"] = pd.DataFrame([
            {"target_kcp_id": "KCP_A", "source_id": "A"},
            {"target_kcp_id": "KCP_A;KCP_SHARED", "source_id": "G1"},
            {"target_kcp_id": "KCP_B", "source_id": "B"},
            {"target_kcp_id": "KCP_B;KCP_SHARED", "source_id": "G2"},
        ])
        path = kcp_contribution_path(pkg, "KCP_A")
        self.assertIn("A", path["parts"])
        self.assertIn("G1", path["interfaces"])
        self.assertNotIn("B", path["parts"])
        self.assertNotIn("G2", path["interfaces"])

    def test_contribution_ledger_reconstructs_each_sample_independently(self) -> None:
        pkg = copy.deepcopy(self.pkg)
        pkg.raw_tables["prediction/contribution_record.csv"] = pd.DataFrame([
            {
                "sample_id": "SAMPLE_A", "target_kcp_id": "KCP_A", "source_class": "SMS",
                "source_id": "SOURCE_A", "origin_stage_id_optional": "", "increment_definition_id": "ABS",
                "contribution_vector": "[1.25]", "consumed_by_prediction_id": "PRED_A",
            },
            {
                "sample_id": "SAMPLE_B", "target_kcp_id": "KCP_A", "source_class": "SMS",
                "source_id": "SOURCE_B", "origin_stage_id_optional": "", "increment_definition_id": "ABS",
                "contribution_vector": "[2.5]", "consumed_by_prediction_id": "PRED_B",
            },
        ])
        pkg.raw_tables["prediction/kcp_prediction_result.csv"] = pd.DataFrame([
            {"prediction_result_id": "PRED_A", "sample_id": "SAMPLE_A", "kcp_ids": "KCP_A", "predicted_values": "[1.25]"},
            {"prediction_result_id": "PRED_B", "sample_id": "SAMPLE_B", "kcp_ids": "KCP_A", "predicted_values": "[2.5]"},
        ])
        ledger = contribution_ledger_summary(pkg)
        self.assertEqual(ledger["status"], "PASS")
        self.assertEqual(ledger["prediction_group_count"], 2)
        groups = ledger["group_results"].set_index("sample_id")
        self.assertEqual(set(groups.index), {"SAMPLE_A", "SAMPLE_B"})
        self.assertEqual(groups.loc["SAMPLE_A", "reconstructed_values"], [1.25])
        self.assertEqual(groups.loc["SAMPLE_B", "reconstructed_values"], [2.5])
        self.assertTrue((groups["reconstruction_error"] == 0.0).all())

    def test_blocking_mask_has_no_fillna_downcasting_warning(self) -> None:
        validation = pd.DataFrame({
            "status": ["PASS", "FAIL"],
            "blocking": pd.Series([False, pd.NA], dtype="boolean"),
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertFalse(has_blocking_failures(validation))
        self.assertFalse(any("downcasting" in str(item.message).lower() for item in caught))

    def test_synthetic_truthfulness_statement_disallows_engineering_claim(self) -> None:
        statement = data_truthfulness_statement(self.pkg)
        self.assertIn("engineering_claim_allowed=False", statement)
        self.assertIn("不代表真实工程预测结果", statement)

    def test_report_zip_contains_all_new_contract_files(self) -> None:
        kcp = extract_kcp(self.pkg, self.result)
        validation = compare_validation(kcp, self.pkg.validation_kcp)
        payload = build_runtime_report_zip(self.pkg, self.result, validation, [])
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
        expected = {
            "topology_summary.csv", "assembly_path_summary.csv", "stage_transition_runtime.csv",
            "interface_stage_summary.csv", "cross_interface_coupling_blocks.csv",
            "coupling_ablation_comparison.csv", "state_lineage.csv", "validation_summary.csv",
            "data_truthfulness_statement.txt",
        }
        self.assertTrue(expected <= names, expected - names)

    def test_new_streamlit_pages_and_isolation_guards_exist(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("装配拓扑、阶段路径与状态传递", source)
        self.assertIn("接口耦合诊断与对照试算", source)
        self.assertEqual(source.count("if active_page == TABS["), 14)
        self.assertNotIn("use_container_width=True", source)


class StreamlitSmokeTests(unittest.TestCase):
    def test_app_and_new_pages_render_without_exception(self) -> None:
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception, app.exception)
        package_selector = next(box for box in app.selectbox if box.label == "标准输入包目录")
        four_part_option = next(option for option in package_selector.options if "01_DEFAULT_MIN_CASE_4_PART" in option)
        package_selector.set_value(four_part_option).run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        navigation = next(radio for radio in app.radio if radio.key == "main_page_navigation")
        navigation.set_value("⑬ 装配拓扑、阶段路径与状态传递").run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        navigation = next(radio for radio in app.radio if radio.key == "main_page_navigation")
        navigation.set_value("⑭ 接口耦合诊断与对照试算").run(timeout=60)
        self.assertFalse(app.exception, app.exception)


if __name__ == "__main__":
    unittest.main()
