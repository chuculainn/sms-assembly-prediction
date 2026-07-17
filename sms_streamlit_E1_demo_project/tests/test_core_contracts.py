from __future__ import annotations

import unittest
from pathlib import Path
import re

import numpy as np

from core.data_loader import load_package
from core.fallback import FallbackSettings, evaluate_validity_and_fallback
from core.kcp import extract_kcp
from core.lcp_solver import solve_lcp_active_set
from core.monte_carlo import distribution_defaults, run_monte_carlo
from core.multi_part import contribution_ledger_summary, coupling_block_summary, topology_summary, vector_layout
from core.numerical_substitution import (
    NumericalSubstitutionSettings,
    assemble_Cn_matrix,
    release_rebound_increment,
)
from core.overconstraint import OverConstraintSettings, build_extended_lcp_model
from core.physical_consistency import physical_consistency_report
from core.sms_mapping import SMSMappingSettings, fit_sms_wls_map, rebuild_gap_from_sms
from core.stage_solver import build_stage_vectors, run_all_stages
from core.tangential_ncp import solve_tangential_projection
from core.validation import validate_package


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class LCPFormulaTests(unittest.TestCase):
    def test_one_point_contact_has_analytic_solution(self) -> None:
        sol = solve_lcp_active_set(np.array([-0.2]), np.array([[0.01]]))
        np.testing.assert_allclose(sol.lambda_n, [20.0], atol=1e-8)
        np.testing.assert_allclose(sol.gap_g, [0.0], atol=1e-8)
        self.assertEqual(sol.convergence_status, "CONVERGED")

    def test_open_contact_has_zero_force(self) -> None:
        sol = solve_lcp_active_set(np.array([0.1, 0.2]), np.eye(2))
        np.testing.assert_allclose(sol.lambda_n, [0.0, 0.0])
        np.testing.assert_allclose(sol.gap_g, [0.1, 0.2])

    def test_mixed_contact_satisfies_complementarity(self) -> None:
        q = np.array([-0.2, 0.1])
        W = np.diag([0.01, 0.02])
        sol = solve_lcp_active_set(q, W)
        self.assertLess(sol.residuals["complementarity_residual"], 1e-8)
        self.assertGreaterEqual(float(sol.lambda_n.min()), -1e-10)
        self.assertGreaterEqual(float(sol.gap_g.min()), -1e-8)

    def test_dimension_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_lcp_active_set(np.zeros(2), np.eye(3))

    def test_tangential_projection_respects_coulomb_cone(self) -> None:
        sol = solve_tangential_projection(
            lambda_n=np.array([10.0, 0.0]),
            q_t=np.array([[1.0, 0.0], [1.0, 1.0]]),
            Ct_diag=np.array([0.01, 0.01]),
            mu=np.array([0.3, 0.3]),
        )
        self.assertLessEqual(float(np.linalg.norm(sol["lambda_t"][0])), 3.0 + 1e-10)
        np.testing.assert_allclose(sol["lambda_t"][1], [0.0, 0.0])


class PackageAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = load_package(DATA / "E1_manual_input_9pt")
        cls.v25 = load_package(DATA / "01_DEFAULT_MIN_CASE")
        cls.multi = load_package(DATA / "01_DEFAULT_MIN_CASE_4_PART")

    def test_both_schema_families_load(self) -> None:
        self.assertTrue(self.legacy.package_type.startswith("E1"))
        self.assertTrue(self.v25.package_type.startswith("V25"))

    def test_quality_checks_have_no_fail_for_bundled_packages(self) -> None:
        for pkg in (self.legacy, self.v25, self.multi):
            checks = validate_package(pkg)
            self.assertFalse((checks["status"] == "FAIL").any(), checks.to_string(index=False))

    def test_q_and_W_follow_documented_formula(self) -> None:
        sid = "S_CLAMP_02"
        q, W, W_struct, Cn, _, _ = build_stage_vectors(self.legacy, sid)
        expected_q = (
            self.legacy.matrices["nominal_gap"]
            + self.legacy.matrices["sms_component"]
            + self.legacy.matrices.get("pose_component", 0.0)
            - self.legacy.matrices[f"u_free__{sid}"]
        )
        np.testing.assert_allclose(q, expected_q)
        np.testing.assert_allclose(W, W_struct + Cn)

    def test_all_stages_satisfy_basic_lcp_physics(self) -> None:
        result = run_all_stages(self.legacy)
        for stage in result.values():
            sol = stage["solution"]
            self.assertEqual(sol.convergence_status, "CONVERGED")
            self.assertLess(sol.residuals["gap_violation"], 1e-7)
            self.assertLess(sol.residuals["force_violation"], 1e-10)
            self.assertLess(sol.residuals["complementarity_residual"], 1e-6)

    def test_pressure_and_local_compression_formulas(self) -> None:
        result = run_all_stages(self.legacy)
        stage = result["S_JOIN_03"]
        lam = stage["solution"].lambda_n
        area = self.legacy.contact_points["area_weight"].to_numpy(float)
        np.testing.assert_allclose(stage["pressure"], lam / area)
        np.testing.assert_allclose(stage["local_compression"], stage["Cn"] @ lam)

    def test_kcp_values_are_finite(self) -> None:
        kcp = extract_kcp(self.legacy, run_all_stages(self.legacy))
        self.assertGreater(len(kcp), 0)
        self.assertTrue(np.isfinite(kcp["predicted_value"].to_numpy(float)).all())

    def test_sms_wls_mapping_produces_auditable_outputs(self) -> None:
        fit = fit_sms_wls_map(self.legacy, SMSMappingSettings(enabled=True))
        rebuilt = rebuild_gap_from_sms(self.legacy, SMSMappingSettings(enabled=True))
        self.assertFalse(fit.empty)
        self.assertEqual(len(rebuilt["g0"]), len(self.legacy.contact_points))
        self.assertIn("difference_from_package_mm", rebuilt["gap_table"].columns)

    def test_v25_sms_mapping_safely_uses_frozen_field_when_raw_points_are_incomplete(self) -> None:
        rebuilt = rebuild_gap_from_sms(self.v25, SMSMappingSettings(enabled=True))
        self.assertEqual(rebuilt["mapping_mode"], "PACKAGE_FROZEN_FALLBACK")
        np.testing.assert_allclose(rebuilt["g0"], self.v25.matrices["g0"])
        self.assertTrue((rebuilt["quality"]["status"] == "WARN").any())

    def test_legacy_partial_sms_mapping_remains_executable(self) -> None:
        rebuilt = rebuild_gap_from_sms(self.legacy, SMSMappingSettings(enabled=True))
        self.assertEqual(rebuilt["mapping_mode"], "LIVE_REBUILD_PARTIAL_LEGACY")

    def test_substitution_matrix_is_symmetric_and_nonnegative(self) -> None:
        settings = NumericalSubstitutionSettings(enabled=True, mode="replace")
        Cn, components = assemble_Cn_matrix(self.legacy, "S_JOIN_03", settings=settings)
        np.testing.assert_allclose(Cn, Cn.T)
        self.assertGreaterEqual(float(np.diag(Cn).min()), 0.0)
        self.assertIn("Cn_runtime_mm_per_N", components.columns)

    def test_v25_replace_without_substitution_data_preserves_package_Cn(self) -> None:
        settings = NumericalSubstitutionSettings(enabled=True, mode="replace")
        Cn, components = assemble_Cn_matrix(self.v25, "S_JOIN_03", settings=settings)
        np.testing.assert_allclose(Cn, self.v25.matrices["Cn_local"])
        self.assertTrue(components["substitution_fallback_flag"].all())

    def test_all_v25_packages_run_with_all_advanced_switches(self) -> None:
        names = [
            "01_DEFAULT_MIN_CASE",
            "01_DEFAULT_MIN_CASE_4_PART",
            "E1_min_closed_loop_V25_DEFAULT_CASE",
            "E1_manual_input_9pt_V25_DEFAULT_CASE",
        ]
        for name in names:
            with self.subTest(package=name):
                pkg = load_package(DATA / name)
                result = run_all_stages(
                    pkg,
                    substitution_settings=NumericalSubstitutionSettings(enabled=True, mode="replace"),
                    sms_mapping_settings=SMSMappingSettings(enabled=True),
                    overconstraint_settings=OverConstraintSettings(enabled=True),
                    tangential_settings={"enabled": True},
                )
                self.assertEqual(set(result), set(pkg.stage_plan["stage_id"].astype(str)))
                for stage in result.values():
                    self.assertEqual(stage["solution"].convergence_status, "CONVERGED")
                    self.assertIsNotNone(stage["sms_rebuild"])

    def test_multi_part_package_loads_as_one_global_problem(self) -> None:
        self.assertEqual(self.multi.package_type, "V25_MULTI_PART")
        self.assertEqual(len(self.multi.parts), 4)
        self.assertEqual(len(self.multi.interfaces), 4)
        self.assertEqual(len(self.multi.contact_points), 12)
        self.assertEqual(self.multi.stage_plan["stage_id"].tolist(), [
            "S_LOCATE_01", "S_CLAMP_02", "S_JOIN_03", "S_RELEASE_04",
        ])
        self.assertEqual(vector_layout(self.multi)[["start_index", "end_index"]].astype(int).values.tolist(), [
            [0, 2], [3, 5], [6, 8], [9, 11],
        ])

    def test_multi_part_gap_fields_are_assembled_by_domain_and_local_index(self) -> None:
        np.testing.assert_allclose(self.multi.matrices["g0"], [
            0.08, 0.04, 0.06, 0.05, 0.03, 0.04,
            0.10, 0.07, 0.09, 0.09, 0.06, 0.08,
        ])
        self.assertEqual(self.multi.contact_points["interface_id"].nunique(), 4)

    def test_multi_part_live_sms_mapping_uses_each_interface_endpoints(self) -> None:
        rebuilt = rebuild_gap_from_sms(self.multi, SMSMappingSettings(enabled=True))
        self.assertEqual(rebuilt["mapping_mode"], "LIVE_REBUILD_MULTI_INTERFACE")
        pairs = rebuilt["gap_table"].groupby("interface_id")[["part_i", "part_j"]].first()
        expected = self.multi.interfaces.set_index("interface_id")[["part_i", "part_j"]].sort_index()
        self.assertTrue(pairs.sort_index().equals(expected))

    def test_multi_part_coupled_lcp_matches_fixture_oracle(self) -> None:
        result = run_all_stages(self.multi)
        for sid, stage in result.items():
            suffix = sid.removeprefix("S_")
            np.testing.assert_allclose(stage["solution"].lambda_n, self.multi.matrices[f"LAMBDA_{suffix}"], atol=1e-7)
            np.testing.assert_allclose(stage["solution"].gap_g, self.multi.matrices[f"GAP_{suffix}"], atol=1e-7)
            np.testing.assert_allclose(stage["W_total"], self.multi.matrices[f"W_TOTAL_{suffix}"], atol=1e-12)

    def test_multi_part_cross_interface_blocks_are_retained(self) -> None:
        topo = topology_summary(self.multi)
        self.assertTrue(topo["has_serial_path"])
        self.assertTrue(topo["has_closed_or_parallel_path"])
        for sid in self.multi.stage_plan["stage_id"]:
            blocks = coupling_block_summary(self.multi, sid)
            cross = blocks[blocks["cross_interface"]]
            self.assertEqual(len(cross), 6)
            self.assertTrue(cross["coupled_flag"].all())

    def test_multi_part_kcp_projection_and_ledger_match_oracle(self) -> None:
        result = run_all_stages(self.multi)
        kcp = extract_kcp(self.multi, result)
        np.testing.assert_allclose(
            kcp["predicted_value"].to_numpy(float),
            self.multi.matrices["PREDICTED_KCP_MULTIPART"],
            atol=1e-9,
        )
        ledger = contribution_ledger_summary(self.multi)
        self.assertEqual(ledger["status"], "PASS")
        self.assertEqual(ledger["duplicate_count"], 0)

    def test_release_increment_only_acts_on_release_stage(self) -> None:
        settings = NumericalSubstitutionSettings(enabled=True, mode="replace")
        clamp = release_rebound_increment(self.legacy, "S_CLAMP_02", settings)
        release = release_rebound_increment(self.legacy, "S_RELEASE_04", settings)
        np.testing.assert_allclose(clamp, np.zeros_like(clamp))
        self.assertEqual(release.shape, clamp.shape)

    def test_extended_lcp_matrix_is_symmetric_and_dimensionally_closed(self) -> None:
        sid = "S_CLAMP_02"
        q, W, *_ = build_stage_vectors(self.legacy, sid)
        model = build_extended_lcp_model(
            self.legacy, sid, q, W, OverConstraintSettings(enabled=True)
        )
        self.assertEqual(model["W_ext"].shape[0], len(model["q_ext"]))
        np.testing.assert_allclose(model["W_ext"], model["W_ext"].T)

    def test_physical_report_and_fallback_are_generated(self) -> None:
        result = run_all_stages(self.legacy)
        kcp = extract_kcp(self.legacy, result)
        report = physical_consistency_report(self.legacy, result, kcp, None)
        self.assertIn(report["overall"]["overall_status"], {"PASS", "WARN", "FAIL"})
        fallback = evaluate_validity_and_fallback(self.legacy, result, FallbackSettings())
        self.assertIn("总体回退决策", set(fallback["check_item"]))

    def test_physical_report_separates_lcp_physics_from_contact_pattern(self) -> None:
        result = run_all_stages(self.v25)
        kcp = extract_kcp(self.v25, result)
        report = physical_consistency_report(self.v25, result, kcp, None)
        self.assertEqual(report["overall"]["physics_status"], "PASS")
        self.assertEqual(report["overall"]["contact_status"], "WARN")
        self.assertTrue((report["stage_summary"]["contact_state"] == "NO_CONTACT").all())

    def test_changed_runtime_configuration_marks_validation_as_reference_only(self) -> None:
        result = run_all_stages(self.legacy, sms_mapping_settings=SMSMappingSettings(enabled=True))
        kcp = extract_kcp(self.legacy, result)
        report = physical_consistency_report(
            self.legacy,
            result,
            kcp,
            self.legacy.validation_kcp,
            validation_comparable=False,
            validation_context="SMS实时重建改变了验证配置。",
        )
        self.assertEqual(report["overall"]["kcp_validation_status"], "REFERENCE_ONLY")
        self.assertNotEqual(report["overall"]["overall_status"], "FAIL")
        self.assertIn("tolerance_status", report["kcp_anomalies"].columns)
        self.assertIn("validation_status", report["kcp_anomalies"].columns)

    def test_extended_lcp_physical_check_uses_full_equilibrium(self) -> None:
        result = run_all_stages(
            self.legacy,
            overconstraint_settings=OverConstraintSettings(enabled=True),
        )
        kcp = extract_kcp(self.legacy, result)
        report = physical_consistency_report(self.legacy, result, kcp, None)
        eq = report["check_details"]
        eq = eq[eq["check_item"] == "equilibrium_reconstruction"]
        self.assertTrue((eq["status"] == "PASS").all(), eq.to_string(index=False))

    def test_monte_carlo_is_reproducible_and_keeps_each_sample_kcp(self) -> None:
        defaults = distribution_defaults(self.legacy)
        a, stats_a = run_monte_carlo(self.legacy, 5, 20260713, defaults)
        b, stats_b = run_monte_carlo(self.legacy, 5, 20260713, defaults)
        self.assertEqual(len(a), 5)
        self.assertTrue(any(c.startswith("KCP_") for c in a.columns))
        self.assertTrue((a["status"] == "PASS").all())
        self.assertTrue(a.equals(b))
        self.assertTrue(stats_a.equals(stats_b))

    def test_validation_rows_are_independent_validation_role(self) -> None:
        roles = set(self.legacy.validation_kcp["data_role"].dropna().astype(str))
        self.assertEqual(roles, {"VALIDATE"})

    def test_ui_has_one_guard_for_each_main_page(self) -> None:
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tab_block = re.search(r"TABS\s*=\s*\[(.*?)\]\s*active_page", source, re.S)
        self.assertIsNotNone(tab_block)
        tab_count = len(re.findall(r"^\s*['\"]", tab_block.group(1), re.M))
        guard_count = len(re.findall(r"^if active_page == TABS\[\d+\]:", source, re.M))
        self.assertEqual(tab_count, 12)
        self.assertEqual(guard_count, tab_count)


class V6MVPAcceptanceTests(unittest.TestCase):
    """Executable acceptance-test placeholders for gaps confirmed by the V5.4 audit."""

    @unittest.skip("V6-MVP: implement true state recursion and connection-lock inheritance")
    def test_stage_state_is_inherited_between_locate_clamp_join_release(self) -> None:
        pass

    @unittest.skip("V6-MVP: generate W_struct from full-order K and DOF partitions")
    def test_schur_condensation_matches_direct_static_solution(self) -> None:
        pass

    @unittest.skip("V6-MVP: execute coordinate transforms and unit normalization, not header checks only")
    def test_coordinate_chain_maps_measurements_to_assembly_frame(self) -> None:
        pass

    @unittest.skip("V6-MVP: replace local projection with fully coupled normal-tangential NCP")
    def test_coupled_ncp_equilibrium_and_friction_residuals(self) -> None:
        pass

    @unittest.skip("V6-MVP: implement JSS/J-T and contribution ledger computation")
    def test_kcp_contributions_sum_to_total_without_double_counting(self) -> None:
        pass

    @unittest.skip("V6-MVP: compute MAE/RMSE/max error from independent validation sets")
    def test_validation_metrics_match_reference_calculation(self) -> None:
        pass

    @unittest.skip("V6-MVP: add contact-domain discretization convergence test")
    def test_contact_domain_refinement_converges_kcp_and_total_force(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
