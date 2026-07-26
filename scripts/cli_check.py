from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_loader import load_package
from core.validation import validate_package
from core.stage_solver import run_all_stages, stage_summary_table
from core.kcp import extract_kcp, compare_validation
from core.monte_carlo import distribution_defaults, run_monte_carlo
from core.numerical_substitution import NumericalSubstitutionSettings
from core.sms_mapping import SMSMappingSettings, rebuild_gap_from_sms
from core.overconstraint import OverConstraintSettings, extended_solution_table
from core.tangential_ncp import TangentialNCPSettings, tangential_summary_table
from core.fallback import FallbackSettings, evaluate_validity_and_fallback, remaining_limitations_table
from core.physical_consistency import physical_consistency_report
from core.stage_measurement_update import validate_measurement_update_package


def _blocking_fail_count(checks) -> int:
    if checks.empty:
        return 0
    blocking = checks.get("blocking")
    if blocking is None:
        blocking = checks["status"].astype(str).eq("FAIL")
    else:
        blocking = blocking.astype("boolean").fillna(False)
    return int((checks["status"].astype(str).eq("FAIL") & blocking).sum())


def _emit_final_summary(
    final_status: str,
    blocking: int,
    physical: int,
    *,
    checkpoint_count: int = 0,
    update_attempt_count: int = 0,
    posterior_accepted_count: int = 0,
    posterior_rollback_count: int = 0,
    measurement_update_fail_count: int = 0,
    physical_residual_gate: str = "NOT_APPLICABLE",
) -> None:
    print(f"FINAL_STATUS={final_status}")
    print(f"BLOCKING_FAIL_COUNT={blocking}")
    print(f"PHYSICAL_FAIL_COUNT={physical}")
    print(f"MEASUREMENT_CHECKPOINT_COUNT={checkpoint_count}")
    print(f"MEASUREMENT_UPDATE_ATTEMPT_COUNT={update_attempt_count}")
    print(f"POSTERIOR_ACCEPTED_COUNT={posterior_accepted_count}")
    print(f"POSTERIOR_ROLLBACK_COUNT={posterior_rollback_count}")
    print(f"MEASUREMENT_UPDATE_FAIL_COUNT={measurement_update_fail_count}")
    print(f"POSTERIOR_PHYSICAL_RESIDUAL_GATE={physical_residual_gate}")


def _runtime_exit_code(
    physical_fail_count: int,
    fallback_status: str,
    overall_physical_status: str,
    measurement_update_fail_count: int = 0,
) -> tuple[str, int]:
    failed = (
        int(physical_fail_count) > 0
        or int(measurement_update_fail_count) > 0
        or str(fallback_status).upper() == "FAIL"
        or str(overall_physical_status).upper() == "FAIL"
    )
    return ("FAIL", 2) if failed else ("PASS", 0)


def _package_validator_exit(root: Path) -> int:
    validator = root / "validation" / "validate_package.py"
    if not validator.exists():
        return 0
    completed = subprocess.run(
        [sys.executable, str(validator), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )
    print("\n=== Package-local validation ===")
    print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description='SMS E1 数据包命令行检查')
    parser.add_argument('data_dir', nargs='?', default=str(ROOT / 'data' / 'E1_min_closed_loop'), help='标准输入包目录')
    parser.add_argument('--mc', type=int, default=0, help='可选：运行 N 个 Monte Carlo 样本')
    parser.add_argument('--subst-mode', choices=['base_only', 'replace', 'add'], default='base_only', help='数值替代Cn装配方式')
    parser.add_argument('--sms-rebuild', action='store_true', help='启用S03/S04：由SMS点实时WLS/MAP重建g0')
    parser.add_argument('--tangential', action='store_true', help='启用S17：Ct/mu切向摩擦投影')
    parser.add_argument('--extended-lcp', action='store_true', help='启用S20：N-2-1扩展LCP')
    args = parser.parse_args()

    pkg = load_package(args.data_dir)
    checks = validate_package(pkg)
    print('=== Package ===')
    print(pkg.root)
    print('\n=== Quality checks ===')
    print(checks.to_string(index=False))
    blocking_fail_count = _blocking_fail_count(checks)
    checkpoint_table = pkg.raw_tables.get(
        "I_meas/measurement_checkpoint.csv", pd.DataFrame()
    )
    checkpoint_count = int(len(checkpoint_table))
    measurement_checks = validate_measurement_update_package(pkg)
    if not measurement_checks.empty:
        print("\n=== Stage measurement update quality checks ===")
        print(measurement_checks.to_string(index=False))
        blocking_fail_count += _blocking_fail_count(measurement_checks)
    package_validator_exit = _package_validator_exit(pkg.root)
    if package_validator_exit != 0:
        blocking_fail_count += 1
    if blocking_fail_count:
        _emit_final_summary(
            "FAIL",
            blocking_fail_count,
            0,
            checkpoint_count=checkpoint_count,
        )
        return 1

    subst = NumericalSubstitutionSettings(enabled=args.subst_mode != 'base_only', mode=args.subst_mode)
    sms_settings = SMSMappingSettings(enabled=args.sms_rebuild)
    tangent_settings = TangentialNCPSettings(enabled=args.tangential)
    oc_settings = OverConstraintSettings(enabled=args.extended_lcp)
    result = run_all_stages(pkg, substitution_settings=subst, sms_mapping_settings=sms_settings, tangential_settings=tangent_settings, overconstraint_settings=oc_settings)
    summary = stage_summary_table(result)
    print('\n=== Stage summary ===')
    print(summary.to_string(index=False))

    if args.sms_rebuild:
        rebuilt = rebuild_gap_from_sms(pkg, sms_settings)
        print('\n=== SMS WLS/MAP summary ===')
        print(rebuilt['fit_summary'].to_string(index=False))
        print('\n=== SMS mapping quality ===')
        print(rebuilt['quality'].to_string(index=False))

    if args.extended_lcp:
        print('\n=== Extended LCP element solutions ===')
        for sid, res in result.items():
            ext = res.get('extended_lcp')
            if ext is not None and ext.get('enabled', False):
                print(f'--- {sid} ---')
                print(extended_solution_table(sid, ext).to_string(index=False))
                print('force_nonuniqueness:', res.get('force_nonuniqueness'))

    if args.tangential:
        print('\n=== Tangential NCP summary ===')
        for sid, res in result.items():
            tang = res.get('tangential_ncp')
            if tang is not None and not tang.empty:
                print(f'--- {sid} ---')
                print(tangential_summary_table(tang).to_string(index=False))

    kcp = extract_kcp(pkg, result)
    val = compare_validation(kcp, pkg.validation_kcp)
    print('\n=== KCP validation ===')
    print(val[['kcp_id', 'predicted_value', 'measured_value', 'abs_error', 'unit']].to_string(index=False))

    changed = []
    if args.subst_mode != 'base_only':
        changed.append('numerical substitution')
    if args.sms_rebuild:
        changed.append('SMS rebuild')
    if args.tangential:
        changed.append('tangential projection')
    if args.extended_lcp:
        changed.append('extended LCP')
    if any(
        bool(getattr(item.get("measurement_update"), "posterior_accepted", False))
        for item in result.values()
    ):
        changed.append('accepted stage measurement posterior update')
    phys = physical_consistency_report(
        pkg, result, kcp, val,
        validation_comparable=not changed,
        validation_context=(
            'package baseline configuration'
            if not changed else
            'changed runtime configuration: ' + ', '.join(changed) + '; validation values are reference-only'
        ),
    )
    print('\n=== Physical consistency overall ===')
    print(phys['overall'])
    print('\n=== Physical consistency stage summary ===')
    print(phys['stage_summary'].to_string(index=False))
    print('\n=== KCP anomaly hints ===')
    print(phys['kcp_anomalies'].to_string(index=False))

    fb = evaluate_validity_and_fallback(pkg, result, FallbackSettings(), rebuild_gap_from_sms(pkg, sms_settings)['quality'] if args.sms_rebuild else None)
    print('\n=== Fallback decision ===')
    print(fb.to_string(index=False))

    print('\n=== Remaining limitations ===')
    print(remaining_limitations_table().to_string(index=False))

    if args.mc > 0:
        defaults = distribution_defaults(pkg)
        samples, stats = run_monte_carlo(pkg, args.mc, 20260708, defaults, substitution_settings=subst, sms_mapping_settings=sms_settings, overconstraint_settings=oc_settings, tangential_settings=tangent_settings)
        print('\n=== Monte Carlo stats ===')
        print(stats.to_string(index=False))
        print('\nSamples:', len(samples))
    physical_fail_count = int(
        phys.get("stage_summary", pd.DataFrame())
        .get("physics_status", pd.Series(dtype=str))
        .astype(str)
        .eq("FAIL")
        .sum()
    )
    fallback_status = (
        str(fb.iloc[-1].get("status", "PASS")).upper()
        if not fb.empty else "PASS"
    )
    overall_physical_status = str(
        phys.get("overall", {}).get("overall_status", "PASS")
    ).upper()
    updates = [
        item.get("measurement_update")
        for item in result.values()
        if item.get("measurement_update") is not None
    ]
    update_attempt_count = len(updates)
    posterior_accepted_count = sum(
        bool(update.posterior_accepted) for update in updates
    )
    posterior_rollback_count = sum(
        update.rollback_record is not None for update in updates
    )
    measurement_update_fail_count = sum(
        str(update.quality_flag).upper() == "FAIL" for update in updates
    )
    physical_gate_values = [
        bool(update.trace.get("physical_residual_improved"))
        for update in updates
        if str(update.trace.get("status", "")).upper()
        in {"POSTERIOR_ACCEPTED", "POSTERIOR_REJECTED_ROLLBACK"}
        and "physical_residual_improved" in update.trace
    ]
    physical_residual_gate = (
        "PASS"
        if physical_gate_values and all(physical_gate_values)
        else "FAIL"
        if physical_gate_values
        else "NOT_APPLICABLE"
    )
    final_status, exit_code = _runtime_exit_code(
        physical_fail_count,
        fallback_status,
        overall_physical_status,
        measurement_update_fail_count,
    )
    _emit_final_summary(
        final_status,
        blocking_fail_count,
        physical_fail_count,
        checkpoint_count=checkpoint_count,
        update_attempt_count=update_attempt_count,
        posterior_accepted_count=posterior_accepted_count,
        posterior_rollback_count=posterior_rollback_count,
        measurement_update_fail_count=measurement_update_fail_count,
        physical_residual_gate=physical_residual_gate,
    )
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"UNEXPECTED_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)
        _emit_final_summary("FAIL", 0, 1)
        raise SystemExit(3)
