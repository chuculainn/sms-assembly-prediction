from __future__ import annotations

from pathlib import Path
import argparse
import sys

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


def main() -> None:
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


if __name__ == '__main__':
    main()
