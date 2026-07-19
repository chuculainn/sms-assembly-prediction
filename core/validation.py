from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import SMSPackage, get_stage_ids


def _row(name: str, status: str, detail: str) -> dict[str, str]:
    return {'check_item': name, 'status': status, 'detail': detail}


def _required_layout_checks(pkg: SMSPackage) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if getattr(pkg, 'package_type', 'E1_LEGACY').startswith('V25'):
        dirs = ['I0', 'I_Gamma', 'I_stage', 'I_key', 'I_meas', 'I_red', 'I_stat', 'sms_update', 'parameter_library', 'solver', 'prediction', 'validation', 'matrices']
        files = [
            'package_manifest.json', 'object_file_map.csv', 'field_dictionary.csv',
            'I0/part.csv', 'I0/interface.csv', 'I0/assembly_topology.csv',
            'I_Gamma/contact_domain.csv', 'I_Gamma/contact_point.csv', 'I_Gamma/gap_field.csv',
            'I_stage/stage_input.csv', 'I_stage/reduced_force_vector.csv',
            'I_key/kcp_definition.csv', 'I_key/kcm_definition.csv',
            'I_meas/measurement_record.csv', 'I_red/condensed_operator.csv',
            'parameter_library/interface_parameter.csv',
            'solver/interface_lcp_model.csv', 'solver/lcp_solution.csv', 'solver/contact_computation_trace.csv',
            'prediction/kcp_prediction_result.csv', 'validation/validation_result.csv',
            'matrices/matrix_manifest.csv',
        ]
        if pkg.package_type == 'V25_MULTI_PART':
            files += [
                'I0/subassembly.csv', 'I0/subassembly_membership.csv', 'I0/joint_definition.csv',
                'I0/stage_definition.csv', 'I_stage/part_stage_state.csv',
                'I_stage/interface_stage_state.csv', 'I_stage/stage_transition_record.csv',
                'prediction/deformation_contribution_ledger.csv',
                'prediction/double_count_check_result.csv', 'matrices/vector_layout.csv',
                'matrices/multi_part_matrices.npz',
            ]
            raw_keys = ['QA_ALL', 'CN_ALL', 'J_INTERFACE_ALL']
            for sid in get_stage_ids(pkg):
                suffix = sid.removeprefix('S_')
                raw_keys += [f'W_STRUCT_{suffix}', f'W_TOTAL_{suffix}', f'Q_{suffix}']
        else:
            files += ['matrices/default_matrices.npz']
            raw_keys = ['G_GAP_DEFAULT', 'BN_DEFAULT', 'QA_MATRIX_DEFAULT', 'CN_DEFAULT']
            for sid in get_stage_ids(pkg):
                raw_keys += [f'W_STRUCT_{sid}', f'Q_{sid}']
    else:
        dirs = ['I0', 'I_Gamma', 'I_stage', 'I_key', 'I_meas', 'I_red', 'I_pred', 'I_stat', 'validation', 'matrices']
        files = [
            'package_manifest.json', 'I0/part_table.csv', 'I0/interface_table.csv', 'I0/assembly_sequence.csv',
            'I_Gamma/contact_points.csv', 'I_Gamma/gap_field.csv', 'I_Gamma/interface_parameter.csv',
            'I_stage/stage_plan.csv', 'I_stage/process_record.csv', 'I_key/KCP_KCM_list.csv',
            'I_red/condensed_operator.csv', 'validation/validation_kcp.csv', 'matrices/E1_matrices.npz',
        ]
        raw_keys = ['g0', 'nominal_gap', 'sms_component', 'QA', 'Cn_local']
        for sid in get_stage_ids(pkg):
            raw_keys += [f'W_struct__{sid}', f'q__{sid}']
    for d in dirs:
        ok = (pkg.root / d).is_dir()
        rows.append(_row(f'required dir:{d}', 'PASS' if ok else 'FAIL', '存在' if ok else '缺失'))
    for f in files:
        ok = (pkg.root / f).exists()
        rows.append(_row(f'required file:{f}', 'PASS' if ok else 'FAIL', '存在' if ok else '缺失'))
    overview = getattr(pkg, 'data_overview', pd.DataFrame())
    if isinstance(overview, pd.DataFrame) and not overview.empty:
        csv_rows = overview[overview['kind'].eq('CSV')]
        header_fail = csv_rows[csv_rows['fields'].fillna('').astype(str).str.len().eq(0)]
        rows.append(_row('CSV headers', 'PASS' if header_fail.empty else 'FAIL', f'CSV={len(csv_rows)}, header_fail={len(header_fail)}'))
    missing_keys = [k for k in raw_keys if k not in pkg.matrices]
    rows.append(_row('NPZ required matrix keys', 'PASS' if not missing_keys else 'FAIL', '完整' if not missing_keys else f'缺失：{missing_keys}'))
    return rows


def _multi_part_checks(pkg: SMSPackage) -> list[dict[str, str]]:
    if pkg.package_type != 'V25_MULTI_PART':
        return []
    from .multi_part import contribution_ledger_summary, coupling_block_summary, topology_summary, vector_layout

    rows: list[dict[str, str]] = []
    topo = topology_summary(pkg)
    rows.append(_row(
        'multi-part topology connected',
        'PASS' if topo['connected'] else 'FAIL',
        f"parts={topo['part_count']}, interfaces={topo['interface_count']}, components={topo['connected_components']}",
    ))
    rows.append(_row(
        'serial path present', 'PASS' if topo['has_serial_path'] else 'FAIL',
        f"max_part_degree={topo['max_part_degree']}",
    ))
    rows.append(_row(
        'parallel/closed path present', 'PASS' if topo['has_closed_or_parallel_path'] else 'WARN',
        f"cycle_rank={topo['cycle_rank']}",
    ))
    layout = vector_layout(pkg)
    if layout.empty:
        layout_ok = False
    else:
        starts = layout['start_index'].astype(int).tolist()
        ends = layout['end_index'].astype(int).tolist()
        layout_ok = starts[0] == 0 and ends[-1] == len(pkg.contact_points) - 1 and all(starts[i] == ends[i - 1] + 1 for i in range(1, len(starts)))
    rows.append(_row('global vector layout contiguous', 'PASS' if layout_ok else 'FAIL', f'blocks={len(layout)}, contact_points={len(pkg.contact_points)}'))
    for sid in get_stage_ids(pkg):
        coupling = coupling_block_summary(pkg, sid)
        cross = coupling[coupling['cross_interface']] if not coupling.empty else pd.DataFrame()
        coupled = bool(not cross.empty and cross['coupled_flag'].all())
        rows.append(_row(f'cross-interface W blocks:{sid}', 'PASS' if coupled else 'WARN', f'cross_blocks={len(cross)}, nonzero={int(cross.get("coupled_flag", pd.Series(dtype=bool)).sum())}; zero blocks require topology/physics review but are not universally invalid'))
        W_total = pkg.matrices.get(f'W_total__{sid}')
        W_struct = pkg.matrices.get(f'W_struct__{sid}')
        relation = W_total is not None and W_struct is not None and np.allclose(W_total, W_struct + pkg.matrices['Cn_local'], atol=1e-12)
        rows.append(_row(f'W_total=W_struct+Cn:{sid}', 'PASS' if relation else 'FAIL', 'global coupled operator relation'))
    ledger = contribution_ledger_summary(pkg)
    rows.append(_row(
        'multi-path contribution ledger', str(ledger.get('status', 'WARN')),
        f"records={ledger.get('record_count', 0)}, duplicates={ledger.get('duplicate_count', 0)}, reconstruction_error={ledger.get('reconstruction_error', np.nan):.3e}",
    ))
    return rows

def validate_package(pkg: SMSPackage) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    checks.extend(_required_layout_checks(pkg))
    checks.extend(_multi_part_checks(pkg))
    m = len(pkg.contact_points)

    required = {
        'parts': len(pkg.parts) > 0,
        'interfaces': len(pkg.interfaces) > 0,
        'contact_points': m > 0,
        'gap_field': len(pkg.gap_field) == m,
        'stage_plan': len(pkg.stage_plan) > 0,
        'kcp_kcm': len(pkg.kcp_kcm) > 0,
        'validation_kcp': len(pkg.validation_kcp) > 0,
    }
    for key, ok in required.items():
        checks.append(_row(key, 'PASS' if ok else 'FAIL', '存在且非空' if ok else '缺失或为空'))

    for key in ['g0', 'nominal_gap', 'sms_component', 'QA', 'Cn_local', 'Bn_mapping']:
        ok = key in pkg.matrices
        checks.append(_row(f'matrix:{key}', 'PASS' if ok else 'FAIL', '已加载' if ok else 'NPZ 中缺失'))

    if 'g0' in pkg.matrices:
        checks.append(_row('g0 dimension', 'PASS' if pkg.matrices['g0'].shape == (m,) else 'FAIL', f"g0 shape={pkg.matrices['g0'].shape}, contact_points={m}"))
    if 'QA' in pkg.matrices:
        checks.append(_row('QA dimension', 'PASS' if pkg.matrices['QA'].shape == (m, m) else 'FAIL', f"QA shape={pkg.matrices['QA'].shape}, expected={(m, m)}"))
    if 'Cn_local' in pkg.matrices:
        Cn = pkg.matrices['Cn_local']
        symmetric = np.allclose(Cn, Cn.T)
        diag_nonnegative = bool(np.all(np.diag(Cn) >= 0))
        checks.append(_row('Cn_local symmetric / nonnegative', 'PASS' if symmetric and diag_nonnegative else 'FAIL', f'symmetric={symmetric}, diag_nonnegative={diag_nonnegative}'))

    for sid in get_stage_ids(pkg):
        needed = [f'W_struct__{sid}', f'u_free__{sid}', f'q__{sid}']
        for key in needed:
            checks.append(_row(f'stage matrix:{key}', 'PASS' if key in pkg.matrices else 'FAIL', '已加载' if key in pkg.matrices else '缺失'))
        if f'W_struct__{sid}' in pkg.matrices:
            W = pkg.matrices[f'W_struct__{sid}']
            symmetric = np.allclose(W, W.T, atol=1e-10)
            eig_min = float(np.linalg.eigvalsh(W).min())
            checks.append(_row(f'W_struct PSD:{sid}', 'PASS' if symmetric and eig_min > -1e-10 else 'FAIL', f'symmetric={symmetric}, min_eig={eig_min:.3e}'))
        q_key = f'q__{sid}'
        W_key = f'W_struct__{sid}'
        if q_key in pkg.matrices and W_key in pkg.matrices and 'Cn_local' in pkg.matrices and 'QA' in pkg.matrices:
            q_ok = pkg.matrices[q_key].shape == (m,)
            W_ok = pkg.matrices[W_key].shape == (m, m)
            C_ok = pkg.matrices['Cn_local'].shape == (m, m)
            QA_ok = pkg.matrices['QA'].shape == (m, m)
            checks.append(_row(f'minimal input dimensions:{sid}', 'PASS' if q_ok and W_ok and C_ok and QA_ok else 'FAIL', f'q={pkg.matrices[q_key].shape}, W={pkg.matrices[W_key].shape}, Cn={pkg.matrices["Cn_local"].shape}, QA={pkg.matrices["QA"].shape}, expected={m}'))


    # Optional numerical substitution module package. These files are not mandatory for the
    # old data chain, but when present they indicate that the software can assemble Cn_local
    # from numerical replacement modules rather than only reading a fixed matrix.
    sub_dir = pkg.root / 'I_substitution'
    sub_files = [
        'numerical_substitution_config.json', 'partition_cn.csv', 'layer_stack.csv',
        'rough_contact.csv', 'local_indent.csv', 'fixture_joint_compliance.csv',
        'release_rebound.csv', 'module_validity.csv',
    ]
    if sub_dir.exists():
        missing = [f for f in sub_files if not (sub_dir / f).exists()]
        checks.append(_row('I_substitution module files', 'PASS' if not missing else 'WARN', '完整' if not missing else f'缺失：{missing}'))
    else:
        checks.append(_row('I_substitution module files', 'WARN', '未提供数值替代模块，使用原始 Cn_local'))

    # Optional v5 modules: SMS mapping, tangential NCP, extended LCP and fallback validity.
    sms_path = pkg.root / 'I_meas' / 'sms_point_or_node.csv'
    v25_meas = pkg.root / 'I_meas' / 'measurement_record.csv'
    v25_sms_field = pkg.root / 'sms_update' / 'sms_field.csv'
    if sms_path.exists():
        sms_status, sms_detail = 'PASS', '旧E1 SMS点可实时WLS/MAP重建g0'
    elif v25_meas.exists() or v25_sms_field.exists():
        sms_status, sms_detail = 'PASS', 'V2.5 MeasurementRecord/SMSField可适配；原始点不足时安全使用包内冻结SMS/g0'
    else:
        sms_status, sms_detail = 'WARN', '缺少SMS点和V2.5 SMSField；高级SMS映射将使用包内冻结g0'
    checks.append(_row('S03/S04 SMS point mapping input', sms_status, sms_detail))

    d12_files = [
        pkg.root / 'I_stage' / 'overconstraint_locator_group.csv',
        pkg.root / 'I_stage' / 'locator_element.csv',
        pkg.root / 'I_stage' / 'locator_compliance.csv',
        pkg.root / 'I_pred' / 'overconstraint_contact_model.json',
        pkg.root / 'I_pred' / 'connection_lock_history.csv',
    ]
    present = [p.name for p in d12_files if p.exists()]
    checks.append(_row('S20 D12 extended LCP files', 'PASS' if len(present) >= 3 else 'WARN', f'found={present}'))

    tpath = pkg.root / 'I_stage' / 'tangential_free_slip.csv'
    checks.append(_row('S17 tangential NCP input', 'PASS' if tpath.exists() else 'WARN', '已提供tangential_free_slip.csv' if tpath.exists() else '缺少切向自由滑移输入，使用梯度近似'))

    validity_path = pkg.root / 'I_substitution' / 'module_validity.csv'
    checks.append(_row('S28 module validity / fallback', 'PASS' if validity_path.exists() else 'WARN', '已提供module_validity.csv' if validity_path.exists() else '缺少适用域，无法严格回退判断'))

    # data role check
    if 'data_role' in pkg.validation_kcp.columns:
        ok = set(pkg.validation_kcp['data_role'].dropna().unique()) <= {'VALIDATE'}
        checks.append(_row('validation data role', 'PASS' if ok else 'FAIL', f"roles={sorted(pkg.validation_kcp['data_role'].dropna().unique())}"))

    base = pd.DataFrame(checks)
    from .package_validator import validate_package_detailed
    detailed = validate_package_detailed(pkg)
    return pd.concat([base, detailed], ignore_index=True, sort=False)
