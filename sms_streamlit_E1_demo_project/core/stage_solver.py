from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import SMSPackage, get_stage_ids
from .lcp_solver import solve_lcp_active_set, LCPSolution
from .numerical_substitution import NumericalSubstitutionSettings, assemble_Cn_matrix, release_rebound_increment
from .sms_mapping import SMSMappingSettings, rebuild_gap_from_sms
from .overconstraint import OverConstraintSettings, solve_extended_lcp, force_nonuniqueness_report
from .tangential_ncp import TangentialNCPSettings, tangential_result_table


def _runtime_gap_components(
    pkg: SMSPackage,
    sms_scale: float,
    sms_mapping_settings: SMSMappingSettings | dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    sms_settings = SMSMappingSettings.from_dict(sms_mapping_settings) if not isinstance(sms_mapping_settings, SMSMappingSettings) else sms_mapping_settings
    if sms_settings is not None and sms_settings.enabled:
        rebuilt = rebuild_gap_from_sms(pkg, sms_settings)
        nominal = rebuilt['nominal_gap']
        sms_component = rebuilt['sms_component']
        pose = rebuilt['pose_component']
        return nominal, sms_component, pose, rebuilt
    nominal = pkg.matrices['nominal_gap']
    sms_component = pkg.matrices['sms_component']
    pose = pkg.matrices.get('pose_component', 0.0)
    return nominal, sms_component, pose, None


def build_stage_vectors(
    pkg: SMSPackage,
    stage_id: str,
    sms_scale: float = 1.0,
    closure_scale: float = 1.0,
    cn_scale: float = 1.0,
    substitution_settings: NumericalSubstitutionSettings | dict | None = None,
    sms_mapping_settings: SMSMappingSettings | dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict | None]:
    """Return q, W_total, W_struct, Cn, substitution_components, sms_rebuild.

    q = g0_scaled - u_free_scaled; W_total = W_struct + Cn_scaled.
    Optional SMSMappingSettings can rebuild the SMS component from I_meas/sms_point_or_node.csv.
    """
    nominal, sms_component, pose_component, sms_rebuild = _runtime_gap_components(pkg, sms_scale, sms_mapping_settings)
    g0_scaled = nominal + sms_scale * sms_component + pose_component
    u_free = closure_scale * pkg.matrices[f'u_free__{stage_id}']
    q = g0_scaled - u_free
    q = q + release_rebound_increment(pkg, stage_id, substitution_settings)
    W_struct = pkg.matrices[f'W_struct__{stage_id}']
    Cn, components = assemble_Cn_matrix(pkg, stage_id, cn_scale=cn_scale, settings=substitution_settings)
    W_total = W_struct + Cn
    return q, W_total, W_struct, Cn, components, sms_rebuild


def run_stage(
    pkg: SMSPackage,
    stage_id: str,
    sms_scale: float = 1.0,
    closure_scale: float = 1.0,
    cn_scale: float = 1.0,
    eps: float = 1e-9,
    substitution_settings: NumericalSubstitutionSettings | dict | None = None,
    sms_mapping_settings: SMSMappingSettings | dict | None = None,
    overconstraint_settings: OverConstraintSettings | dict | None = None,
    tangential_settings: TangentialNCPSettings | dict | None = None,
) -> dict:
    q, W_total, W_struct, Cn, components, sms_rebuild = build_stage_vectors(
        pkg, stage_id,
        sms_scale=sms_scale,
        closure_scale=closure_scale,
        cn_scale=cn_scale,
        substitution_settings=substitution_settings,
        sms_mapping_settings=sms_mapping_settings,
    )

    over_settings = OverConstraintSettings.from_dict(overconstraint_settings) if not isinstance(overconstraint_settings, OverConstraintSettings) else overconstraint_settings
    extended = None
    if over_settings is not None and over_settings.enabled:
        extended = solve_extended_lcp(pkg, stage_id, q, W_total, eps=eps, settings=over_settings)
        sol_ext: LCPSolution = extended['solution']
        lam_contact = extended['contact_lambda']
        gap_contact = extended['contact_gap']
        sol = LCPSolution(
            lambda_n=lam_contact,
            gap_g=gap_contact,
            active_indices=[i for i in sol_ext.active_indices if i < len(q)],
            inactive_indices=[i for i in range(len(q)) if i not in [j for j in sol_ext.active_indices if j < len(q)]],
            residuals=sol_ext.residuals,
            iteration_count=sol_ext.iteration_count,
            convergence_status=sol_ext.convergence_status,
            active_set_trace=sol_ext.active_set_trace,
        )
    else:
        sol = solve_lcp_active_set(q, W_total, eps=eps)

    area = pkg.contact_points['area_weight'].to_numpy(dtype=float)
    pressure = sol.lambda_n / area
    local_compression = Cn @ sol.lambda_n
    stage_name = pkg.stage_plan.loc[pkg.stage_plan['stage_id'] == stage_id, 'operation_type'].iloc[0]

    tangential_df = pd.DataFrame()
    tangential_settings_obj = TangentialNCPSettings.from_dict(tangential_settings) if not isinstance(tangential_settings, TangentialNCPSettings) else tangential_settings
    if tangential_settings_obj is not None and tangential_settings_obj.enabled:
        tangential_df = tangential_result_table(pkg, stage_id, sol.lambda_n, q, components, tangential_settings_obj)

    nonunique = None
    if extended is not None:
        nonunique = force_nonuniqueness_report(extended['W_ext'], extended['solution'].active_indices)

    return {
        'stage_id': stage_id,
        'stage_name': stage_name,
        'q': q,
        'W_total': W_total,
        'W_struct': W_struct,
        'Cn': Cn,
        'solution': sol,
        'pressure': pressure,
        'local_compression': local_compression,
        'substitution_components': components,
        'sms_rebuild': sms_rebuild,
        'extended_lcp': extended,
        'force_nonuniqueness': nonunique,
        'tangential_ncp': tangential_df,
        'runtime_scales': {
            'sms_scale': float(sms_scale),
            'closure_scale': float(closure_scale),
            'cn_scale': float(cn_scale),
        },
    }


def run_all_stages(
    pkg: SMSPackage,
    sms_scale: float = 1.0,
    closure_scale: float = 1.0,
    cn_scale: float = 1.0,
    eps: float = 1e-9,
    substitution_settings: NumericalSubstitutionSettings | dict | None = None,
    sms_mapping_settings: SMSMappingSettings | dict | None = None,
    overconstraint_settings: OverConstraintSettings | dict | None = None,
    tangential_settings: TangentialNCPSettings | dict | None = None,
) -> dict[str, dict]:
    return {
        sid: run_stage(
            pkg, sid,
            sms_scale=sms_scale,
            closure_scale=closure_scale,
            cn_scale=cn_scale,
            eps=eps,
            substitution_settings=substitution_settings,
            sms_mapping_settings=sms_mapping_settings,
            overconstraint_settings=overconstraint_settings,
            tangential_settings=tangential_settings,
        ) for sid in get_stage_ids(pkg)
    }


def point_result_table(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    base_cols = [
        c for c in ['candidate_id', 'contact_domain_id', 'interface_id', 'local_index', 'x_i0', 'y_i0', 'area_weight']
        if c in pkg.contact_points.columns
    ]
    base = pkg.contact_points[base_cols].copy()
    rows = []
    for sid, res in result.items():
        sol: LCPSolution = res['solution']
        df = base.copy()
        df['stage_id'] = sid
        df['stage_name'] = res['stage_name']
        df['q_free_gap_mm'] = res['q']
        df['gap_g_mm'] = sol.gap_g
        df['lambda_n_N'] = sol.lambda_n
        df['pressure_p_n_MPa'] = res['pressure']
        df['local_compression_w_n_mm'] = res['local_compression']
        df['active_flag'] = [int(i in sol.active_indices) for i in range(len(df))]
        if res.get('tangential_ncp') is not None and not res['tangential_ncp'].empty:
            t = res['tangential_ncp'][['candidate_id', 'lambda_t_norm_N', 'friction_limit_N', 'stick_slip_state']]
            df = df.merge(t, on='candidate_id', how='left')
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def stage_summary_table(result: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sid, res in result.items():
        sol: LCPSolution = res['solution']
        pressure = res['pressure']
        gap = sol.gap_g
        comp = res['local_compression']
        ext = res.get('extended_lcp')
        tang = res.get('tangential_ncp')
        rows.append({
            'stage_id': sid,
            'stage_name': res['stage_name'],
            'active_count': len(sol.active_indices),
            'active_ratio': len(sol.active_indices) / len(gap),
            'lambda_sum_N': float(sol.lambda_n.sum()),
            'lambda_max_N': float(np.max(sol.lambda_n)) if sol.lambda_n.size else 0.0,
            'pressure_max_MPa': float(np.max(pressure)) if pressure.size else 0.0,
            'gap_min_mm': float(np.min(gap)) if gap.size else np.nan,
            'gap_mean_mm': float(np.mean(gap)) if gap.size else np.nan,
            'compression_mean_mm': float(np.mean(comp)) if comp.size else np.nan,
            'iteration_count': sol.iteration_count,
            'convergence_status': sol.convergence_status,
            'extended_element_count': int(ext.get('extended_count', 0)) if ext is not None else 0,
            'extended_lambda_sum_N': float(np.sum(ext.get('extra_lambda', []))) if ext is not None else 0.0,
            'tangential_slip_count': int((tang['stick_slip_state'] == 'slip').sum()) if tang is not None and not tang.empty else 0,
            'tangential_max_lambda_N': float(tang['lambda_t_norm_N'].max()) if tang is not None and not tang.empty else 0.0,
            **sol.residuals,
        })
    return pd.DataFrame(rows)
