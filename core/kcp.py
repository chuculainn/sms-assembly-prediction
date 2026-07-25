from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .schema_adapter import to_vector


def _stage_or_default(result: dict[str, dict], requested: object | None, default: str = 'S_RELEASE_04') -> str:
    sid = str(requested) if requested is not None and pd.notna(requested) else default
    if sid in result:
        return sid
    matching = [key for key, value in result.items() if str(value.get('stage_id', '')) == sid]
    if matching:
        return matching[-1]
    if default in result:
        return default
    default_matching = [key for key, value in result.items() if str(value.get('stage_id', '')) == default]
    if default_matching:
        return default_matching[-1]
    return list(result.keys())[-1]


def _target_contact_index(cp: pd.DataFrame, definition: pd.Series) -> int | None:
    """Return nearest contact-point index for V2.5 KCP definitions.

    V2.5 stores target locations either as nominal_location="[x,y,z]" or as
    legacy-compatible position_x/position_y/position_z columns.  When a target
    location is available, using the nearest candidate point keeps converted E1
    packages numerically comparable with the old E1 center-gap/springback KCPs.
    """
    if cp.empty:
        return None
    loc = None
    if 'nominal_location' in definition.index and pd.notna(definition.get('nominal_location')):
        arr = to_vector(definition.get('nominal_location'), length=3, default=np.nan)
        if arr.size == 3 and np.all(np.isfinite(arr)):
            loc = arr
    if loc is None and {'position_x', 'position_y'} <= set(definition.index):
        try:
            x = float(definition.get('position_x'))
            y = float(definition.get('position_y'))
            z = float(definition.get('position_z', 0.0)) if pd.notna(definition.get('position_z', 0.0)) else 0.0
            loc = np.array([x, y, z], dtype=float)
        except Exception:
            loc = None
    if loc is None:
        return None
    coords = np.column_stack([
        pd.to_numeric(cp.get('x_i0', pd.Series(np.zeros(len(cp)))), errors='coerce').fillna(0.0).to_numpy(float),
        pd.to_numeric(cp.get('y_i0', pd.Series(np.zeros(len(cp)))), errors='coerce').fillna(0.0).to_numpy(float),
        pd.to_numeric(cp.get('z_i0', pd.Series(np.zeros(len(cp)))), errors='coerce').fillna(0.0).to_numpy(float),
    ])
    return int(np.argmin(np.sum((coords - loc.reshape(1, 3)) ** 2, axis=1)))


def _predict_generic_kcp(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    defs = pkg.kcp_kcm[pkg.kcp_kcm.get('feature_role', '') == 'KCP'].copy()
    if defs.empty:
        return pd.DataFrame(columns=['kcp_id', 'feature_type', 'stage_id', 'predicted_value', 'unit'])
    cp = pkg.contact_points.copy()
    rows = []
    for _, d in defs.iterrows():
        kcp_id = str(d.get('feature_id', d.get('kcp_id', 'KCP_UNKNOWN')))
        ftype = str(d.get('feature_type', 'gap')).lower()
        sid = _stage_or_default(result, d.get('stage_id_optional', d.get('stage_id', None)))
        gap = np.asarray(result[sid]['solution'].gap_g, dtype=float)
        pressure = np.asarray(result[sid]['pressure'], dtype=float)
        unit = d.get('expected_unit', d.get('unit', 'mm'))
        target_idx = _target_contact_index(cp, d)
        if ftype in {'gap', 'point_deviation', 'gap_center', 'center_gap'}:
            value = float(gap[target_idx]) if target_idx is not None and gap.size else (float(gap.mean()) if gap.size else np.nan)
        elif ftype in {'pressure', 'max_pressure'}:
            value = float(pressure.max(initial=0.0))
            unit = unit if pd.notna(unit) else 'MPa'
        elif ftype in {'pressure_at_point'}:
            value = float(pressure[target_idx]) if target_idx is not None and pressure.size else (float(pressure.max(initial=0.0)) if pressure.size else np.nan)
            unit = unit if pd.notna(unit) else 'MPa'
        elif ftype in {'contact_ratio', 'active_ratio'}:
            value = float(len(result[sid]['solution'].active_indices) / len(gap)) if len(gap) else np.nan
            unit = '1'
        elif ftype in {'springback', 'release_rebound'}:
            join_sid = _stage_or_default(result, 'S_JOIN_03', default=list(result.keys())[0])
            join_gap = np.asarray(result[join_sid]['solution'].gap_g, dtype=float)
            if target_idx is not None and gap.size and join_gap.size:
                value = float(gap[target_idx] - join_gap[target_idx])
            else:
                value = float(gap.mean() - join_gap.mean()) if gap.size and join_gap.size else np.nan
        elif ftype in {'step', 'step_proxy'}:
            if 'y_i0' in cp.columns and len(gap):
                y = cp['y_i0'].to_numpy(dtype=float)
                top = np.where(y == y.max())[0]
                bot = np.where(y == y.min())[0]
                value = float(gap[top].mean() - gap[bot].mean())
            else:
                value = 0.0
        else:
            value = float(gap[target_idx]) if target_idx is not None and gap.size else (float(gap.mean()) if gap.size else np.nan)
        rows.append({
            'kcp_id': kcp_id,
            'feature_type': d.get('feature_type', ftype),
            'stage_id': sid,
            'predicted_value': value,
            'unit': unit,
            'nominal_value': d.get('nominal_value', np.nan),
            'lower_tol': d.get('lower_tol', np.nan),
            'upper_tol': d.get('upper_tol', np.nan),
            'description': d.get('description', d.get('kcp_name', '')),
        })
    return pd.DataFrame(rows)


def _component_vector(row: pd.Series, name: str, n: int) -> np.ndarray:
    if name not in row.index:
        return np.zeros(n, dtype=float)
    return to_vector(row.get(name), length=n, default=0.0)


def _predict_multi_part_kcp(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    """Project the coupled global interface response into KCP space.

    The multi-part fixture defines J_INTERFACE_ALL and a contribution ledger.
    At runtime the package baseline SMS/process terms are retained while the
    contact contribution is recomputed from Cn @ lambda for the selected
    stage.  This reproduces the fixture oracle at unit scales and remains
    responsive to SMS/closure/Cn changes.
    """
    defs = pkg.kcp_kcm[pkg.kcp_kcm.get('feature_role', '') == 'KCP'].copy()
    if defs.empty:
        return pd.DataFrame(columns=['kcp_id', 'feature_type', 'stage_id', 'predicted_value', 'unit'])
    prediction = pkg.raw_tables.get('prediction/kcp_prediction_result.csv', pd.DataFrame())
    n = len(defs)
    ids = defs['feature_id'].astype(str).tolist()
    pred_row = prediction.iloc[0] if not prediction.empty else pd.Series(dtype=object)
    declared_ids = parse_ids = []
    if not prediction.empty:
        raw_ids = str(pred_row.get('kcp_ids', ''))
        parse_ids = [x for x in raw_ids.split(';') if x]
        declared_ids = parse_ids if len(parse_ids) == n else ids
    else:
        declared_ids = ids
    order = {kcp_id: i for i, kcp_id in enumerate(declared_ids)}

    J = np.asarray(pkg.matrices.get('J_INTERFACE_ALL', np.empty((0, 0))), dtype=float)
    sms = _component_vector(pred_row, 'sms_contribution', n)
    process = _component_vector(pred_row, 'process_contribution', n)
    rebound = _component_vector(pred_row, 'rebound_contribution', n)
    slip = _component_vector(pred_row, 'slip_contribution_optional', n)
    measurement = _component_vector(pred_row, 'measurement_uncertainty_contribution_optional', n)
    runtime_sms_scale = float(next(iter(result.values())).get('runtime_scales', {}).get('sms_scale', 1.0))

    rows = []
    for def_pos, (_, definition) in enumerate(defs.iterrows()):
        kcp_id = str(definition.get('feature_id', definition.get('kcp_id', f'KCP_{def_pos + 1}')))
        proj_pos = order.get(kcp_id, def_pos)
        sid = _stage_or_default(result, definition.get('stage_id_optional', definition.get('stage_id', None)))
        local_compression = np.asarray(result[sid]['local_compression'], dtype=float)
        if J.shape == (n, local_compression.size):
            contact = float(J[proj_pos] @ local_compression)
        else:
            contact = 0.0
        base_sms = float(sms[proj_pos]) if proj_pos < sms.size else 0.0
        other = sum(float(v[proj_pos]) if proj_pos < v.size else 0.0 for v in (process, rebound, slip, measurement))
        value = runtime_sms_scale * base_sms + contact + other
        rows.append({
            'kcp_id': kcp_id,
            'feature_type': definition.get('feature_type', 'projection'),
            'stage_id': sid,
            'predicted_value': value,
            'unit': definition.get('expected_unit', 'mm'),
            'nominal_value': definition.get('nominal_value', np.nan),
            'lower_tol': definition.get('lower_tol', np.nan),
            'upper_tol': definition.get('upper_tol', np.nan),
            'description': definition.get('description', definition.get('kcp_name', '')),
            'sms_contribution': runtime_sms_scale * base_sms,
            'contact_contribution': contact,
            'other_contribution': other,
            'projection_matrix_id': definition.get('extraction_matrix_id', 'J_INTERFACE_ALL'),
        })
    return pd.DataFrame(rows)


def extract_kcp(pkg: SMSPackage, result: dict[str, dict]) -> pd.DataFrame:
    if getattr(pkg, 'package_type', '') == 'V25_MULTI_PART':
        return _predict_multi_part_kcp(pkg, result)
    if getattr(pkg, 'package_type', 'E1_LEGACY').startswith('V25'):
        return _predict_generic_kcp(pkg, result)

    cp = pkg.contact_points
    x = cp['x_i0'].to_numpy(dtype=float)
    y = cp['y_i0'].to_numpy(dtype=float)
    center_idx = int(np.argmin((x - 200.0) ** 2 + y ** 2))
    top_idx = np.where(y == y.max())[0]
    bot_idx = np.where(y == y.min())[0]

    join_stage = 'S_JOIN_03'
    release_stage = 'S_RELEASE_04'
    join_gap = result[join_stage]['solution'].gap_g
    release_gap = result[release_stage]['solution'].gap_g
    join_pressure = result[join_stage]['pressure']
    release_active_ratio = len(result[release_stage]['solution'].active_indices) / len(release_gap)

    rows = [
        ('KCP_GAP_CENTER_RELEASE', 'gap', release_stage, release_gap[center_idx], 'mm'),
        ('KCP_MAX_PRESSURE_JOIN', 'pressure', join_stage, float(join_pressure.max(initial=0.0)), 'MPa'),
        ('KCP_ACTIVE_RATIO_RELEASE', 'contact_ratio', release_stage, release_active_ratio, '1'),
        ('KCP_SPRINGBACK_CENTER', 'springback', release_stage, release_gap[center_idx] - join_gap[center_idx], 'mm'),
        ('KCP_EDGE_STEP_RELEASE', 'step_proxy', release_stage, release_gap[top_idx].mean() - release_gap[bot_idx].mean(), 'mm'),
    ]
    df = pd.DataFrame(rows, columns=['kcp_id', 'feature_type', 'stage_id', 'predicted_value', 'unit'])

    if {'feature_role', 'feature_id'} <= set(pkg.kcp_kcm.columns):
        defs = pkg.kcp_kcm[pkg.kcp_kcm['feature_role'] == 'KCP'][['feature_id', 'nominal_value', 'lower_tol', 'upper_tol', 'description']].rename(columns={'feature_id': 'kcp_id'})
        df = df.merge(defs, on='kcp_id', how='left')
    return df


def compare_validation(kcp: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    val_cols = [c for c in ['kcp_id', 'measured_value', 'uncertainty', 'data_role'] if c in validation.columns]
    if 'kcp_id' not in val_cols:
        out = kcp.copy()
        out['measured_value'] = np.nan
        out['uncertainty'] = np.nan
        out['data_role'] = 'VALIDATE'
    else:
        out = kcp.merge(validation[val_cols], on='kcp_id', how='left')
    out['measured_value'] = pd.to_numeric(out.get('measured_value'), errors='coerce')
    out['uncertainty'] = pd.to_numeric(out.get('uncertainty'), errors='coerce').fillna(0.0)
    out['error'] = out['predicted_value'] - out['measured_value']
    out['abs_error'] = out['error'].abs()
    out['within_uncertainty_2sigma'] = out['abs_error'] <= 2 * out['uncertainty']
    return out
