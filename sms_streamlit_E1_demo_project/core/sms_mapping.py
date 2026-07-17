from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import ast

import numpy as np
import pandas as pd

from .data_loader import SMSPackage


SMS_POINT_COLUMNS = [
    'part_id', 'x', 'y', 'z', 'normal_deviation', 'sigma',
    'quality_flag', 'support_state', 'coordinate_system_id', 'source_measurement_id',
]


@dataclass
class SMSMappingSettings:
    """Runtime SMS update / gap-mapping options.

    enabled=False keeps the package-provided GapField and matrices.
    method='wls_basis' fits a small low-order SMS basis for each part, then maps it to
    contact candidates. method='idw' uses inverse-distance interpolation directly.
    """

    enabled: bool = False
    method: str = 'wls_basis'  # package | wls_basis | idw
    ridge_lambda: float = 1e-6
    idw_power: float = 2.0
    nominal_gap_source: str = 'package'  # package | interface_table
    residual_warn_mm: float = 0.020
    residual_fail_mm: float = 0.050

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'SMSMappingSettings':
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _sms_points(pkg: SMSPackage) -> pd.DataFrame:
    """Return one canonical SMS point table for legacy E1 and formal V2.5 packages.

    Legacy E1 stores a dedicated sms_point_or_node.csv. Formal V2.5 stores measurements
    in MeasurementRecord. Only records explicitly targeting SMS are adapted here; process
    force/locator/KCP records must never be interpreted as surface deviations.
    """
    path = pkg.root / 'I_meas' / 'sms_point_or_node.csv'
    if path.exists():
        df = pd.read_csv(path)
        for col in SMS_POINT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df.attrs['sms_source'] = 'legacy:I_meas/sms_point_or_node.csv'
    else:
        v25_path = pkg.root / 'I_meas' / 'measurement_record.csv'
        if not v25_path.exists():
            out = pd.DataFrame(columns=SMS_POINT_COLUMNS)
            out.attrs['sms_source'] = 'none'
            return out
        raw = pd.read_csv(v25_path)
        if raw.empty:
            out = pd.DataFrame(columns=SMS_POINT_COLUMNS)
            out.attrs['sms_source'] = 'v25:measurement_record(empty)'
            return out
        update_target = raw.get('update_target', pd.Series('', index=raw.index)).astype(str).str.upper()
        data_role = raw.get('data_role', pd.Series('', index=raw.index)).astype(str).str.upper()
        unit = raw.get('unit', pd.Series('', index=raw.index)).astype(str).str.lower()
        part = raw.get('part_id_optional', raw.get('part_id', pd.Series(np.nan, index=raw.index)))
        mask = update_target.eq('SMS') & data_role.isin(['IDENTIFY', 'CALIBRATE']) & unit.isin(['mm', 'millimeter', 'millimetre']) & part.notna()
        raw = raw[mask].copy()
        rows = []
        for _, r in raw.iterrows():
            try:
                loc = ast.literal_eval(str(r.get('location', '[0,0,0]')))
                xyz = np.asarray(loc, dtype=float).reshape(-1)
                if xyz.size != 3 or not np.all(np.isfinite(xyz)):
                    continue
            except (ValueError, SyntaxError, TypeError):
                continue
            rows.append({
                'part_id': r.get('part_id_optional', r.get('part_id')),
                'x': float(xyz[0]), 'y': float(xyz[1]), 'z': float(xyz[2]),
                'normal_deviation': float(r.get('value', 0.0)),
                'sigma': float(r.get('standard_uncertainty', 0.006) if pd.notna(r.get('standard_uncertainty', np.nan)) else 0.006),
                'quality_flag': r.get('quality_flag', 'WARN'),
                'support_state': r.get('support_state', 'UNKNOWN'),
                'coordinate_system_id': r.get('coordinate_system_id', ''),
                'source_measurement_id': r.get('measurement_id', ''),
            })
        df = pd.DataFrame(rows, columns=SMS_POINT_COLUMNS)
        df.attrs['sms_source'] = 'v25:I_meas/measurement_record.csv'
    if 'quality_flag' in df.columns:
        df = df[df['quality_flag'].astype(str).str.upper().isin(['PASS', 'WARN', 'OUTLIER_REVIEWED'])]
    if 'support_state' in df.columns:
        df = df[df['support_state'].astype(str).str.upper().isin(['FREE', 'KNOWN_SUPPORT', ''])]
    return df.reset_index(drop=True)


def _basis_matrix(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0 = float(np.mean(x)) if x.size else 0.0
    y0 = float(np.mean(y)) if y.size else 0.0
    sx = float(np.ptp(x)) if np.ptp(x) > 1e-12 else 1.0
    sy = float(np.ptp(y)) if np.ptp(y) > 1e-12 else 1.0
    xn = (x - x0) / sx
    yn = (y - y0) / sy
    B = np.column_stack([
        np.ones_like(xn),
        xn,
        yn,
        xn * yn,
        xn ** 2 - yn ** 2,
    ])
    return B, {'x0': x0, 'y0': y0, 'sx': sx, 'sy': sy}


def _basis_eval(x: np.ndarray, y: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xn = (x - norm['x0']) / max(norm['sx'], 1e-12)
    yn = (y - norm['y0']) / max(norm['sy'], 1e-12)
    return np.column_stack([
        np.ones_like(xn),
        xn,
        yn,
        xn * yn,
        xn ** 2 - yn ** 2,
    ])


def _fit_part_wls(points: pd.DataFrame, ridge_lambda: float) -> dict[str, Any]:
    x = points['x'].to_numpy(float)
    y = points['y'].to_numpy(float)
    z = points['normal_deviation'].to_numpy(float)
    sigma = pd.to_numeric(points.get('sigma', pd.Series(np.ones(len(points)) * 0.006)), errors='coerce').fillna(0.006).to_numpy(float)
    sigma = np.maximum(sigma, 1e-9)
    B, norm = _basis_matrix(x, y)
    W = np.diag(1.0 / sigma ** 2)
    A = B.T @ W @ B + float(ridge_lambda) * np.eye(B.shape[1])
    b = B.T @ W @ z
    try:
        alpha = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, b, rcond=None)[0]
    fitted = B @ alpha
    residual = z - fitted
    cov = np.linalg.pinv(A)
    return {
        'alpha': alpha,
        'covariance': cov,
        'norm': norm,
        'rms_residual': float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0,
        'max_abs_residual': float(np.max(np.abs(residual))) if residual.size else 0.0,
        'n_points': int(len(points)),
        'condition_number': float(np.linalg.cond(A)) if A.size else np.nan,
    }


def fit_sms_wls_map(pkg: SMSPackage, settings: SMSMappingSettings | dict[str, Any] | None = None) -> pd.DataFrame:
    """Fit a low-order SMS basis for every part in sms_point_or_node.csv."""
    settings = SMSMappingSettings.from_dict(settings) if not isinstance(settings, SMSMappingSettings) else settings
    pts = _sms_points(pkg)
    rows = []
    if pts.empty:
        return pd.DataFrame(rows)
    for part_id, group in pts.groupby('part_id'):
        if len(group) < 3:
            rows.append({
                'part_id': part_id, 'method': 'wls_basis', 'n_points': len(group),
                'status': 'WARN', 'detail': '测点少于3个，无法稳定拟合低阶SMS基',
                'rms_residual_mm': np.nan, 'max_abs_residual_mm': np.nan,
                'condition_number': np.nan,
            })
            continue
        fit = _fit_part_wls(group, settings.ridge_lambda)
        status = 'PASS'
        if fit['rms_residual'] > settings.residual_fail_mm:
            status = 'FAIL'
        elif fit['rms_residual'] > settings.residual_warn_mm:
            status = 'WARN'
        row = {
            'part_id': part_id,
            'method': 'wls_basis',
            'n_points': fit['n_points'],
            'alpha_0_constant': float(fit['alpha'][0]),
            'alpha_1_x_tilt': float(fit['alpha'][1]),
            'alpha_2_y_tilt': float(fit['alpha'][2]),
            'alpha_3_xy_warp': float(fit['alpha'][3]),
            'alpha_4_quadratic': float(fit['alpha'][4]),
            'rms_residual_mm': fit['rms_residual'],
            'max_abs_residual_mm': fit['max_abs_residual'],
            'condition_number': fit['condition_number'],
            'status': status,
            'detail': 'SMS WLS/MAP 低阶基拟合结果',
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_idw(points: pd.DataFrame, xq: np.ndarray, yq: np.ndarray, power: float) -> np.ndarray:
    if points.empty:
        return np.zeros(len(xq), dtype=float)
    xy = points[['x', 'y']].to_numpy(float)
    vals = points['normal_deviation'].to_numpy(float)
    qxy = np.column_stack([xq, yq])
    out = np.zeros(len(qxy), dtype=float)
    for i, q in enumerate(qxy):
        d = np.sqrt(np.sum((xy - q) ** 2, axis=1))
        if np.min(d) < 1e-9:
            out[i] = vals[int(np.argmin(d))]
        else:
            w = 1.0 / np.maximum(d, 1e-9) ** power
            out[i] = float(np.sum(w * vals) / np.sum(w))
    return out


def _predict_wls(points: pd.DataFrame, xq: np.ndarray, yq: np.ndarray, ridge_lambda: float) -> np.ndarray:
    if points.empty:
        return np.zeros(len(xq), dtype=float)
    if len(points) < 3:
        return _predict_idw(points, xq, yq, power=2.0)
    fit = _fit_part_wls(points, ridge_lambda)
    Bq = _basis_eval(np.asarray(xq, dtype=float), np.asarray(yq, dtype=float), fit['norm'])
    return Bq @ fit['alpha']


def rebuild_gap_from_sms(
    pkg: SMSPackage,
    settings: SMSMappingSettings | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild g0 and sms_component from SMS point data and current ContactDomain.

    The sign convention follows g0 = nominal_gap + delta_j - delta_i for normal z-type
    deviations. If both interface sides cannot be reconstructed from eligible raw points,
    the authoritative package SMS component is retained and a WARN is recorded.
    """
    settings = SMSMappingSettings.from_dict(settings) if not isinstance(settings, SMSMappingSettings) else settings
    pts = _sms_points(pkg)
    sms_source = pts.attrs.get('sms_source', 'unknown')
    cp = pkg.contact_points.copy()
    interface = pkg.interfaces.iloc[0] if not pkg.interfaces.empty else pd.Series({})
    part_i = interface.get('part_i', None)
    part_j = interface.get('part_j', None)
    multi_interface = 'interface_id' in cp.columns and pkg.interfaces.get('interface_id', pd.Series(dtype=str)).nunique() > 1

    if settings.nominal_gap_source == 'interface_table' and 'nominal_gap' in interface:
        nominal = np.full(len(cp), float(interface['nominal_gap']), dtype=float)
    else:
        nominal = pkg.matrices.get('nominal_gap', pkg.gap_field['nominal_component'].to_numpy(float))

    def predict_at(part_id: str | None, xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, str]:
        if part_id is None:
            return np.zeros(len(xq)), 'WARN:no_part_id'
        p = pts[pts['part_id'].astype(str).eq(str(part_id))]
        if p.empty:
            return np.zeros(len(xq)), f'WARN:no_sms_points_for_{part_id}'
        if settings.method == 'idw':
            return _predict_idw(p, xq, yq, settings.idw_power), 'PASS:idw'
        return _predict_wls(p, xq, yq, settings.ridge_lambda), 'PASS:wls_basis'

    def predict(part_id: str | None, xcol: str, ycol: str) -> tuple[np.ndarray, str]:
        return predict_at(part_id, cp[xcol].to_numpy(float), cp[ycol].to_numpy(float))

    available_parts = set(pts['part_id'].dropna().astype(str)) if 'part_id' in pts.columns else set()
    if multi_interface:
        required_parts = set(pkg.interfaces.get('part_i', pd.Series(dtype=str)).dropna().astype(str))
        required_parts |= set(pkg.interfaces.get('part_j', pd.Series(dtype=str)).dropna().astype(str))
    else:
        required_parts = {str(p) for p in [part_i, part_j] if p is not None and pd.notna(p)}
    can_rebuild_both_sides = bool(required_parts) and required_parts <= available_parts
    point_part_i = np.full(len(cp), part_i, dtype=object)
    point_part_j = np.full(len(cp), part_j, dtype=object)
    if can_rebuild_both_sides and multi_interface:
        delta_i = np.zeros(len(cp), dtype=float)
        delta_j = np.zeros(len(cp), dtype=float)
        statuses_i: list[str] = []
        statuses_j: list[str] = []
        interface_map = pkg.interfaces.set_index('interface_id', drop=False)
        if settings.nominal_gap_source == 'interface_table':
            nominal = np.asarray(nominal, dtype=float).copy()
        for interface_id, indices in cp.groupby('interface_id', sort=False).groups.items():
            if interface_id not in interface_map.index:
                statuses_i.append(f'WARN:missing_interface_{interface_id}')
                statuses_j.append(f'WARN:missing_interface_{interface_id}')
                continue
            interface_row = interface_map.loc[interface_id]
            pi, pj = interface_row.get('part_i'), interface_row.get('part_j')
            idx = np.asarray(list(indices), dtype=int)
            point_part_i[idx] = pi
            point_part_j[idx] = pj
            delta_i[idx], si = predict_at(pi, cp.loc[idx, 'x_i0'].to_numpy(float), cp.loc[idx, 'y_i0'].to_numpy(float))
            delta_j[idx], sj = predict_at(pj, cp.loc[idx, 'x_j0'].to_numpy(float), cp.loc[idx, 'y_j0'].to_numpy(float))
            statuses_i.append(f'{interface_id}:{si}')
            statuses_j.append(f'{interface_id}:{sj}')
            if settings.nominal_gap_source == 'interface_table' and pd.notna(interface_row.get('nominal_gap')):
                nominal[idx] = float(interface_row.get('nominal_gap'))
        sms_component = delta_j - delta_i
        status_i = ('WARN:' if any('WARN:' in s for s in statuses_i) else 'PASS:') + ';'.join(statuses_i)
        status_j = ('WARN:' if any('WARN:' in s for s in statuses_j) else 'PASS:') + ';'.join(statuses_j)
        mapping_mode = 'LIVE_REBUILD_MULTI_INTERFACE'
    elif can_rebuild_both_sides:
        delta_i, status_i = predict(part_i, 'x_i0', 'y_i0')
        delta_j, status_j = predict(part_j, 'x_j0', 'y_j0')
        sms_component = delta_j - delta_i
        mapping_mode = 'LIVE_REBUILD'
    elif not pkg.package_type.startswith('V25') and not pts.empty:
        # Preserve the legacy E1 convention in which the measured panel side is updated
        # and an intentionally nominal/unmeasured opposite side contributes zero.
        delta_i, status_i = predict(part_i, 'x_i0', 'y_i0')
        delta_j, status_j = predict(part_j, 'x_j0', 'y_j0')
        sms_component = delta_j - delta_i
        mapping_mode = 'LIVE_REBUILD_PARTIAL_LEGACY'
    else:
        # V2.5 may intentionally provide the already reconstructed SMSField/GapField rather
        # than raw points for both interface sides. Preserve that authoritative frozen field
        # instead of treating a missing side as zero or raising KeyError.
        delta_i = np.zeros(len(cp), dtype=float)
        delta_j = np.zeros(len(cp), dtype=float)
        sms_component = np.asarray(pkg.matrices.get('sms_component', np.zeros(len(cp))), dtype=float).reshape(-1)
        status_i = 'WARN:fallback_to_package_sms_component'
        status_j = 'WARN:fallback_to_package_sms_component'
        mapping_mode = 'PACKAGE_FROZEN_FALLBACK'
    pose = pkg.matrices.get('pose_component', np.zeros(len(cp)))
    g0 = nominal + sms_component + pose
    base_g0 = pkg.matrices.get('g0', pkg.gap_field['values_g'].to_numpy(float))
    diff = g0 - base_g0

    gap_cols = [
        c for c in ['candidate_id', 'contact_domain_id', 'interface_id', 'local_index', 'x_i0', 'y_i0', 'x_j0', 'y_j0']
        if c in cp.columns
    ]
    gap_table = cp[gap_cols].copy()
    gap_table['part_i'] = point_part_i
    gap_table['part_j'] = point_part_j
    gap_table['delta_i_sms_mm'] = delta_i
    gap_table['delta_j_sms_mm'] = delta_j
    gap_table['sms_component_rebuilt_mm'] = sms_component
    gap_table['nominal_component_mm'] = nominal
    gap_table['g0_rebuilt_mm'] = g0
    gap_table['g0_package_mm'] = base_g0
    gap_table['difference_from_package_mm'] = diff

    quality = pd.DataFrame([
        {'check_item': 'SMS source', 'status': 'PASS' if not pts.empty else 'WARN', 'detail': f'{sms_source}; {len(pts)} eligible records'},
        {'check_item': 'SMS mapping mode', 'status': 'PASS' if mapping_mode.startswith('LIVE_REBUILD') else 'WARN', 'detail': mapping_mode},
        {'check_item': 'part_i mapping', 'status': status_i.split(':')[0], 'detail': status_i},
        {'check_item': 'part_j mapping', 'status': status_j.split(':')[0], 'detail': status_j},
        {'check_item': 'g0 rebuild difference', 'status': 'PASS' if np.max(np.abs(diff)) < 0.030 else 'WARN', 'detail': f'max_abs_diff={np.max(np.abs(diff)):.5f} mm'},
    ])
    fit_summary = fit_sms_wls_map(pkg, settings)
    return {
        'nominal_gap': nominal,
        'sms_component': sms_component,
        'pose_component': pose,
        'g0': g0,
        'gap_table': gap_table,
        'fit_summary': fit_summary,
        'quality': quality,
        'mapping_mode': mapping_mode,
        'sms_source': sms_source,
    }
