from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .stage_solver import run_all_stages
from .numerical_substitution import NumericalSubstitutionSettings
from .sms_mapping import SMSMappingSettings
from .overconstraint import OverConstraintSettings
from .tangential_ncp import TangentialNCPSettings
from .kcp import extract_kcp


_VAR_ALIASES = {
    'sms_scale': 'sms_scale',
    'SMS_scale': 'sms_scale',
    'clamp_closure_scale': 'closure_scale',
    'closure_scale': 'closure_scale',
    'stage_closure_scale': 'closure_scale',
    'Cn_scale': 'cn_scale',
    'cn_scale': 'cn_scale',
    'C_n_scale': 'cn_scale',
}


def _distribution_file(root: Path) -> dict[str, Any]:
    path = root / 'I_stat' / 'distributions.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def distribution_defaults(pkg: SMSPackage) -> dict[str, tuple[float, float]]:
    """Return mean/std defaults for sms_scale, closure_scale and cn_scale."""
    raw = _distribution_file(pkg.root)
    raw_vars = raw.get('variables', raw) if isinstance(raw, dict) else {}
    out = {
        'sms_scale': (1.0, 0.08),
        'closure_scale': (1.0, 0.05),
        'cn_scale': (1.0, 0.10),
    }
    for k, v in raw_vars.items():
        canonical = _VAR_ALIASES.get(k)
        if not canonical or not isinstance(v, dict):
            continue
        out[canonical] = (float(v.get('mean', out[canonical][0])), float(v.get('std', out[canonical][1])))
    return out


def run_monte_carlo(
    pkg: SMSPackage,
    n_samples: int,
    seed: int,
    means_stds: dict[str, tuple[float, float]],
    *,
    eps: float = 1e-9,
    clip_min: float = 0.05,
    substitution_settings: NumericalSubstitutionSettings | dict | None = None,
    sms_mapping_settings: SMSMappingSettings | dict | None = None,
    overconstraint_settings: OverConstraintSettings | dict | None = None,
    tangential_settings: TangentialNCPSettings | dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for i in range(int(n_samples)):
        sms_scale = max(clip_min, float(rng.normal(*means_stds['sms_scale'])))
        closure_scale = max(clip_min, float(rng.normal(*means_stds['closure_scale'])))
        cn_scale = max(clip_min, float(rng.normal(*means_stds['cn_scale'])))
        try:
            result = run_all_stages(pkg, sms_scale=sms_scale, closure_scale=closure_scale, cn_scale=cn_scale, eps=eps, substitution_settings=substitution_settings, sms_mapping_settings=sms_mapping_settings, overconstraint_settings=overconstraint_settings, tangential_settings=tangential_settings)
            kcp = extract_kcp(pkg, result)
            row: dict[str, float | int | str] = {
                'sample_no': i + 1,
                'sms_scale': sms_scale,
                'closure_scale': closure_scale,
                'cn_scale': cn_scale,
                'status': 'PASS',
            }
            for _, r in kcp.iterrows():
                row[str(r['kcp_id'])] = float(r['predicted_value'])
            rows.append(row)
        except Exception as exc:
            rows.append({
                'sample_no': i + 1,
                'sms_scale': sms_scale,
                'closure_scale': closure_scale,
                'cn_scale': cn_scale,
                'status': f'FAIL: {exc}',
            })

    samples = pd.DataFrame(rows)
    kcp_cols = [c for c in samples.columns if c.startswith('KCP_')]
    stat_rows = []
    val = pkg.validation_kcp.set_index('kcp_id') if 'kcp_id' in pkg.validation_kcp.columns else pd.DataFrame()
    kcp_defs = pkg.kcp_kcm[pkg.kcp_kcm.get('feature_role', '') == 'KCP'].set_index('feature_id') if 'feature_id' in pkg.kcp_kcm.columns else pd.DataFrame()
    for col in kcp_cols:
        s = pd.to_numeric(samples[col], errors='coerce').dropna()
        lower = float(kcp_defs.loc[col, 'lower_tol']) if col in kcp_defs.index and 'lower_tol' in kcp_defs.columns else np.nan
        upper = float(kcp_defs.loc[col, 'upper_tol']) if col in kcp_defs.index and 'upper_tol' in kcp_defs.columns else np.nan
        stat_rows.append({
            'kcp_id': col,
            'mean': float(s.mean()) if len(s) else np.nan,
            'std': float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            'p05': float(s.quantile(0.05)) if len(s) else np.nan,
            'p50': float(s.quantile(0.50)) if len(s) else np.nan,
            'p95': float(s.quantile(0.95)) if len(s) else np.nan,
            'min': float(s.min()) if len(s) else np.nan,
            'max': float(s.max()) if len(s) else np.nan,
            'lower_tol': lower,
            'upper_tol': upper,
            'exceed_prob': _exceed_probability(s, lower, upper),
            'validation_value': float(val.loc[col, 'measured_value']) if col in val.index and 'measured_value' in val.columns else np.nan,
        })
    stats = pd.DataFrame(stat_rows)
    return samples, stats


def _exceed_probability(s: pd.Series, lower: float, upper: float) -> float:
    if len(s) == 0 or (not np.isfinite(lower) and not np.isfinite(upper)):
        return np.nan
    mask = pd.Series(False, index=s.index)
    if np.isfinite(lower):
        mask |= s < lower
    if np.isfinite(upper):
        mask |= s > upper
    return float(mask.mean())


def one_factor_sweep(
    pkg: SMSPackage,
    variable: str,
    values: list[float],
    baseline: dict[str, float],
    *,
    eps: float = 1e-9,
    substitution_settings: NumericalSubstitutionSettings | dict | None = None,
    sms_mapping_settings: SMSMappingSettings | dict | None = None,
    overconstraint_settings: OverConstraintSettings | dict | None = None,
    tangential_settings: TangentialNCPSettings | dict | None = None,
) -> pd.DataFrame:
    rows = []
    for v in values:
        params = dict(baseline)
        params[variable] = float(v)
        result = run_all_stages(pkg, sms_scale=params['sms_scale'], closure_scale=params['closure_scale'], cn_scale=params['cn_scale'], eps=eps, substitution_settings=substitution_settings, sms_mapping_settings=sms_mapping_settings, overconstraint_settings=overconstraint_settings, tangential_settings=tangential_settings)
        kcp = extract_kcp(pkg, result)
        row: dict[str, float | str] = {'variable': variable, 'value': float(v)}
        for _, r in kcp.iterrows():
            row[str(r['kcp_id'])] = float(r['predicted_value'])
        rows.append(row)
    return pd.DataFrame(rows)
