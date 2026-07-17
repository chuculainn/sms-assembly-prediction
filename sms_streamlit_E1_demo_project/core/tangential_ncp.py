from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage


@dataclass
class TangentialNCPSettings:
    enabled: bool = False
    shear_scale: float = 1.0
    fallback_shear_from_gradient: bool = True
    slip_tolerance: float = 1e-10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'TangentialNCPSettings':
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def load_tangential_free_slip(pkg: SMSPackage, stage_id: str) -> pd.DataFrame:
    paths = [
        pkg.root / 'I_stage' / 'tangential_free_slip.csv',
        pkg.root / 'I_substitution' / 'tangential_free_slip.csv',
    ]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            if 'stage_id' in df.columns:
                df = df[df['stage_id'].astype(str).apply(lambda s: s == stage_id or s == 'ALL' or stage_id in s.split('|'))]
            return df.reset_index(drop=True)
    return pd.DataFrame()


def build_tangential_free_vector(pkg: SMSPackage, stage_id: str, qn: np.ndarray, settings: TangentialNCPSettings) -> np.ndarray:
    n = len(pkg.contact_points)
    df = load_tangential_free_slip(pkg, stage_id)
    if not df.empty and {'candidate_id', 't1_free_mm', 't2_free_mm'} <= set(df.columns):
        merged = pkg.contact_points[['candidate_id']].merge(df[['candidate_id', 't1_free_mm', 't2_free_mm']], on='candidate_id', how='left')
        qt = merged[['t1_free_mm', 't2_free_mm']].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(float)
        return settings.shear_scale * qt

    if not settings.fallback_shear_from_gradient:
        return np.zeros((n, 2), dtype=float)
    # Deterministic fallback: small tangential free slip proportional to contact closure
    # and normalized x/y position. This keeps the module executable when no process shear
    # sensor data has been provided, but should be replaced by measured/FE tangential data.
    cp = pkg.contact_points
    x = cp['x_i0'].to_numpy(float)
    y = cp['y_i0'].to_numpy(float)
    xr = (x - x.mean()) / max(np.ptp(x), 1.0)
    yr = (y - y.mean()) / max(np.ptp(y), 1.0)
    amp = np.maximum(-np.asarray(qn, dtype=float), 0.0)
    return settings.shear_scale * np.column_stack([0.15 * amp * xr, 0.10 * amp * yr])


def solve_tangential_projection(
    lambda_n: np.ndarray,
    q_t: np.ndarray,
    Ct_diag: np.ndarray,
    mu: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Projected local tangential NCP/friction cone solve.

    This is an uncoupled local complement to the normal LCP. It uses the normal force from
    the normal solve, computes a trial stick tangential force, then projects it onto the
    Coulomb cone ||lambda_t|| <= mu * lambda_n.
    """
    lambda_n = np.asarray(lambda_n, dtype=float).reshape(-1)
    q_t = np.asarray(q_t, dtype=float).reshape((-1, 2))
    Ct_diag = np.maximum(np.asarray(Ct_diag, dtype=float).reshape(-1), eps)
    mu = np.maximum(np.asarray(mu, dtype=float).reshape(-1), 0.0)
    n = len(lambda_n)
    lam_t = np.zeros((n, 2), dtype=float)
    d_t = np.zeros((n, 2), dtype=float)
    state = np.array(['open'] * n, dtype=object)
    friction_limit = mu * np.maximum(lambda_n, 0.0)
    for k in range(n):
        if lambda_n[k] <= eps:
            d_t[k] = q_t[k]
            state[k] = 'open'
            continue
        trial = -q_t[k] / Ct_diag[k]
        norm_trial = float(np.linalg.norm(trial))
        limit = friction_limit[k]
        if norm_trial <= limit + eps:
            lam_t[k] = trial
            d_t[k] = q_t[k] + Ct_diag[k] * lam_t[k]
            state[k] = 'stick'
        else:
            qnorm = float(np.linalg.norm(q_t[k]))
            if qnorm <= eps:
                direction = trial / max(norm_trial, eps)
            else:
                direction = -q_t[k] / qnorm
            lam_t[k] = limit * direction
            d_t[k] = q_t[k] + Ct_diag[k] * lam_t[k]
            state[k] = 'slip'
    return {
        'lambda_t': lam_t,
        'd_t': d_t,
        'state': state,
        'friction_limit': friction_limit,
        'cone_residual': np.maximum(np.linalg.norm(lam_t, axis=1) - friction_limit, 0.0),
    }


def tangential_result_table(
    pkg: SMSPackage,
    stage_id: str,
    lambda_n: np.ndarray,
    qn: np.ndarray,
    components: pd.DataFrame | None,
    settings: TangentialNCPSettings | dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = TangentialNCPSettings.from_dict(settings) if not isinstance(settings, TangentialNCPSettings) else settings
    n = len(pkg.contact_points)
    if components is not None and not components.empty:
        Ct = pd.to_numeric(components.get('Ct_equiv_mm_per_N', pd.Series(np.zeros(n))), errors='coerce').fillna(0.0).to_numpy(float)
        mu = pd.to_numeric(components.get('mu_eff', pd.Series(np.ones(n) * 0.25)), errors='coerce').fillna(0.25).to_numpy(float)
    else:
        Ct = np.ones(n, dtype=float) * 1e-5
        mu = np.ones(n, dtype=float) * 0.25
    # Prevent all-zero Ct from making tangential forces unbounded; if a parameter table lacks Ct,
    # use a small transparent default and mark quality WARN later.
    Ct_eff = np.where(Ct > 1e-12, Ct, 1e-5)
    q_t = build_tangential_free_vector(pkg, stage_id, qn, settings)
    sol = solve_tangential_projection(lambda_n, q_t, Ct_eff, mu)
    out = pkg.contact_points[['candidate_id', 'local_index', 'x_i0', 'y_i0']].copy()
    out['stage_id'] = stage_id
    out['t1_free_mm'] = q_t[:, 0]
    out['t2_free_mm'] = q_t[:, 1]
    out['Ct_eff_mm_per_N'] = Ct_eff
    out['mu_eff'] = mu
    out['lambda_t1_N'] = sol['lambda_t'][:, 0]
    out['lambda_t2_N'] = sol['lambda_t'][:, 1]
    out['lambda_t_norm_N'] = np.linalg.norm(sol['lambda_t'], axis=1)
    out['friction_limit_N'] = sol['friction_limit']
    out['d_t1_mm'] = sol['d_t'][:, 0]
    out['d_t2_mm'] = sol['d_t'][:, 1]
    out['stick_slip_state'] = sol['state']
    out['cone_residual'] = sol['cone_residual']
    out['quality_flag'] = np.where(Ct > 1e-12, 'PASS', 'WARN_NO_CT')
    return out


def tangential_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame([
        {'metric': 'open_count', 'value': int((df['stick_slip_state'] == 'open').sum())},
        {'metric': 'stick_count', 'value': int((df['stick_slip_state'] == 'stick').sum())},
        {'metric': 'slip_count', 'value': int((df['stick_slip_state'] == 'slip').sum())},
        {'metric': 'max_lambda_t_norm_N', 'value': float(df['lambda_t_norm_N'].max())},
        {'metric': 'max_cone_residual', 'value': float(df['cone_residual'].max())},
    ])
