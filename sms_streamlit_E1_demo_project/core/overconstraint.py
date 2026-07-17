from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from .data_loader import SMSPackage
from .lcp_solver import solve_lcp_active_set, LCPSolution


@dataclass
class OverConstraintSettings:
    enabled: bool = False
    use_locator: bool = True
    use_clamp: bool = True
    use_joint: bool = True
    scale_gap_effect: float = 1.0
    scale_compliance: float = 1.0
    coupling_ratio: float = 0.05
    regularization: float = 1e-12

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'OverConstraintSettings':
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_overconstraint_tables(pkg: SMSPackage) -> dict[str, pd.DataFrame | dict[str, Any]]:
    model_path = pkg.root / 'I_pred' / 'overconstraint_contact_model.json'
    model: dict[str, Any] = {}
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding='utf-8'))
    return {
        'group': _read_csv(pkg.root / 'I_stage' / 'overconstraint_locator_group.csv'),
        'elements': _read_csv(pkg.root / 'I_stage' / 'locator_element.csv'),
        'compliance': _read_csv(pkg.root / 'I_stage' / 'locator_compliance.csv'),
        'connection_lock': _read_csv(pkg.root / 'I_pred' / 'connection_lock_history.csv'),
        'model': model,
    }


def _stage_active_mask(values: pd.Series, stage_id: str) -> pd.Series:
    def ok(s: str) -> bool:
        s = str(s)
        return s in ('ALL', '*') or stage_id == s or stage_id in s.split('|')
    return values.astype(str).apply(ok)


def active_overconstraint_elements(pkg: SMSPackage, stage_id: str, settings: OverConstraintSettings | dict[str, Any] | None = None) -> pd.DataFrame:
    settings = OverConstraintSettings.from_dict(settings) if not isinstance(settings, OverConstraintSettings) else settings
    tables = load_overconstraint_tables(pkg)
    el = tables['elements']  # type: ignore[index]
    if el.empty:
        return pd.DataFrame()
    df = el.copy()
    if 'activation_stages' in df.columns:
        df = df[_stage_active_mask(df['activation_stages'], stage_id)]
    elif 'stage_id' in df.columns:
        df = df[_stage_active_mask(df['stage_id'], stage_id)]
    if 'element_type' in df.columns:
        mask = pd.Series(True, index=df.index)
        mask &= df['element_type'].astype(str).str.lower().ne('locator') | bool(settings.use_locator)
        mask &= df['element_type'].astype(str).str.lower().ne('clamp') | bool(settings.use_clamp)
        mask &= df['element_type'].astype(str).str.lower().ne('joint') | bool(settings.use_joint)
        df = df[mask]
    if 'quality_flag' in df.columns:
        df = df[df['quality_flag'].astype(str).str.upper().isin(['PASS', 'WARN'])]
    comp = tables['compliance']  # type: ignore[index]
    if not comp.empty and 'element_id' in comp.columns:
        keep_cols = [c for c in ['element_id', 'C_locator', 'covariance', 'D_valid', 'stiffness_source'] if c in comp.columns]
        df = df.merge(comp[keep_cols], on='element_id', how='left')
    return df.reset_index(drop=True)


def _extra_compliance(row: pd.Series, scale: float) -> float:
    for col in ['C_locator', 'compliance_mm_per_N']:
        if col in row and pd.notna(row[col]):
            val = float(row[col])
            if val > 0:
                return scale * val
    k = float(row.get('stiffness_N_per_mm', row.get('stiffness', 0.0)) or 0.0)
    if k > 0:
        return scale * (1.0 / k)
    return scale * 1e-6


def _element_stage_effect(row: pd.Series, stage_id: str) -> float:
    # stage_effect_mm is a compact manual/demo field. Specific fields can override it.
    specific = f'stage_effect_{stage_id}_mm'
    if specific in row and pd.notna(row[specific]):
        return float(row[specific])
    return float(row.get('stage_effect_mm', 0.0) or 0.0)


def build_extended_lcp_model(
    pkg: SMSPackage,
    stage_id: str,
    q_contact: np.ndarray,
    W_contact: np.ndarray,
    settings: OverConstraintSettings | dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = OverConstraintSettings.from_dict(settings) if not isinstance(settings, OverConstraintSettings) else settings
    q_contact = np.asarray(q_contact, dtype=float).reshape(-1)
    W_contact = np.asarray(W_contact, dtype=float)
    m = len(q_contact)
    elements = active_overconstraint_elements(pkg, stage_id, settings)
    if not settings.enabled or elements.empty:
        return {
            'enabled': False,
            'q_ext': q_contact,
            'W_ext': W_contact,
            'elements': elements,
            'contact_count': m,
            'extended_count': 0,
        }

    q_extra = []
    c_extra = []
    cp_xy = pkg.contact_points[['x_i0', 'y_i0']].to_numpy(float)
    coupling_cols = []
    diag_contact = np.maximum(np.diag(W_contact), 1e-15)
    for _, row in elements.iterrows():
        gap0 = float(row.get('gap0', row.get('gap0_mm', 0.0)) or 0.0)
        effect = _element_stage_effect(row, stage_id)
        q_extra.append(gap0 - settings.scale_gap_effect * effect)
        ce = _extra_compliance(row, settings.scale_compliance)
        c_extra.append(ce)
        x = float(row.get('x', row.get('position_x', np.nan)))
        y = float(row.get('y', row.get('position_y', np.nan)))
        if not np.isfinite(x) or not np.isfinite(y):
            weight = np.zeros(m)
        else:
            radius = max(float(row.get('effect_radius_mm', 25.0) or 25.0), 1e-6)
            dist2 = np.sum((cp_xy - np.array([x, y])) ** 2, axis=1)
            weight = np.exp(-dist2 / (2 * radius ** 2))
            if weight.max(initial=0.0) > 0:
                weight = weight / weight.max()
        coupling_cols.append(settings.coupling_ratio * np.sqrt(diag_contact * ce) * weight)
    q_extra_arr = np.asarray(q_extra, dtype=float)
    c_extra_arr = np.asarray(c_extra, dtype=float)
    n_extra = len(q_extra_arr)
    coupling = np.column_stack(coupling_cols) if coupling_cols else np.zeros((m, 0))
    W_extra = np.diag(np.maximum(c_extra_arr, 1e-15))
    W_ext = np.block([
        [W_contact, coupling],
        [coupling.T, W_extra],
    ])
    W_ext = 0.5 * (W_ext + W_ext.T) + settings.regularization * np.eye(m + n_extra)
    q_ext = np.concatenate([q_contact, q_extra_arr])
    elements = elements.copy()
    elements['q_extra_mm'] = q_extra_arr
    elements['C_extra_mm_per_N'] = c_extra_arr
    return {
        'enabled': True,
        'q_ext': q_ext,
        'W_ext': W_ext,
        'elements': elements,
        'contact_count': m,
        'extended_count': n_extra,
        'coupling_matrix': coupling,
    }


def solve_extended_lcp(
    pkg: SMSPackage,
    stage_id: str,
    q_contact: np.ndarray,
    W_contact: np.ndarray,
    eps: float = 1e-9,
    settings: OverConstraintSettings | dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = build_extended_lcp_model(pkg, stage_id, q_contact, W_contact, settings)
    sol = solve_lcp_active_set(model['q_ext'], model['W_ext'], eps=eps)
    m = int(model['contact_count'])
    elements = model['elements'].copy()
    if not elements.empty:
        elements['lambda_ext_N'] = sol.lambda_n[m:]
        elements['gap_ext_mm'] = sol.gap_g[m:]
        elements['active_flag'] = [int((m + i) in sol.active_indices) for i in range(len(elements))]
    return {
        **model,
        'solution': sol,
        'contact_lambda': sol.lambda_n[:m],
        'contact_gap': sol.gap_g[:m],
        'extra_lambda': sol.lambda_n[m:],
        'extra_gap': sol.gap_g[m:],
        'elements_solution': elements,
    }


def extended_solution_table(stage_id: str, ext: dict[str, Any]) -> pd.DataFrame:
    elements = ext.get('elements_solution', pd.DataFrame())
    if elements is None or elements.empty:
        return pd.DataFrame()
    df = elements.copy()
    df['stage_id'] = stage_id
    return df


def active_set_stability_trace(sol: LCPSolution) -> pd.DataFrame:
    rows = []
    last = None
    for item in sol.active_set_trace:
        aset = tuple(item.get('active_set', []))
        jaccard = 1.0
        if last is not None:
            a, b = set(aset), set(last)
            union = len(a | b)
            jaccard = len(a & b) / union if union else 1.0
        rows.append({
            'iteration': item.get('iteration'),
            'action': item.get('action'),
            'index': item.get('index'),
            'active_count': len(aset),
            'rho_chi': jaccard,
            'oscillation_flag': False,
        })
        last = aset
    return pd.DataFrame(rows)


def force_nonuniqueness_report(W: np.ndarray, active_indices: list[int], kcp_map_norm: float = 1.0, tol: float = 1e-10) -> dict[str, Any]:
    if not active_indices:
        return {'nullspace_dimension': 0, 'min_singular_value': np.nan, 'kcp_stability': 'PASS', 'detail': 'no active constraints'}
    idx = np.array(active_indices, dtype=int)
    Wcc = W[np.ix_(idx, idx)]
    s = np.linalg.svd(Wcc, compute_uv=False)
    null_dim = int(np.sum(s < tol))
    min_sv = float(s.min()) if len(s) else np.nan
    # A compact engineering indicator: if Wcc has a nullspace, KCP can still be stable when
    # the reported output variation bound is tiny. Here we conservatively warn.
    variation_bound = float(null_dim * kcp_map_norm * tol / max(min_sv, tol)) if np.isfinite(min_sv) else np.nan
    return {
        'nullspace_dimension': null_dim,
        'min_singular_value': min_sv,
        'alternative_force_bounds': 'not enumerated in demo; use local FE/regularized solve for bounds',
        'output_variation_bounds': variation_bound,
        'kcp_stability': 'WARN' if null_dim > 0 else 'PASS',
        'detail': 'active-set W_CC singularity check',
    }
