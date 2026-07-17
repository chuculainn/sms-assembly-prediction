from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from .data_loader import SMSPackage


STAGE_ORDER = ['S_LOCATE_01', 'S_CLAMP_02', 'S_JOIN_03', 'S_RELEASE_04']


@dataclass
class NumericalSubstitutionSettings:
    """Runtime switches for numerical-substitution modules.

    mode:
        base_only  -> use the Cn matrix already stored in matrices/E1_matrices.npz
        replace    -> use only the numerical substitution Cn components
        add        -> use base Cn + numerical substitution Cn components
    """

    enabled: bool = False
    mode: str = 'base_only'
    use_partition: bool = True
    use_layer: bool = True
    use_rough: bool = True
    use_indent: bool = True
    use_locator: bool = True
    use_clamp: bool = True
    use_joint: bool = True
    use_release: bool = True
    scale_partition: float = 1.0
    scale_layer: float = 1.0
    scale_rough: float = 1.0
    scale_indent: float = 1.0
    scale_locator: float = 1.0
    scale_clamp: float = 1.0
    scale_joint: float = 1.0
    scale_release: float = 1.0
    global_scale: float = 1.0
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'NumericalSubstitutionSettings':
        if data is None:
            return cls()
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def substitution_dir(pkg_or_root: SMSPackage | str | Path) -> Path:
    root = pkg_or_root.root if isinstance(pkg_or_root, SMSPackage) else Path(pkg_or_root)
    return Path(root) / 'I_substitution'


def load_substitution_config(pkg: SMSPackage) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Read optional I_substitution tables. Missing files are replaced by empty/default tables.

    The app is intentionally tolerant: old data packages still run, while new packages can
    supply additional numerical substitution tables.
    """
    root = substitution_dir(pkg)
    config_path = root / 'numerical_substitution_config.json'
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))

    def csv(name: str, columns: list[str]) -> pd.DataFrame:
        path = root / name
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame(columns=columns)

    return {
        'config': config,
        'partition_cn': csv('partition_cn.csv', ['candidate_id', 'zone_id', 'C_partition_mm_per_N', 'Ct_mm_per_N', 'mu', 'beta_r', 'quality_flag']),
        'layer_stack': csv('layer_stack.csv', ['layer_id', 'zone_id', 'thickness_mm', 'E_eff_MPa', 'active', 'quality_flag']),
        'rough_contact': csv('rough_contact.csv', ['zone_id', 'c_ref_mm_per_N', 'p_ref_MPa', 'exponent', 'roughness_factor', 'active', 'quality_flag']),
        'local_indent': csv('local_indent.csv', ['zone_id', 'C_indent_mm_per_N', 'load_softening_exponent', 'active', 'quality_flag']),
        'fixture_joint': csv('fixture_joint_compliance.csv', ['element_id', 'element_type', 'stage_id', 'x', 'y', 'effect_radius_mm', 'stiffness_N_per_mm', 'active', 'quality_flag']),
        'release_rebound': csv('release_rebound.csv', ['candidate_id', 'zone_id', 'release_rebound_mm', 'beta_r', 'active', 'quality_flag']),
        'validity': csv('module_validity.csv', ['module_name', 'valid_stage_ids', 'valid_load_min_N', 'valid_load_max_N', 'included_physics', 'excluded_physics', 'quality_flag']),
    }


def _stage_load_estimate(pkg: SMSPackage, stage_id: str) -> float:
    """Estimate a representative positive load for rough/contact validity checks."""
    proc = pkg.process_record.copy()
    if 'stage_id' not in proc.columns:
        return 1.0
    stage_proc = proc[proc['stage_id'] == stage_id]
    if stage_proc.empty or 'value' not in stage_proc.columns:
        return 1.0
    vals = pd.to_numeric(stage_proc['value'], errors='coerce').abs().dropna()
    if vals.empty:
        return 1.0
    return float(vals.sum())


def _zone_series(pkg: SMSPackage, partition: pd.DataFrame) -> pd.Series:
    cp = pkg.contact_points[['candidate_id', 'edge_or_interior_flag']].copy()
    fallback = pd.Series(
        np.where(cp['edge_or_interior_flag'].astype(str).str.lower().eq('edge'), 'edge', 'center'),
        index=cp.index,
    )
    if not partition.empty and {'candidate_id', 'zone_id'} <= set(partition.columns):
        zones = cp[['candidate_id']].merge(partition[['candidate_id', 'zone_id']], on='candidate_id', how='left')['zone_id']
        zones.index = cp.index
    else:
        zones = fallback.copy()
    return zones.where(zones.notna(), fallback)


def _map_by_zone(zones: pd.Series, table: pd.DataFrame, value_col: str, default: float = 0.0) -> np.ndarray:
    if table.empty or value_col not in table.columns or 'zone_id' not in table.columns:
        return np.full(len(zones), default, dtype=float)
    t = table.copy()
    if 'active' in t.columns:
        t = t[t['active'].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active'])]
    if t.empty:
        return np.full(len(zones), default, dtype=float)
    mapping = t.groupby('zone_id')[value_col].apply(lambda s: float(pd.to_numeric(s, errors='coerce').dropna().sum())).to_dict()
    return zones.map(lambda z: mapping.get(z, default)).to_numpy(float)


def _fixture_influence(pkg: SMSPackage, table: pd.DataFrame, stage_id: str, element_type: str) -> np.ndarray:
    cp = pkg.contact_points
    out = np.zeros(len(cp), dtype=float)
    if table.empty:
        return out
    required = {'element_type', 'stage_id', 'x', 'y', 'effect_radius_mm', 'stiffness_N_per_mm'}
    if not required <= set(table.columns):
        return out
    t = table[table['element_type'].astype(str).str.lower().eq(element_type.lower())].copy()
    # stage_id may be ALL, a single stage, or pipe-separated stages
    t = t[t['stage_id'].astype(str).apply(lambda s: s == 'ALL' or stage_id in s.split('|'))]
    if 'active' in t.columns:
        t = t[t['active'].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active'])]
    if t.empty:
        return out
    xy = cp[['x_i0', 'y_i0']].to_numpy(float)
    for _, row in t.iterrows():
        k = float(row['stiffness_N_per_mm'])
        if k <= 0:
            continue
        center = np.array([float(row['x']), float(row['y'])], dtype=float)
        radius = max(float(row['effect_radius_mm']), 1e-6)
        dist2 = np.sum((xy - center) ** 2, axis=1)
        weight = np.exp(-dist2 / (2 * radius ** 2))
        # Spread one local spring contribution over nearby contact points. The 1/sqrt(area)
        # normalization avoids excessive sensitivity when point density changes.
        if weight.sum() > 0:
            weight = weight / weight.max()
        out += (1.0 / k) * weight
    return out


def _release_rebound_vector(pkg: SMSPackage, tables: dict[str, pd.DataFrame | dict[str, Any]], settings: NumericalSubstitutionSettings) -> np.ndarray:
    rel = tables['release_rebound']  # type: ignore[index]
    n = len(pkg.contact_points)
    out = np.zeros(n, dtype=float)
    if not settings.enabled or not settings.use_release or rel.empty:
        return out
    cp = pkg.contact_points[['candidate_id']].copy()
    df = cp.merge(rel, on='candidate_id', how='left') if 'candidate_id' in rel.columns else cp
    if 'release_rebound_mm' in df.columns:
        vals = pd.to_numeric(df['release_rebound_mm'], errors='coerce').fillna(0.0).to_numpy(float)
        beta = pd.to_numeric(df.get('beta_r', pd.Series(np.ones(n))), errors='coerce').fillna(1.0).to_numpy(float)
        active = df.get('active', pd.Series(['1'] * n)).astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active']).to_numpy()
        out = vals * beta * active * settings.scale_release
    return out


def compute_substitution_components(
    pkg: SMSPackage,
    stage_id: str,
    settings: NumericalSubstitutionSettings | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute per-candidate numerical-substitution components for one stage."""
    settings = NumericalSubstitutionSettings.from_dict(settings) if not isinstance(settings, NumericalSubstitutionSettings) else settings
    tables = load_substitution_config(pkg)
    n = len(pkg.contact_points)
    cp = pkg.contact_points[['candidate_id', 'x_i0', 'y_i0', 'area_weight', 'edge_or_interior_flag']].copy()
    partition = tables['partition_cn']  # type: ignore[index]
    zones = _zone_series(pkg, partition)
    area = cp['area_weight'].to_numpy(float)
    stage_load = _stage_load_estimate(pkg, stage_id)
    total_area = max(float(area.sum()), 1e-9)
    p_stage = max(stage_load / total_area, 1e-6)  # N/mm^2 = MPa

    base_diag = np.diag(pkg.matrices['Cn_local']).astype(float)

    # Partition Cn, Ct, mu and beta_r are point/zone parameters from local FE/test equivalents.
    if not partition.empty and {'candidate_id', 'C_partition_mm_per_N'} <= set(partition.columns):
        merged = cp[['candidate_id']].merge(partition, on='candidate_id', how='left')
        C_partition = pd.to_numeric(merged.get('C_partition_mm_per_N'), errors='coerce').fillna(0.0).to_numpy(float)
        Ct = pd.to_numeric(merged.get('Ct_mm_per_N'), errors='coerce').fillna(0.0).to_numpy(float)
        mu = pd.to_numeric(merged.get('mu'), errors='coerce').fillna(0.25).to_numpy(float)
        beta = pd.to_numeric(merged.get('beta_r'), errors='coerce').fillna(0.30).to_numpy(float)
    else:
        C_partition = np.zeros(n)
        Ct = np.zeros(n)
        mu = np.full(n, 0.25)
        beta = np.full(n, 0.30)

    # Thin layer / coating: C = t / (E * A)
    layer = tables['layer_stack']  # type: ignore[index]
    C_layer = np.zeros(n)
    if not layer.empty and settings.use_layer:
        t = layer.copy()
        if 'active' in t.columns:
            t = t[t['active'].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active'])]
        by_zone: dict[str, float] = {}
        for _, row in t.iterrows():
            E = float(row.get('E_eff_MPa', 0.0))
            thickness = float(row.get('thickness_mm', 0.0))
            if E > 0 and thickness >= 0:
                by_zone[row['zone_id']] = by_zone.get(row['zone_id'], 0.0) + thickness / E
        C_layer = np.array([by_zone.get(z, 0.0) for z in zones], dtype=float) / area

    # Rough contact: stage-pressure dependent softened compliance.
    rough = tables['rough_contact']  # type: ignore[index]
    C_rough = np.zeros(n)
    if not rough.empty and settings.use_rough:
        t = rough.copy()
        if 'active' in t.columns:
            t = t[t['active'].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active'])]
        rough_map: dict[str, float] = {}
        for _, row in t.iterrows():
            c_ref = float(row.get('c_ref_mm_per_N', 0.0))
            p_ref = max(float(row.get('p_ref_MPa', p_stage)), 1e-6)
            exponent = float(row.get('exponent', 0.0))
            factor = float(row.get('roughness_factor', 1.0))
            rough_map[row['zone_id']] = rough_map.get(row['zone_id'], 0.0) + c_ref * (p_ref / p_stage) ** exponent * factor
        C_rough = np.array([rough_map.get(z, 0.0) for z in zones], dtype=float)

    # Local indentation / CFRP skin indentation.
    indent = tables['local_indent']  # type: ignore[index]
    C_indent = np.zeros(n)
    if not indent.empty and settings.use_indent:
        t = indent.copy()
        if 'active' in t.columns:
            t = t[t['active'].astype(str).str.lower().isin(['1', 'true', 'yes', 'y', 'active'])]
        ind_map: dict[str, float] = {}
        for _, row in t.iterrows():
            c0 = float(row.get('C_indent_mm_per_N', 0.0))
            exp = float(row.get('load_softening_exponent', 0.0))
            # Higher pressure generally increases tangent stiffness; represent it by reducing C.
            ind_map[row['zone_id']] = ind_map.get(row['zone_id'], 0.0) + c0 / max((p_stage / 0.05) ** exp, 1e-6)
        C_indent = np.array([ind_map.get(z, 0.0) for z in zones], dtype=float)

    fixture = tables['fixture_joint']  # type: ignore[index]
    C_locator = _fixture_influence(pkg, fixture, stage_id, 'locator') if settings.use_locator else np.zeros(n)
    C_clamp = _fixture_influence(pkg, fixture, stage_id, 'clamp') if settings.use_clamp else np.zeros(n)
    C_joint = _fixture_influence(pkg, fixture, stage_id, 'joint') if settings.use_joint else np.zeros(n)

    if not settings.use_partition:
        C_partition[:] = 0.0

    C_partition = C_partition * settings.scale_partition
    C_layer = C_layer * settings.scale_layer
    C_rough = C_rough * settings.scale_rough
    C_indent = C_indent * settings.scale_indent
    C_locator = C_locator * settings.scale_locator
    C_clamp = C_clamp * settings.scale_clamp
    C_joint = C_joint * settings.scale_joint

    Cn_substitution = settings.global_scale * (C_partition + C_layer + C_rough + C_indent + C_locator + C_clamp + C_joint)
    release_offset = _release_rebound_vector(pkg, tables, settings)

    out = cp.copy()
    out['zone_id'] = zones.to_numpy()
    out['stage_id'] = stage_id
    out['stage_load_estimate_N'] = stage_load
    out['stage_pressure_estimate_MPa'] = p_stage
    out['base_Cn_from_package_mm_per_N'] = base_diag
    out['C_partition_mm_per_N'] = C_partition
    out['C_layer_mm_per_N'] = C_layer
    out['C_rough_mm_per_N'] = C_rough
    out['C_indent_mm_per_N'] = C_indent
    out['C_locator_mm_per_N'] = C_locator
    out['C_clamp_mm_per_N'] = C_clamp
    out['C_joint_mm_per_N'] = C_joint
    out['Cn_substitution_mm_per_N'] = Cn_substitution
    out['Ct_equiv_mm_per_N'] = Ct
    out['mu_eff'] = mu
    out['beta_r'] = beta
    out['release_rebound_increment_mm'] = release_offset
    out['quality_flag'] = 'PASS'
    return out


def assemble_Cn_matrix(
    pkg: SMSPackage,
    stage_id: str,
    cn_scale: float = 1.0,
    settings: NumericalSubstitutionSettings | dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    settings = NumericalSubstitutionSettings.from_dict(settings) if not isinstance(settings, NumericalSubstitutionSettings) else settings
    base_Cn = cn_scale * pkg.matrices['Cn_local']
    comp = compute_substitution_components(pkg, stage_id, settings)
    if not settings.enabled or settings.mode == 'base_only':
        comp['Cn_runtime_mm_per_N'] = np.diag(base_Cn)
        return base_Cn, comp

    subst_diag = comp['Cn_substitution_mm_per_N'].to_numpy(float)
    if settings.mode == 'replace':
        has_replacement_data = bool(np.any(np.isfinite(subst_diag) & (subst_diag > 0.0)))
        if has_replacement_data:
            Cn = np.diag(cn_scale * subst_diag)
            comp['substitution_fallback_flag'] = False
            comp['substitution_fallback_reason'] = ''
        else:
            # Formal V2.5 packages do not necessarily include the optional legacy
            # I_substitution extension. Never replace a valid package Cn with zeros.
            Cn = base_Cn
            comp['substitution_fallback_flag'] = True
            comp['substitution_fallback_reason'] = 'replace requested but no positive substitution compliance; using package Cn_local'
    elif settings.mode == 'add':
        Cn = base_Cn + np.diag(cn_scale * subst_diag)
    else:
        # Unknown mode: fail safe to base only.
        Cn = base_Cn
    if 'substitution_fallback_flag' not in comp.columns:
        comp['substitution_fallback_flag'] = False
        comp['substitution_fallback_reason'] = ''
    comp['Cn_runtime_mm_per_N'] = np.diag(Cn)
    return Cn, comp


def release_rebound_increment(
    pkg: SMSPackage,
    stage_id: str,
    settings: NumericalSubstitutionSettings | dict[str, Any] | None = None,
) -> np.ndarray:
    settings = NumericalSubstitutionSettings.from_dict(settings) if not isinstance(settings, NumericalSubstitutionSettings) else settings
    if stage_id != 'S_RELEASE_04' or not settings.enabled or not settings.use_release:
        return np.zeros(len(pkg.contact_points), dtype=float)
    tables = load_substitution_config(pkg)
    return _release_rebound_vector(pkg, tables, settings)


def module_summary_table(components: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'C_partition_mm_per_N', 'C_layer_mm_per_N', 'C_rough_mm_per_N', 'C_indent_mm_per_N',
        'C_locator_mm_per_N', 'C_clamp_mm_per_N', 'C_joint_mm_per_N', 'Cn_substitution_mm_per_N',
        'release_rebound_increment_mm'
    ]
    rows = []
    for c in cols:
        if c in components.columns:
            s = pd.to_numeric(components[c], errors='coerce')
            rows.append({
                'module_component': c,
                'min': float(s.min()),
                'mean': float(s.mean()),
                'max': float(s.max()),
                'sum': float(s.sum()),
            })
    return pd.DataFrame(rows)


def quality_gate_substitution(pkg: SMSPackage, components: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    base = components['base_Cn_from_package_mm_per_N'].to_numpy(float)
    subst = components['Cn_substitution_mm_per_N'].to_numpy(float)
    if 'Cn_runtime_mm_per_N' in components.columns:
        runtime = components['Cn_runtime_mm_per_N'].to_numpy(float)
    elif mode == 'replace':
        runtime = subst
    elif mode == 'add':
        runtime = base + subst
    else:
        runtime = base

    rows.append({'check_item': 'substitution mode', 'status': 'PASS', 'detail': mode})
    fallback = bool(components.get('substitution_fallback_flag', pd.Series([False])).astype(bool).any())
    rows.append({'check_item': 'substitution input availability', 'status': 'WARN' if fallback else 'PASS', 'detail': str(components.get('substitution_fallback_reason', pd.Series([''])).iloc[0]) if fallback else 'replacement/additional parameters available'})
    rows.append({'check_item': 'substitution nonnegative', 'status': 'PASS' if np.all(subst >= -1e-15) else 'FAIL', 'detail': f'min={subst.min():.3e}'})
    rows.append({'check_item': 'runtime Cn nonnegative', 'status': 'PASS' if np.all(runtime >= -1e-15) else 'FAIL', 'detail': f'min={runtime.min():.3e}'})
    if mode == 'add':
        ratio = np.divide(subst, np.maximum(base, 1e-15))
        warn = bool(np.nanmax(ratio) > 2.0)
        rows.append({'check_item': 'double-count warning', 'status': 'WARN' if warn else 'PASS', 'detail': f'max(substitution/base)={np.nanmax(ratio):.2f}; add模式要确认未与原始Cn重复'})
    else:
        rows.append({'check_item': 'double-count warning', 'status': 'PASS', 'detail': 'replace/base_only 模式不叠加原始Cn'})
    rows.append({'check_item': 'mu range', 'status': 'PASS' if np.all((components['mu_eff'] >= 0) & (components['mu_eff'] <= 1.5)) else 'WARN', 'detail': f"mu=[{components['mu_eff'].min():.3f},{components['mu_eff'].max():.3f}]"})
    return pd.DataFrame(rows)
