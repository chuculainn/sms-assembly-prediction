from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_ORDER = ['S_LOCATE_01', 'S_CLAMP_02', 'S_JOIN_03', 'S_RELEASE_04']


def parse_literal(value: Any, default: Any = None) -> Any:
    """Parse JSON/Python-list-like cells used by the V2.5 CSV templates.

    The V2.5 blank package stores many vectors as compact strings such as
    ``"[0.0,0.0,1.0]"``.  This helper returns Python lists/scalars while staying
    tolerant of empty/default placeholder cells.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return value
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none', 'null'}:
        return default
    try:
        return ast.literal_eval(text)
    except Exception:
        # Fallback for semicolon/pipe separated identifiers.  Numerical vectors
        # should normally be valid list strings, so this only supports metadata.
        if ';' in text:
            return [x.strip() for x in text.split(';') if x.strip()]
        if '|' in text:
            return [x.strip() for x in text.split('|') if x.strip()]
        return text


def to_vector(value: Any, length: int | None = None, default: float = 0.0) -> np.ndarray:
    parsed = parse_literal(value, default=None)
    if parsed is None:
        arr = np.array([], dtype=float)
    elif isinstance(parsed, np.ndarray):
        arr = parsed.astype(float).reshape(-1)
    elif isinstance(parsed, (list, tuple)):
        # Flatten one level only for scalar vectors.
        if parsed and isinstance(parsed[0], (list, tuple, np.ndarray)):
            arr = np.asarray(parsed, dtype=float).reshape(-1)
        else:
            arr = np.asarray(parsed, dtype=float).reshape(-1)
    else:
        try:
            arr = np.asarray([float(parsed)], dtype=float)
        except Exception:
            arr = np.array([], dtype=float)
    if length is not None:
        if arr.size == length:
            return arr
        if arr.size == 1:
            return np.full(length, float(arr[0]), dtype=float)
        if arr.size == 0:
            return np.full(length, default, dtype=float)
        out = np.full(length, default, dtype=float)
        n = min(length, arr.size)
        out[:n] = arr[:n]
        return out
    return arr


def to_matrix(value: Any, shape: tuple[int, int] | None = None, default: float = 0.0) -> np.ndarray:
    parsed = parse_literal(value, default=None)
    if parsed is None:
        arr = np.empty((0, 0), dtype=float)
    else:
        try:
            arr = np.asarray(parsed, dtype=float)
        except Exception:
            arr = np.empty((0, 0), dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if shape is not None and arr.shape != shape:
        out = np.full(shape, default, dtype=float)
        r = min(shape[0], arr.shape[0])
        c = min(shape[1], arr.shape[1])
        if r and c:
            out[:r, :c] = arr[:r, :c]
        return out
    return arr.astype(float)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding='utf-8-sig')


def read_json_optional(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def detect_package_type(root: str | Path) -> str:
    root = Path(root)
    manifest_path = root / 'package_manifest.json'
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = read_json_optional(manifest_path)  # type: ignore[assignment]
        except Exception:
            manifest = {}
    schema = str(manifest.get('schema_version', '')).upper()
    package_name = str(manifest.get('package_name', root.name))
    if (root / 'I0' / 'part.csv').exists() and (root / 'matrices' / 'multi_part_matrices.npz').exists():
        return 'V25_MULTI_PART'
    if (root / 'I0' / 'part.csv').exists() and (root / 'matrices' / 'default_matrices.npz').exists():
        if schema.startswith('V2.5') or package_name == '01_DEFAULT_MIN_CASE':
            return 'V25_DEFAULT_MIN_CASE' if package_name == '01_DEFAULT_MIN_CASE' or root.name == '01_DEFAULT_MIN_CASE' else 'V25_STANDARD'
        return 'V25_STANDARD'
    if (root / 'I0' / 'part_table.csv').exists() and (root / 'matrices' / 'E1_matrices.npz').exists():
        return 'E1_LEGACY'
    if (root / 'manual_input_table.csv').exists():
        return 'E1_LEGACY'
    return 'UNKNOWN'


def scan_package_files(root: str | Path) -> pd.DataFrame:
    """Return a file-level read overview for CSV/JSON/NPZ files."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame([{
            'relative_path': '.', 'kind': 'directory', 'status': 'MISSING',
            'rows': np.nan, 'fields': '', 'matrix_keys': '', 'message': f'{root} not found'
        }])
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.csv', '.json', '.npz'}:
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower().lstrip('.')
        row: dict[str, Any] = {
            'relative_path': rel,
            'kind': suffix.upper(),
            'status': 'PENDING',
            'rows': np.nan,
            'fields': '',
            'matrix_keys': '',
            'message': '',
        }
        try:
            if suffix == 'csv':
                df = pd.read_csv(path, encoding='utf-8-sig')
                row.update({
                    'status': 'PASS',
                    'rows': int(len(df)),
                    'fields': ', '.join(map(str, df.columns.tolist())),
                    'message': 'CSV读取成功' if len(df.columns) else 'CSV无表头',
                })
            elif suffix == 'json':
                data = read_json_optional(path)
                if isinstance(data, dict):
                    fields = ', '.join(map(str, data.keys()))
                    n = len(data)
                elif isinstance(data, list):
                    fields = 'list_items'
                    n = len(data)
                else:
                    fields = type(data).__name__
                    n = np.nan
                row.update({'status': 'PASS', 'rows': n, 'fields': fields, 'message': 'JSON读取成功'})
            elif suffix == 'npz':
                z = np.load(path, allow_pickle=False)
                matrix_items = [f'{k}{tuple(z[k].shape)}' for k in z.files]
                row.update({
                    'status': 'PASS',
                    'rows': len(z.files),
                    'fields': 'npz_keys',
                    'matrix_keys': ', '.join(matrix_items),
                    'message': 'NPZ读取成功',
                })
                z.close()
        except Exception as exc:
            row.update({'status': 'FAIL', 'message': f'{type(exc).__name__}: {exc}'})
        rows.append(row)
    return pd.DataFrame(rows)


def load_all_raw_tables(root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    root = Path(root)
    tables: dict[str, pd.DataFrame] = {}
    jsons: dict[str, Any] = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == '.csv':
            try:
                tables[rel] = pd.read_csv(path, encoding='utf-8-sig')
            except Exception:
                tables[rel] = pd.DataFrame()
        elif path.suffix.lower() == '.json':
            try:
                jsons[rel] = read_json_optional(path)
            except Exception as exc:
                jsons[rel] = {'_read_error': f'{type(exc).__name__}: {exc}'}
    return tables, jsons


def _first_table(tables: dict[str, pd.DataFrame], *names: str) -> pd.DataFrame:
    for name in names:
        df = tables.get(name)
        if df is not None and not df.empty:
            return df.copy()
    # Return header-only if available, otherwise empty.
    for name in names:
        if name in tables:
            return tables[name].copy()
    return pd.DataFrame()


def normalize_v25_contact_points(cp_raw: pd.DataFrame) -> pd.DataFrame:
    if cp_raw.empty:
        return pd.DataFrame(columns=['candidate_id', 'local_index', 'x_i0', 'y_i0', 'z_i0', 'x_j0', 'y_j0', 'z_j0', 'normal_nx', 'normal_ny', 'normal_nz', 'area_weight'])
    out = cp_raw.copy()
    out['candidate_id'] = out.get('point_id', out.index.astype(str)).astype(str)
    
    local_index_raw = pd.to_numeric(out.get('local_index', pd.Series(range(len(out)), index=out.index)), errors='coerce')
    out['local_index'] = local_index_raw.fillna(pd.Series(range(len(out)), index=out.index)).astype(int)
    for base, prefix in [('x_i0', 'x_i0'), ('x_j0', 'x_j0')]:
        values = [to_vector(v, length=3, default=0.0) for v in out.get(base, pd.Series(['[0,0,0]'] * len(out)))]
        arr = np.vstack(values) if values else np.zeros((0, 3))
        out[f'{prefix}_vec'] = list(arr)
        out[prefix] = arr[:, 0] if len(arr) else []
        out[f'y_{prefix.split("_")[1]}0'] = arr[:, 1] if len(arr) else []
        out[f'z_{prefix.split("_")[1]}0'] = arr[:, 2] if len(arr) else []
    normals = [to_vector(v, length=3, default=0.0) for v in out.get('normal_n', pd.Series(['[0,0,1]'] * len(out)))]
    n_arr = np.vstack(normals) if normals else np.zeros((0, 3))
    if len(n_arr):
        out['normal_nx'] = n_arr[:, 0]
        out['normal_ny'] = n_arr[:, 1]
        out['normal_nz'] = n_arr[:, 2]
    if 'area_weight' not in out.columns:
        out['area_weight'] = 1.0
    out['area_weight'] = pd.to_numeric(out['area_weight'], errors='coerce').fillna(1.0).astype(float)
    if 'edge_or_interior_flag' not in out.columns:
        out['edge_or_interior_flag'] = 'INTERIOR'
    if 'candidate_flag' not in out.columns:
        out['candidate_flag'] = True
    if 'correspondence_quality' not in out.columns:
        out['correspondence_quality'] = 'WARN'
    # The app expects y_i0/y_j0 naming exactly.
    if 'y_i0' not in out.columns and 'y_i00' in out.columns:
        out = out.rename(columns={'y_i00': 'y_i0'})
    if 'z_i0' not in out.columns and 'z_i00' in out.columns:
        out = out.rename(columns={'z_i00': 'z_i0'})
    if 'y_j0' not in out.columns and 'y_j00' in out.columns:
        out = out.rename(columns={'y_j00': 'y_j0'})
    if 'z_j0' not in out.columns and 'z_j00' in out.columns:
        out = out.rename(columns={'z_j00': 'z_j0'})
    return out


def normalize_v25_gap_field(gap_raw: pd.DataFrame, contact_points: pd.DataFrame, matrices_raw: dict[str, np.ndarray]) -> pd.DataFrame:
    m = len(contact_points)
    if m == 0:
        return pd.DataFrame(columns=['candidate_id', 'values_g', 'nominal_component', 'sms_component', 'pose_bias_component_optional'])
    if gap_raw.empty:
        values = to_vector(matrices_raw.get('GAP_S_RELEASE_04', np.zeros(m)), length=m, default=0.0)
        nominal = values.copy()
        sms = np.zeros(m)
        pose = np.zeros(m)
        meta = {}
    elif len(gap_raw) == 1:
        row = gap_raw.iloc[0]
        values = to_vector(row.get('values_g', None), length=m, default=0.0)
        nominal = to_vector(row.get('nominal_component', None), length=m, default=float(values[0] if values.size else 0.0))
        sms = to_vector(row.get('sms_component', None), length=m, default=0.0)
        pose = to_vector(row.get('pose_bias_component_optional', None), length=m, default=0.0)
        meta = row.to_dict()
    else:
        # Multi-interface packages store one compact vector per ContactDomain.
        # Expand those vectors according to ContactPoint.local_index so the
        # global vector order is explicit and independent of CSV row order.
        values = np.zeros(m, dtype=float)
        nominal = np.zeros(m, dtype=float)
        sms = np.zeros(m, dtype=float)
        pose = np.zeros(m, dtype=float)
        meta = {'gap_field_id': 'GF_GLOBAL_ASSEMBLED', 'contact_domain_id': 'MULTI_DOMAIN'}
        rows_by_domain = {
            str(row.get('contact_domain_id')): row
            for _, row in gap_raw.iterrows()
        }
        for pos, (_, cp) in enumerate(contact_points.iterrows()):
            domain_id = str(cp.get('contact_domain_id', ''))
            row = rows_by_domain.get(domain_id)
            if row is None:
                continue
            local_index = int(cp.get('local_index', 0))
            raw_values = to_vector(row.get('values_g'), length=None, default=0.0)
            raw_nominal = to_vector(row.get('nominal_component'), length=None, default=0.0)
            raw_sms = to_vector(row.get('sms_component'), length=None, default=0.0)
            raw_pose = to_vector(row.get('pose_bias_component_optional'), length=None, default=0.0)
            values[pos] = raw_values[local_index] if local_index < raw_values.size else 0.0
            nominal[pos] = raw_nominal[local_index] if local_index < raw_nominal.size else values[pos]
            sms[pos] = raw_sms[local_index] if local_index < raw_sms.size else 0.0
            pose[pos] = raw_pose[local_index] if local_index < raw_pose.size else 0.0
    out = pd.DataFrame({
        'candidate_id': contact_points['candidate_id'].to_numpy(),
        'gap_field_id': meta.get('gap_field_id', 'GF_INITIAL_DEFAULT'),
        'contact_domain_id': meta.get('contact_domain_id', contact_points.get('contact_domain_id', pd.Series(['CD_DEFAULT'])).iloc[0] if len(contact_points) else 'CD_DEFAULT'),
        'sample_id': meta.get('sample_id', 'SAMPLE_001'),
        'stage_id_or_initial': meta.get('stage_id_or_initial', 'INITIAL'),
        'values_g': values,
        'nominal_component': nominal,
        'sms_component': sms,
        'pose_bias_component_optional': pose,
        'sign_convention': meta.get('sign_convention', 'g>0 separation; g=0 contact; g<0 initial interference'),
        'source_sms_update_result_ids': meta.get('source_sms_update_result_ids', ''),
        'double_count_check_id': meta.get('double_count_check_id', ''),
        'reference_state_id': meta.get('reference_state_id', 'REF_NOMINAL'),
        'unit': meta.get('unit', 'mm'),
    })
    return out


def normalize_v25_stage_plan(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stage_def = _first_table(tables, 'I0/stage_definition.csv')
    if not stage_def.empty and 'stage_id' in stage_def.columns:
        out = stage_def.copy()
        out['operation_order'] = pd.to_numeric(
            out.get('stage_order', pd.Series(range(1, len(out) + 1), index=out.index)),
            errors='coerce',
        ).fillna(pd.Series(range(1, len(out) + 1), index=out.index)).astype(int)
        out['operation_type'] = out.get('stage_type', out['stage_id']).astype(str)
        out['stage_name'] = out['operation_type']
        return out[['stage_id', 'stage_name', 'operation_type', 'operation_order'] + [
            c for c in out.columns if c not in {'stage_id', 'stage_name', 'operation_type', 'operation_order'}
        ]].sort_values('operation_order').reset_index(drop=True)
    topo = _first_table(tables, 'I0/assembly_topology.csv')
    if not topo.empty and {'stage_id', 'operation_type'} <= set(topo.columns):
        out = topo.copy()
        if 'operation_order' not in out.columns:
            
            order_raw = pd.to_numeric(out.get('assembly_step', pd.Series(range(1, len(out) + 1), index=out.index)), errors='coerce')
            out['operation_order'] = order_raw.fillna(pd.Series(range(1, len(out) + 1), index=out.index)).astype(int)
        if 'stage_name' not in out.columns:
            out['stage_name'] = out['operation_type']
        out = out.sort_values('operation_order').drop_duplicates('stage_id', keep='last')
        out['operation_order'] = range(1, len(out) + 1)
        return out[['stage_id', 'stage_name', 'operation_type', 'operation_order'] + [c for c in out.columns if c not in {'stage_id','stage_name','operation_type','operation_order'}]].copy()
    stg = _first_table(tables, 'I_stage/stage_input.csv')
    if not stg.empty and 'stage_id' in stg.columns:
        out = stg[['stage_id']].copy()
        out['operation_order'] = range(1, len(out) + 1)
        out['operation_type'] = out['stage_id'].astype(str).str.extract(r'S_([^_]+)', expand=False).fillna(out['stage_id'])
        out['stage_name'] = out['operation_type']
        return out
    return pd.DataFrame({'stage_id': STAGE_ORDER, 'stage_name': ['LOCATE','CLAMP','JOIN','RELEASE'], 'operation_type': ['LOCATE','CLAMP','JOIN','RELEASE'], 'operation_order': [1,2,3,4]})


def normalize_v25_kcp_kcm(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    kcp = _first_table(tables, 'I_key/kcp_definition.csv')
    if not kcp.empty:
        df = kcp.copy()
        df['feature_id'] = df.get('kcp_id', df.index.astype(str))
        df['feature_role'] = 'KCP'
        df['target_part_or_interface'] = df.get('target_object_id', '')
        df['measurement_method'] = df.get('acceptance_rule', '')
        df['update_target'] = 'KCP_VALIDATION'
        if 'description' not in df.columns:
            df['description'] = df.get('kcp_name', '')
        rows.append(df)
    kcm = _first_table(tables, 'I_key/kcm_definition.csv')
    if not kcm.empty:
        df = kcm.copy()
        df['feature_id'] = df.get('kcm_id', df.index.astype(str))
        df['feature_role'] = 'KCM'
        df['target_part_or_interface'] = df.get('target_object_id', '')
        if 'lower_tol' not in df.columns:
            df['lower_tol'] = np.nan
        if 'upper_tol' not in df.columns:
            df['upper_tol'] = np.nan
        if 'description' not in df.columns:
            df['description'] = df.get('kcm_name', '')
        rows.append(df)
    if rows:
        out = pd.concat(rows, ignore_index=True, sort=False)
        for col in ['nominal_value', 'lower_tol', 'upper_tol']:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        return out
    return pd.DataFrame(columns=['feature_id', 'feature_role', 'feature_type', 'nominal_value', 'lower_tol', 'upper_tol', 'description'])


def normalize_v25_validation(tables: dict[str, pd.DataFrame], matrices: dict[str, np.ndarray], kcp_kcm: pd.DataFrame) -> pd.DataFrame:
    kcp_defs = kcp_kcm[kcp_kcm.get('feature_role', '') == 'KCP'].copy()
    if kcp_defs.empty:
        return pd.DataFrame(columns=['kcp_id', 'measured_value', 'uncertainty', 'data_role'])
    # Prefer explicit validation_result reference_values; fall back to default release gap for the min case.
    val_raw = _first_table(tables, 'validation/validation_result.csv')
    rows = []
    for i, (_, row) in enumerate(kcp_defs.iterrows()):
        kcp_id = str(row.get('feature_id'))
        measured = np.nan
        if not val_raw.empty and 'reference_values' in val_raw.columns:
            ref = to_vector(val_raw.iloc[0].get('reference_values'), length=None, default=np.nan)
            if ref.size > i and np.isfinite(ref[i]):
                measured = float(ref[i])
        if not np.isfinite(measured):
            # For 01_DEFAULT_MIN_CASE, the only meaningful placeholder KCP is the release gap.
            gap = matrices.get('GAP_S_RELEASE_04', matrices.get('g0', np.array([0.0])))
            measured = float(np.asarray(gap).reshape(-1)[0]) if np.asarray(gap).size else 0.0
        rows.append({
            'kcp_id': kcp_id,
            'measured_value': measured,
            'uncertainty': float(row.get('required_uncertainty', 0.01) if pd.notna(row.get('required_uncertainty', np.nan)) else 0.01),
            'data_role': 'VALIDATE',
        })
    return pd.DataFrame(rows)


def _v25_npz_path(root: Path) -> Path:
    for name in ('multi_part_matrices.npz', 'default_matrices.npz'):
        path = root / 'matrices' / name
        if path.exists():
            return path
    return root / 'matrices' / 'default_matrices.npz'


def _stage_npz_suffix(stage_id: str, multi_part: bool) -> str:
    return stage_id.removeprefix('S_') if multi_part else stage_id


def _assemble_multi_part_bn(raw: dict[str, np.ndarray], m: int) -> np.ndarray:
    blocks = [np.asarray(raw[k], dtype=float) for k in raw if k.startswith('BN_') and k != 'BN_DEFAULT']
    compatible = [block for block in blocks if block.ndim == 2 and block.shape[1] == m]
    return np.vstack(compatible) if compatible and sum(block.shape[0] for block in compatible) == m else np.eye(m)


def build_v25_matrices(root: Path, stage_plan: pd.DataFrame, gap_field: pd.DataFrame, contact_points: pd.DataFrame) -> dict[str, np.ndarray]:
    npz_path = _v25_npz_path(root)
    multi_part = npz_path.name == 'multi_part_matrices.npz'
    raw: dict[str, np.ndarray] = {}
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        raw = {k: z[k].copy() for k in z.files}
        z.close()
    matrices: dict[str, np.ndarray] = dict(raw)
    m = len(contact_points)
    g0 = gap_field['values_g'].to_numpy(dtype=float) if 'values_g' in gap_field.columns else to_vector(raw.get('GAP_S_RELEASE_04'), m, 0.0)
    nominal = gap_field['nominal_component'].to_numpy(dtype=float) if 'nominal_component' in gap_field.columns else g0.copy()
    sms = gap_field['sms_component'].to_numpy(dtype=float) if 'sms_component' in gap_field.columns else np.zeros(m)
    pose = gap_field['pose_bias_component_optional'].to_numpy(dtype=float) if 'pose_bias_component_optional' in gap_field.columns else np.zeros(m)
    matrices.update({
        'g0': to_vector(g0, m, 0.0),
        'nominal_gap': to_vector(nominal, m, 0.0),
        'sms_component': to_vector(sms, m, 0.0),
        'pose_component': to_vector(pose, m, 0.0),
        'G_gap_mapping': raw.get('G_GAP_DEFAULT', np.eye(m)),
        'Bn_mapping': raw.get('BN_DEFAULT', _assemble_multi_part_bn(raw, m)),
        'Bt_mapping': raw.get('BT_DEFAULT', np.zeros((2 * m, m))),
        'QA': raw.get('QA_ALL', raw.get('QA_MATRIX_DEFAULT', np.diag(contact_points['area_weight'].to_numpy(float) if m else []))),
        'Cn_local': raw.get('CN_ALL', raw.get('CN_DEFAULT', np.eye(m) * 1e-6)),
    })
    # Ensure core matrix dimensions are consistent with contact point count.
    for key, default in [('G_gap_mapping', np.eye(m)), ('Bn_mapping', np.eye(m)), ('QA', np.eye(m)), ('Cn_local', np.eye(m) * 1e-6)]:
        matrices[key] = np.asarray(matrices[key], dtype=float)
        if matrices[key].shape != (m, m):
            matrices[key] = default.astype(float)
    if matrices['Bt_mapping'].shape != (2 * m, m):
        matrices['Bt_mapping'] = np.zeros((2 * m, m), dtype=float)
    for sid in stage_plan['stage_id'].astype(str).tolist():
        suffix = _stage_npz_suffix(sid, multi_part)
        w_key = f'W_STRUCT_{suffix}'
        wt_key = f'W_TOTAL_{suffix}'
        q_key = f'Q_{suffix}'
        u_key = f'U_FREE_{suffix}'
        W_struct = np.asarray(raw.get(w_key, np.eye(m) * 1e-3), dtype=float)
        if W_struct.shape != (m, m):
            W_struct = np.eye(m) * 1e-3
        q_default = to_vector(raw.get(q_key, matrices['g0']), m, 0.0)
        # Current solver reconstructs q as g0_runtime - u_free.  Store u_free so that
        # at default scale q_runtime equals the V2.5 q vector.
        u_free = to_vector(raw.get(u_key, matrices['g0'] - q_default), m, 0.0)
        matrices[f'W_struct__{sid}'] = W_struct
        matrices[f'W_total__{sid}'] = np.asarray(raw.get(wt_key, W_struct + matrices['Cn_local']), dtype=float)
        if matrices[f'W_total__{sid}'].shape != (m, m):
            matrices[f'W_total__{sid}'] = W_struct + matrices['Cn_local']
        matrices[f'u_free__{sid}'] = u_free
        matrices[f'q__{sid}'] = q_default
    return matrices


def adapt_v25_package(root: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    tables, jsons = load_all_raw_tables(root)
    manifest = manifest or read_json_optional(root / 'package_manifest.json')  # type: ignore[assignment]

    parts = _first_table(tables, 'I0/part.csv')
    interfaces = _first_table(tables, 'I0/interface.csv')
    contact_points = normalize_v25_contact_points(_first_table(tables, 'I_Gamma/contact_point.csv'))
    contact_domains = _first_table(tables, 'I_Gamma/contact_domain.csv')
    if (
        not contact_points.empty
        and not contact_domains.empty
        and {'contact_domain_id', 'interface_id'} <= set(contact_domains.columns)
    ):
        domain_map = contact_domains[['contact_domain_id', 'interface_id']].drop_duplicates('contact_domain_id')
        contact_points = contact_points.merge(domain_map, on='contact_domain_id', how='left')
    # Need matrices for gap fallback, but gap also contributes to matrices. Load raw here first.
    raw_npz: dict[str, np.ndarray] = {}
    npz_path = _v25_npz_path(root)
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        raw_npz = {k: z[k].copy() for k in z.files}
        z.close()
    gap_field = normalize_v25_gap_field(_first_table(tables, 'I_Gamma/gap_field.csv'), contact_points, raw_npz)
    stage_plan = normalize_v25_stage_plan(tables)
    matrices = build_v25_matrices(root, stage_plan, gap_field, contact_points)

    kcp_kcm = normalize_v25_kcp_kcm(tables)
    validation_kcp = normalize_v25_validation(tables, matrices, kcp_kcm)
    condensed_operator = _first_table(tables, 'I_red/condensed_operator.csv')
    interface_parameters = _first_table(tables, 'parameter_library/interface_parameter.csv', 'I_Gamma/interface_parameter.csv')

    load_items = _first_table(tables, 'I_stage/load_item.csv')
    if not load_items.empty:
        process_record = load_items.copy()
        process_record['measurement_type'] = process_record.get('load_type', 'load')
        process_record['value'] = process_record.get('magnitude', 0.0).apply(lambda x: float(to_vector(x, length=1, default=0.0)[0]) if not isinstance(x, (int, float)) else x)
    else:
        process_record = pd.DataFrame(columns=['stage_id', 'measurement_type', 'value', 'unit'])

    return {
        'manifest': manifest,
        'parts': parts,
        'interfaces': interfaces,
        'contact_points': contact_points,
        'gap_field': gap_field,
        'interface_parameters': interface_parameters,
        'stage_plan': stage_plan.sort_values('operation_order').reset_index(drop=True),
        'process_record': process_record,
        'kcp_kcm': kcp_kcm,
        'condensed_operator': condensed_operator,
        'validation_kcp': validation_kcp,
        'matrices': matrices,
        'raw_tables': tables,
        'raw_json': jsons,
    }
