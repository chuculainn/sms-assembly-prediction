from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STAGE_IDS = ['S_LOCATE_01', 'S_CLAMP_02', 'S_JOIN_03', 'S_RELEASE_04']


def has_manual_input(root: str | Path) -> bool:
    return (Path(root) / 'manual_input_table.csv').exists()


def read_manual_input(root: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / 'manual_input_table.csv')


def save_manual_input(root: str | Path, manual: pd.DataFrame) -> None:
    root = Path(root)
    # Recompute g0 for display consistency. It is derived from nominal_gap + sms_component.
    if {'nominal_gap_mm', 'sms_component_mm'} <= set(manual.columns):
        manual['g0_mm'] = manual['nominal_gap_mm'].astype(float) + manual['sms_component_mm'].astype(float)
    manual.to_csv(root / 'manual_input_table.csv', index=False, encoding='utf-8-sig')


def rebuild_from_manual_input(root: str | Path, *, add_neighbor_coupling: bool = True) -> dict[str, str]:
    """Rebuild the small E1 matrix package from manual_input_table.csv.

    This is intentionally scoped to the E1 manual-input package. It updates the three places
    the demo solver consumes directly: I_Gamma/gap_field.csv, I_stage/stage_plan.csv and
    matrices/E1_matrices.npz.
    """
    root = Path(root)
    manual_path = root / 'manual_input_table.csv'
    if not manual_path.exists():
        raise FileNotFoundError(f'找不到手动录入表：{manual_path}')

    manual = pd.read_csv(manual_path)
    required = ['candidate_id', 'area_weight', 'nominal_gap_mm', 'sms_component_mm', 'Cn_local_diag_mm_per_N']
    for col in required:
        if col not in manual.columns:
            raise ValueError(f'手动录入表缺少字段：{col}')

    nominal_gap = manual['nominal_gap_mm'].to_numpy(float)
    sms_component = manual['sms_component_mm'].to_numpy(float)
    g0 = nominal_gap + sms_component
    pose_component = np.zeros_like(g0)
    area = manual['area_weight'].to_numpy(float)
    QA = np.diag(area)
    Cn_local = np.diag(manual['Cn_local_diag_mm_per_N'].to_numpy(float))
    n = len(g0)
    Bn_mapping = np.eye(n)
    Bt_mapping = np.zeros((2 * n, n))
    G_gap_mapping = np.eye(n)

    gap_path = root / 'I_Gamma' / 'gap_field.csv'
    gap = pd.read_csv(gap_path)
    gap['nominal_component'] = nominal_gap
    gap['sms_component'] = sms_component
    gap['values_g'] = g0
    gap.to_csv(gap_path, index=False, encoding='utf-8-sig')

    payload: dict[str, np.ndarray] = {
        'g0': g0,
        'nominal_gap': nominal_gap,
        'sms_component': sms_component,
        'pose_component': pose_component,
        'QA': QA,
        'Cn_local': Cn_local,
        'Bn_mapping': Bn_mapping,
        'Bt_mapping': Bt_mapping,
        'G_gap_mapping': G_gap_mapping,
    }

    coords = manual[['x_i0', 'y_i0']].to_numpy(float) if {'x_i0', 'y_i0'} <= set(manual.columns) else None
    for sid in STAGE_IDS:
        u_col = f'u_free_{sid}_mm'
        w_col = f'W_struct_diag_{sid}_mm_per_N'
        if u_col not in manual.columns or w_col not in manual.columns:
            raise ValueError(f'手动录入表缺少阶段字段：{u_col} 或 {w_col}')
        u = manual[u_col].to_numpy(float)
        diag = manual[w_col].to_numpy(float)
        W = np.diag(diag)
        if add_neighbor_coupling and coords is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    val = min(diag[i], diag[j]) * 0.035 * np.exp(-dist / 30.0)
                    W[i, j] = W[j, i] = val
            # keep a strictly diagonally dominant, positive matrix for the small demo
            for i in range(n):
                W[i, i] = max(W[i, i], float(np.sum(np.abs(W[i])) - abs(W[i, i]) + 1e-7))
        payload[f'u_free__{sid}'] = u
        payload[f'W_struct__{sid}'] = W
        payload[f'q__{sid}'] = g0 - u

    matrix_dir = root / 'matrices'
    matrix_dir.mkdir(exist_ok=True)
    np.savez(matrix_dir / 'E1_matrices.npz', **payload)

    stage_path = root / 'I_stage' / 'stage_plan.csv'
    stage_plan = pd.read_csv(stage_path)
    for sid in STAGE_IDS:
        stage_plan.loc[stage_plan['stage_id'] == sid, 'free_closure_mean_mm'] = float(np.mean(payload[f'u_free__{sid}']))
    stage_plan.to_csv(stage_path, index=False, encoding='utf-8-sig')

    return {
        'gap_field': str(gap_path),
        'stage_plan': str(stage_path),
        'matrix_package': str(matrix_dir / 'E1_matrices.npz'),
    }
