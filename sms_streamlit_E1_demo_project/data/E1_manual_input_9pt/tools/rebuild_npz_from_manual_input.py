from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manual = pd.read_csv(ROOT / 'manual_input_table.csv')

stage_ids = ['S_LOCATE_01', 'S_CLAMP_02', 'S_JOIN_03', 'S_RELEASE_04']
nominal_gap = manual['nominal_gap_mm'].to_numpy(float)
sms_component = manual['sms_component_mm'].to_numpy(float)
g0 = nominal_gap + sms_component
pose_component = np.zeros_like(g0)
area = manual['area_weight'].to_numpy(float)
QA = np.diag(area)
Cn_local = np.diag(manual['Cn_local_diag_mm_per_N'].to_numpy(float))
Bn_mapping = np.eye(len(g0))
Bt_mapping = np.zeros((2*len(g0), len(g0)))
G_gap_mapping = np.eye(len(g0))

# update gap_field.csv
gap = pd.read_csv(ROOT / 'I_Gamma' / 'gap_field.csv')
gap['nominal_component'] = nominal_gap
gap['sms_component'] = sms_component
gap['values_g'] = g0
gap.to_csv(ROOT / 'I_Gamma' / 'gap_field.csv', index=False)

payload = {
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
for sid in stage_ids:
    u = manual[f'u_free_{sid}_mm'].to_numpy(float)
    diag = manual[f'W_struct_diag_{sid}_mm_per_N'].to_numpy(float)
    W = np.diag(diag)
    # small symmetric coupling for neighboring points; leave diagonal dominant
    coords = manual[['x_i0','y_i0']].to_numpy(float)
    for i in range(len(g0)):
        for j in range(i+1, len(g0)):
            dist = np.linalg.norm(coords[i]-coords[j])
            val = min(diag[i], diag[j]) * 0.035 * np.exp(-dist/30.0)
            W[i,j] = W[j,i] = val
    for i in range(len(g0)):
        W[i,i] = max(W[i,i], np.sum(np.abs(W[i])) - abs(W[i,i]) + 1e-7)
    payload[f'u_free__{sid}'] = u
    payload[f'W_struct__{sid}'] = W
    payload[f'q__{sid}'] = g0 - u

np.savez(ROOT / 'matrices' / 'E1_matrices.npz', **payload)

stage_plan = pd.read_csv(ROOT / 'I_stage' / 'stage_plan.csv')
for sid in stage_ids:
    stage_plan.loc[stage_plan['stage_id'] == sid, 'free_closure_mean_mm'] = float(np.mean(payload[f'u_free__{sid}']))
stage_plan.to_csv(ROOT / 'I_stage' / 'stage_plan.csv', index=False)
print('已更新：I_Gamma/gap_field.csv, I_stage/stage_plan.csv, matrices/E1_matrices.npz')
