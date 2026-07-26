from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import json
import zipfile

import altair as alt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import load_package, detect_package_type
from core.validation import validate_package
from core.package_validator import data_truthfulness_statement, has_blocking_failures
from core.reporting import (
    build_runtime_report_zip,
    coupling_ablation_export,
    measurement_update_report_tables,
    runtime_state_lineage,
)
from core.stage_measurement_update import (
    load_measurement_checkpoints,
    validate_runtime_measurement_override,
)
from core.topology_step import (
    connection_lock_history_table,
    release_history_table,
    topology_step_contact_summary_table,
    topology_step_execution_table,
    topology_step_operator_usage_table,
    topology_step_state_lineage_table,
    topology_step_table,
    uses_precomputed_topology_operators,
    validate_topology_steps,
)
from core.stage_state import stage_transition_runtime_table
from core.stage_solver import run_all_stages, point_result_table, stage_summary_table, build_stage_vectors
from core.kcp import extract_kcp, compare_validation
from core.monte_carlo import distribution_defaults, run_monte_carlo, one_factor_sweep
from core.manual_input import has_manual_input, read_manual_input, save_manual_input, rebuild_from_manual_input
from core.numerical_substitution import (
    NumericalSubstitutionSettings,
    compute_substitution_components,
    module_summary_table,
    quality_gate_substitution,
    load_substitution_config,
)
from core.sms_mapping import SMSMappingSettings, rebuild_gap_from_sms, fit_sms_wls_map
from core.overconstraint import (
    OverConstraintSettings,
    extended_solution_table,
    active_set_stability_trace,
)
from core.tangential_ncp import TangentialNCPSettings, tangential_summary_table
from core.fallback import FallbackSettings, evaluate_validity_and_fallback, remaining_limitations_table
from core.physical_consistency import physical_consistency_report
from core.multi_part import (
    assembly_graph,
    assembly_path_summary,
    contribution_ledger_summary,
    coupling_ablation_comparison,
    coupling_block_summary,
    interface_stage_summary,
    interface_stage_state_table,
    is_multi_part_package,
    kcp_contribution_path,
    state_lineage_summary,
    topology_summary,
    vector_layout,
)

st.set_page_config(page_title='SMS 装配接触快速预测平台', layout='wide')

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / 'data'
DEFAULT_PACKAGE_NAME = 'E1_min_closed_loop'
REPORT_ROOT = PROJECT_ROOT / 'reports'

# 只在页面展示时中文化，底层 CSV/JSON/NPZ 字段名仍保持数据结构说明书中的英文 schema。
COLUMN_ZH: dict[str, str] = {
    # 通用
    'part_id': '零件ID', 'part_name': '零件名称', 'material_type': '材料类型', 'role_in_assembly': '装配角色',
    'nominal_model_id': '名义模型ID', 'sms_prior_id': 'SMS先验ID', 'material_id': '材料ID',
    'thickness_definition_id': '厚度定义ID', 'layup_id': '铺层ID', 'local_coordinate_system_id': '局部坐标系ID',
    'mesh_or_point_set_id': '网格/点集ID', 'key_interface_ids': '关键结合面ID', 'related_kcm_ids': '关联KCM',
    'related_kcp_ids': '关联KCP', 'metadata': '元数据/备注', 'interface_id': '结合面ID', 'part_i': '零件i',
    'part_j': '零件j', 'surface_i': '表面i', 'surface_j': '表面j', 'material_pair': '材料组合',
    'nominal_gap': '名义间隙', 'contact_candidate_flag': '候选接触标志', 'key_interface_flag': '关键结合面标志',
    'revision': '版本', 'stage_id': '阶段ID', 'stage_name': '阶段名称', 'operation_type': '工艺阶段',
    'operation_order': '阶段顺序', 'unit': '单位', 'description': '说明', 'data_role': '数据用途',
    'quality_flag': '质量标志', 'source': '来源',
    'block_order': '分块顺序', 'object_id': '对象ID', 'start_index': '起始索引', 'end_index': '结束索引',
    'interface_i': '接口i', 'interface_j': '接口j', 'block_shape': '分块维度',
    'frobenius_norm': 'Frobenius范数', 'cross_interface': '是否跨接口', 'coupled_flag': '耦合非零',
    'relative_coupling_strength': '相对耦合强度', 'max_absolute_value': '最大绝对值',
    'signed_sum': '块元素和', 'coupling_sign': '耦合符号类型', 'zero_block': '是否零块',
    'contact_structural_response_norm_mm': '接触结构柔性响应范数(mm)',
    'contact_structural_response_increment_norm_mm': '接触结构柔性响应增量范数(mm)',
    'response_type': '响应类型',
    'contact_point_count': '接触点数', 'part_state_count': '零件状态数', 'interface_state_count': '接口状态数',
    'stage_state_snapshot_id': '阶段状态快照ID', 'parent_stage_state_id': '父阶段状态ID',
    'file': '文件', 'field': '字段', 'object_id': '对象ID', 'suggestion': '修改建议', 'blocking': '是否阻断求解',
    'path_type': '路径类型', 'endpoint_i': '端点i', 'endpoint_j': '端点j', 'path_rank': '路径序号',
    'part_ids': '零件路径', 'interface_ids': '接口路径', 'edge_count': '接口数', 'edge_state': '边状态',
    'parent_interface_state_id': '父接口状态ID', 'joint_lock_history_id': '连接锁定历史ID',
    'state_source': '状态来源', 'stage_state_id': '运行时阶段状态ID', 'stage_type': '阶段语义',
    'parent_stage_id': '父阶段ID', 'data_source': '数据来源', 'fallback_flag': '兼容回退标记',
    'result_role': '结果角色', 'warning_flag': '是否超过警告阈值', 'warning_threshold': '警告阈值',

    # 质量门
    'check_item': '检查项', 'status': '状态', 'detail': '说明',

    # 阶段输入
    'stage_input_id': '阶段输入ID', 'boundary_item_ids': '边界项ID', 'load_item_ids': '载荷项ID',
    'active_joint_ids': '激活连接ID', 'constraint_activation_vector': '约束激活向量',
    'load_activation_vector': '载荷激活向量', 'joint_activation_vector': '连接激活向量',
    'reduced_stiffness_id': '凝聚刚度ID', 'reduced_force_vector_id': '凝聚载荷ID',
    'inherited_state_id': '继承状态ID', 'expected_measurement_set_ids': '预期测量集ID',
    'reference_state_id': '参考状态ID', 'free_closure_mean_mm': '自由闭合均值/mm',

    # 接触点与 GapField
    'contact_domain_id': '接触域ID', 'candidate_id': '候选点ID', 'local_index': '局部序号',
    'x_i0': 'i侧x/mm', 'y_i0': 'i侧y/mm', 'z_i0': 'i侧z/mm',
    'x_j0': 'j侧x/mm', 'y_j0': 'j侧y/mm', 'z_j0': 'j侧z/mm',
    'normal_nx': '法向nx', 'normal_ny': '法向ny', 'normal_nz': '法向nz',
    'tangent1_x': '切向1x', 'tangent1_y': '切向1y', 'tangent1_z': '切向1z',
    'tangent2_x': '切向2x', 'tangent2_y': '切向2y', 'tangent2_z': '切向2z',
    'area_weight': '面积权重/mm²', 'candidate_flag': '候选标志', 'edge_or_interior_flag': '边缘/内部',
    'correspondence_quality': '对应质量', 'gap_field_id': '间隙场ID', 'sample_id': '样本ID',
    'stage_id_or_initial': '阶段/初始', 'values_g': '初始间隙g0/mm', 'nominal_component': '名义项/mm',
    'sms_component': 'SMS项/mm', 'pose_bias_component_optional': '位姿偏置项/mm', 'sign_convention': '符号约定',
    'source_sms_update_result_ids': 'SMS更新结果来源', 'double_count_check_id': '防重复检查ID',

    # 参数库/手动表
    'candidate_id': '候选点ID', 'nominal_gap_mm': '名义间隙/mm', 'sms_component_mm': 'SMS形貌项/mm',
    'g0_mm': '初始间隙g0/mm', 'Cn_local_diag_mm_per_N': '局部法向柔度Cn/mm/N',
    'u_free_S_LOCATE_01_mm': 'LOCATE自由闭合/mm', 'W_struct_diag_S_LOCATE_01_mm_per_N': 'LOCATE结构柔度对角/mm/N',
    'u_free_S_CLAMP_02_mm': 'CLAMP自由闭合/mm', 'W_struct_diag_S_CLAMP_02_mm_per_N': 'CLAMP结构柔度对角/mm/N',
    'u_free_S_JOIN_03_mm': 'JOIN自由闭合/mm', 'W_struct_diag_S_JOIN_03_mm_per_N': 'JOIN结构柔度对角/mm/N',
    'u_free_S_RELEASE_04_mm': 'RELEASE自由闭合/mm', 'W_struct_diag_S_RELEASE_04_mm_per_N': 'RELEASE结构柔度对角/mm/N',
    '说明': '说明', 'parameter_id': '参数ID', 'parameter_name': '参数名称', 'value': '数值',
    'validity': '适用域', 'C_n': '法向柔度Cn', 'C_t': '切向柔度Ct', 'mu': '摩擦系数μ', 'beta_r': '回弹系数βr',

    # 求解结果
    'active_count': '主动接触点数', 'active_ratio': '主动接触比例', 'lambda_sum_N': '接触力合计/N',
    'lambda_max_N': '最大接触力/N', 'pressure_max_MPa': '最大压力/MPa', 'gap_min_mm': '最小间隙/mm',
    'gap_mean_mm': '平均间隙/mm', 'compression_mean_mm': '平均局部压缩/mm', 'iteration_count': '迭代次数',
    'convergence_status': '收敛状态', 'gap_violation': '间隙违反量', 'force_violation': '力违反量',
    'complementarity_residual': '互补残差', 'equilibrium_residual': '平衡残差', 'q_free_gap_mm': '自由间隙q/mm',
    'gap_g_mm': '平衡间隙g/mm', 'lambda_n_N': '法向接触力λ/N', 'pressure_p_n_MPa': '接触压力p/MPa',
    'local_compression_w_n_mm': '局部压缩w/mm', 'active_flag': '是否主动接触',

    # KCP/KCM 与验证
    'feature_id': '特性ID', 'feature_role': '特性角色', 'feature_type': '特性类型',
    'target_part_or_interface': '目标零件/结合面', 'position_x': '位置x/mm', 'position_y': '位置y/mm',
    'position_z': '位置z/mm', 'direction_x': '方向x', 'direction_y': '方向y', 'direction_z': '方向z',
    'nominal_value': '名义值', 'lower_tol': '下容差', 'upper_tol': '上容差', 'measurement_method': '测量/提取方法',
    'update_target': '更新目标', 'kcp_id': 'KCP ID', 'predicted_value': '预测值', 'measured_value': '验证/实测值',
    'uncertainty': '不确定度', 'error': '误差', 'abs_error': '绝对误差', 'within_uncertainty_2sigma': '是否在2σ内',
    'sms_contribution': 'SMS贡献', 'contact_contribution': '接触贡献', 'other_contribution': '其他贡献',
    'projection_matrix_id': '投影矩阵ID',

    # Monte Carlo / 敏感性
    'sample_no': '样本序号', 'sms_scale': 'SMS倍率', 'closure_scale': '闭合/载荷倍率', 'cn_scale': 'Cn倍率',
    'mean': '均值', 'std': '标准差', 'p05': 'P05', 'p50': 'P50', 'p95': 'P95', 'min': '最小值', 'max': '最大值',
    'lower_tol': '下容差', 'upper_tol': '上容差', 'exceed_prob': '超差概率', 'validation_value': '验证值',
    'variable': '变量', 'value': '变量值',
    'sample_no': '样本编号', 'status': '样本状态', 'baseline_value': '基准预测值',
    'delta_vs_baseline': '相对基准差值', 'mc_percentile': 'Monte Carlo百分位',
    'tolerance_status': '容差状态',

    # 数值替代模块
    'module_component': '模块分量', 'C_partition_mm_per_N': '分区等效Cn/mm/N', 'C_layer_mm_per_N': '薄层/涂层Cn/mm/N',
    'C_rough_mm_per_N': '粗糙接触Cn/mm/N', 'C_indent_mm_per_N': '局部压陷Cn/mm/N',
    'C_locator_mm_per_N': '定位器附加柔度/mm/N', 'C_clamp_mm_per_N': '夹持头附加柔度/mm/N',
    'C_joint_mm_per_N': '连接区附加柔度/mm/N', 'Cn_substitution_mm_per_N': '数值替代Cn合计/mm/N',
    'Cn_runtime_mm_per_N': '本次求解Cn/mm/N', 'base_Cn_from_package_mm_per_N': '原始包Cn/mm/N',
    'Ct_equiv_mm_per_N': '等效Ct/mm/N', 'mu_eff': '等效摩擦系数μ', 'beta_r': '回弹系数βr',
    'release_rebound_increment_mm': '释放回弹增量/mm', 'zone_id': '分区ID', 'stage_load_estimate_N': '阶段载荷估计/N',
    'stage_pressure_estimate_MPa': '阶段压力估计/MPa', 'module_name': '模块名', 'valid_stage_ids': '适用阶段',
    'valid_load_min_N': '最小载荷/N', 'valid_load_max_N': '最大载荷/N', 'included_physics': '包含物理来源',
    'excluded_physics': '排除物理来源', 'check_item': '检查项', 'detail': '说明',
    'min': '最小值', 'max': '最大值', 'sum': '合计',

    # v5 SMS映射 / NCP / 过约束 / 回退
    'delta_i_sms_mm': 'i侧SMS偏差/mm', 'delta_j_sms_mm': 'j侧SMS偏差/mm',
    'sms_component_rebuilt_mm': '重建SMS间隙项/mm', 'nominal_component_mm': '名义间隙项/mm',
    'g0_rebuilt_mm': '重建g0/mm', 'g0_package_mm': '包内g0/mm', 'difference_from_package_mm': '与包内差值/mm',
    'method': '方法', 'n_points': '测点数', 'alpha_0_constant': '常数模态', 'alpha_1_x_tilt': 'x斜率模态',
    'alpha_2_y_tilt': 'y斜率模态', 'alpha_3_xy_warp': 'xy翘曲模态', 'alpha_4_quadratic': '二次模态',
    'rms_residual_mm': 'RMS残差/mm', 'max_abs_residual_mm': '最大残差/mm', 'condition_number': '条件数',
    'element_id': '元素ID', 'group_id': '组ID', 'element_type': '元素类型', 'activation_stages': '激活阶段',
    'x': 'x/mm', 'y': 'y/mm', 'z': 'z/mm', 'nx': 'nx', 'ny': 'ny', 'nz': 'nz',
    'gap0': '扩展初始间隙/mm', 'stiffness_N_per_mm': '刚度/N/mm', 'effect_radius_mm': '影响半径/mm',
    'stage_effect_mm': '阶段闭合/零偏/mm', 'q_extra_mm': '扩展自由间隙/mm', 'C_extra_mm_per_N': '扩展柔度/mm/N',
    'lambda_ext_N': '扩展接触力/N', 'gap_ext_mm': '扩展平衡间隙/mm',
    'rho_chi': '主动集稳定系数ρ', 'oscillation_flag': '振荡标志',
    't1_free_mm': '切向1自由滑移/mm', 't2_free_mm': '切向2自由滑移/mm', 'Ct_eff_mm_per_N': '切向柔度Ct/mm/N',
    'lambda_t1_N': '切向力1/N', 'lambda_t2_N': '切向力2/N', 'lambda_t_norm_N': '切向力模/N',
    'friction_limit_N': '摩擦上限/N', 'd_t1_mm': '切向位移1/mm', 'd_t2_mm': '切向位移2/mm',
    'stick_slip_state': '粘/滑状态', 'cone_residual': '摩擦锥残差',
    'overall_status': '总体状态', 'gap_violation_mm': '间隙违反量/mm', 'lambda_min_N': '最小接触力/N',
    'force_violation_N': '力违反量/N', 'complementarity_residual_Nmm': '互补残差/N·mm',
    'equilibrium_residual_mm': '平衡重构残差/mm', 'tolerance': '阈值', 'finite_numbers': '有限值检查',
    'physics_status': 'LCP物理状态', 'contact_state': '接触形态', 'contact_state_status': '接触形态提示',
    'validation_status': '验证偏差状态', 'validation_sigma': '验证偏差/σ',
    'tolerance_detail': '产品容差说明', 'validation_detail': '验证基准说明',
    'solver_convergence': '求解器收敛', 'gap_feasibility': '间隙可行性', 'force_feasibility': '接触力可行性',
    'complementarity': '互补性', 'equilibrium_reconstruction': '平衡重构', 'active_contact_count': '主动接触点数',
    'stage_pass_count': '阶段PASS数', 'stage_warn_count': '阶段WARN数', 'stage_fail_count': '阶段FAIL数',
    'kcp_warn_count': 'KCP_WARN数', 'kcp_fail_count': 'KCP_FAIL数', 'min_gap_mm': '最小间隙/mm',
    'min_lambda_N': '最小接触力/N', 'max_complementarity_residual_Nmm': '最大互补残差/N·mm',
    'max_gap_violation_mm': '最大间隙违反量/mm', 'max_force_violation_N': '最大力违反量/N',
    'fallback_action': '回退动作', 'required_data': '需要补充的数据', '模块': '模块', '剩余不足': '剩余不足', '需要数据/接口': '需要数据/接口',
}

REVERSE_COLUMN_ZH = {v: k for k, v in COLUMN_ZH.items()}


def zh_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ZH.get(c, c) for c in df.columns})


def unzh_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: REVERSE_COLUMN_ZH.get(c, c) for c in df.columns})


def discover_input_packages(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(
        [p for p in data_root.iterdir() if p.is_dir() and (p / 'package_manifest.json').exists()],
        key=lambda p: (p.name != DEFAULT_PACKAGE_NAME, p.name),
    )


def package_label(path: Path) -> str:
    try:
        manifest = json.loads((path / 'package_manifest.json').read_text(encoding='utf-8'))
        sample_id = manifest.get('sample_id', manifest.get('package_name', path.name))
        n = manifest.get('quality_summary', {}).get('candidate_count')
        package_type = detect_package_type(path)
        tag = package_type
        if n is not None:
            return f'{path.name}｜{sample_id}｜{n}点｜{tag}'
        return f'{path.name}｜{sample_id}｜{tag}'
    except Exception:
        return f'{path.name}｜{detect_package_type(path)}'


def sync_widget_value(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]


def slider_with_number(
    label: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float,
    *,
    key: str,
    fmt: str = '%.2f',
    disabled: bool = False,
) -> float:
    slider_key = f'{key}_slider'
    number_key = f'{key}_number'
    if disabled:
        st.session_state[slider_key] = default
        st.session_state[number_key] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if number_key not in st.session_state:
        st.session_state[number_key] = default

    left, right = st.columns([3, 1])
    with left:
        st.slider(label, min_value=min_value, max_value=max_value, step=step, key=slider_key,
                  on_change=sync_widget_value, args=(slider_key, number_key), disabled=disabled)
    with right:
        st.number_input('数值', min_value=min_value, max_value=max_value, step=step, format=fmt,
                        key=number_key, label_visibility='collapsed',
                        on_change=sync_widget_value, args=(number_key, slider_key),
                        disabled=disabled)
    return default if disabled else float(st.session_state[number_key])


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def stage_format(pkg, stage_id: str) -> str:
    route = topology_step_table(pkg)
    if not route.empty and 'topology_step_id' in route.columns:
        step = route.loc[route['topology_step_id'].astype(str) == str(stage_id)]
        if not step.empty:
            row = step.iloc[0]
            return f"{stage_id} - {row['operation_type']} / {row.get('assembly_cycle_id', '')}"
    row = pkg.stage_plan.loc[pkg.stage_plan['stage_id'] == stage_id]
    if row.empty:
        return stage_id
    return f"{stage_id} - {row['operation_type'].iloc[0]}"


EDGE_STATE_STYLE = {
    'NOT_ACTIVATED': ('#9ca3af', 'dot'),
    'ACTIVE_NO_CONTACT': ('#f59e0b', 'dash'),
    'ACTIVE_CONTACT': ('#16a34a', 'solid'),
    'SEPARATED': ('#dc2626', 'dot'),
    'JOIN_LOCKED': ('#7c3aed', 'solid'),
    'RETAINED_AFTER_RELEASE': ('#2563eb', 'solid'),
}


def interface_plot_geometry(graph: nx.MultiGraph, pos: dict) -> list[tuple[str, str, str, dict, list[float], list[float], tuple[float, float]]]:
    """Give parallel interfaces distinct visible midpoints without changing topology."""
    seen: dict[tuple[str, str], int] = {}
    rows = []
    for a, b, key, data in graph.edges(keys=True, data=True):
        pair = tuple(sorted((str(a), str(b))))
        index = seen.get(pair, 0)
        seen[pair] = index + 1
        count = graph.number_of_edges(a, b)
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        dx, dy = x1 - x0, y1 - y0
        length = max(float(np.hypot(dx, dy)), 1e-12)
        offset = (index - (count - 1) / 2.0) * 0.12
        mid_x = (x0 + x1) / 2.0 - dy / length * offset
        mid_y = (y0 + y1) / 2.0 + dx / length * offset
        rows.append((a, b, str(key), data, [x0, mid_x, x1], [y0, mid_y, y1], (mid_x, mid_y)))
    return rows


def topology_plot(pkg, result: dict[str, dict], stage_id: str, highlighted: dict[str, set[str]] | None = None) -> go.Figure:
    graph = assembly_graph(pkg)
    highlighted = highlighted or {'parts': set(), 'interfaces': set(), 'stages': set()}
    pos = nx.spring_layout(graph, seed=20260718, weight=None) if graph.number_of_nodes() else {}
    state = interface_stage_state_table(pkg, result, stage_id)
    state_by_interface = state.set_index('interface_id').to_dict('index') if not state.empty else {}
    fig = go.Figure()
    midpoint_x, midpoint_y, midpoint_text, midpoint_custom = [], [], [], []
    shown_states: set[str] = set()
    for a, b, _, data, edge_x, edge_y, midpoint in interface_plot_geometry(graph, pos):
        interface_id = str(data.get('interface_id', ''))
        row = state_by_interface.get(interface_id, {})
        edge_state = str(row.get('edge_state', 'NOT_ACTIVATED'))
        color, dash = EDGE_STATE_STYLE.get(edge_state, ('#64748b', 'solid'))
        width = 7 if interface_id in highlighted.get('interfaces', set()) else 3
        hover = (
            f"接口: {interface_id}<br>接触域: {data.get('contact_domain_id', '')}<br>"
            f"连接: {data.get('joint_id', '')}<br>阶段状态: {edge_state}<br>"
            f"接触点: {row.get('contact_point_count', 0)}<br>活动点: {row.get('active_count', 0)}<br>"
            f"总接触力: {row.get('lambda_sum_N', 0):.6g} N<br>最大压力: {row.get('pressure_max_MPa', 0):.6g} MPa<br>"
            f"最小间隙: {row.get('gap_min_mm', float('nan')):.6g} mm"
        )
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines', line={'color': color, 'width': width, 'dash': dash, 'shape': 'spline'},
            hoverinfo='text', text=[hover, hover], name=edge_state,
            showlegend=edge_state not in shown_states,
        ))
        shown_states.add(edge_state)
        midpoint_x.append(midpoint[0])
        midpoint_y.append(midpoint[1])
        midpoint_text.append(hover + '<br>点击该标记查看接口详情')
        midpoint_custom.append(interface_id)
    if midpoint_x:
        fig.add_trace(go.Scatter(
            x=midpoint_x, y=midpoint_y, mode='markers+text',
            marker={'size': 16, 'color': '#ffffff', 'line': {'color': '#111827', 'width': 2}},
            text=midpoint_custom, textposition='top center', customdata=midpoint_custom,
            hovertext=midpoint_text, hoverinfo='text', name='接口（可点击）',
        ))
    node_x, node_y, node_text, node_labels, node_colors, node_sizes = [], [], [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        node_text.append(
            f"零件: {node}<br>名称: {data.get('part_name', '')}<br>材料: {data.get('material_type', '')}<br>"
            f"刚柔/角色: {data.get('rigid_flexible_flag', data.get('role_in_assembly', ''))}<br>接口度: {graph.degree[node]}"
        )
        is_highlighted = node in highlighted.get('parts', set())
        node_colors.append('#f97316' if is_highlighted else ('#0ea5e9' if graph.degree[node] >= 2 else '#475569'))
        node_sizes.append(42 if is_highlighted else 32)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_labels, textposition='bottom center',
        hovertext=node_text, hoverinfo='text', name='零件',
        marker={'size': node_sizes, 'color': node_colors, 'line': {'color': 'white', 'width': 2}},
    ))
    fig.update_layout(
        height=590, margin={'l': 20, 'r': 20, 't': 35, 'b': 20}, hovermode='closest',
        xaxis={'visible': False}, yaxis={'visible': False}, legend={'orientation': 'h'},
        title=f'数据驱动装配拓扑 | {stage_format(pkg, stage_id)}',
    )
    return fig


def coupling_network_plot(pkg, stage_id: str) -> go.Figure:
    graph = assembly_graph(pkg)
    pos = nx.spring_layout(graph, seed=20260718, weight=None) if graph.number_of_nodes() else {}
    interface_pos: dict[str, tuple[float, float]] = {}
    for _, _, _, data, _, _, midpoint in interface_plot_geometry(graph, pos):
        interface_pos[str(data.get('interface_id', ''))] = midpoint
    blocks = coupling_block_summary(pkg, stage_id)
    fig = go.Figure()
    cross = blocks[blocks['cross_interface']] if not blocks.empty else pd.DataFrame()
    max_strength = float(cross['relative_coupling_strength'].replace([np.inf, -np.inf], np.nan).max()) if not cross.empty else 1.0
    max_strength = max(max_strength if np.isfinite(max_strength) else 1.0, 1e-12)
    for _, row in cross.iterrows():
        left, right = str(row['interface_i']), str(row['interface_j'])
        if left not in interface_pos or right not in interface_pos:
            continue
        x0, y0 = interface_pos[left]
        x1, y1 = interface_pos[right]
        strength = float(row['relative_coupling_strength']) if np.isfinite(row['relative_coupling_strength']) else 0.0
        color = '#dc2626' if row['coupling_sign'] == 'NEGATIVE' else ('#16a34a' if row['coupling_sign'] == 'POSITIVE' else '#64748b')
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line={'width': 1.0 + 8.0 * strength / max_strength, 'color': color, 'dash': 'dot'},
            hovertext=[f"{left} ↔ {right}<br>相对耦合={strength:.3e}<br>F范数={row['frobenius_norm']:.3e}"] * 2,
            hoverinfo='text', showlegend=False,
        ))
    if interface_pos:
        ids = list(interface_pos)
        fig.add_trace(go.Scatter(
            x=[interface_pos[i][0] for i in ids], y=[interface_pos[i][1] for i in ids], mode='markers+text',
            text=ids, textposition='bottom center', hoverinfo='text', hovertext=ids, name='接口',
            marker={'size': 24, 'color': '#2563eb', 'symbol': 'diamond', 'line': {'color': 'white', 'width': 2}},
        ))
    fig.update_layout(
        height=480, margin={'l': 20, 'r': 20, 't': 35, 'b': 20},
        xaxis={'visible': False}, yaxis={'visible': False},
        title='接口耦合网络（绿/红表示块元素和的正/负，线宽表示相对强度）',
    )
    return fig


def make_runtime_traces(
    pkg,
    result: dict[str, dict],
    substitution_settings: NumericalSubstitutionSettings | None = None,
    sms_mapping_settings: SMSMappingSettings | None = None,
    overconstraint_settings: OverConstraintSettings | None = None,
    tangential_settings: TangentialNCPSettings | None = None,
) -> list[dict]:
    traces = []
    for sid, res in result.items():
        sol = res['solution']
        ext = res.get('extended_lcp')
        tang = res.get('tangential_ncp')
        interface_active_sets = {}
        if 'interface_id' in pkg.contact_points.columns:
            for interface_id, indices in pkg.contact_points.reset_index(drop=True).groupby('interface_id', sort=False).groups.items():
                index_list = list(indices)
                interface_active_sets[str(interface_id)] = [i for i in index_list if i in sol.active_indices]
        traces.append({
            'trace_id': f'TRACE_RUNTIME_{sid}',
            'sample_id': pkg.manifest.get('sample_id'),
            'stage_id': sid,
            'contact_domain_ids': pkg.contact_points.get('contact_domain_id', pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
            'interface_active_sets': interface_active_sets,
            'gap_field_id': 'GAP_INITIAL_E1' if not res.get('sms_rebuild') else 'GAP_RUNTIME_SMS_WLS_MAP',
            'q_vector_minmax': [float(res['q'].min()), float(res['q'].max())],
            'W_total_shape': list(res['W_total'].shape),
            'lambda_sum_N': float(sol.lambda_n.sum()),
            'active_set': sol.active_indices,
            'residuals': sol.residuals,
            'substitution_settings': substitution_settings.__dict__ if substitution_settings is not None else {},
            'sms_mapping_settings': sms_mapping_settings.__dict__ if sms_mapping_settings is not None else {},
            'overconstraint_settings': overconstraint_settings.__dict__ if overconstraint_settings is not None else {},
            'tangential_settings': tangential_settings.__dict__ if tangential_settings is not None else {},
            'extended_lcp': {
                'enabled': bool(ext is not None and ext.get('enabled', False)),
                'extended_count': int(ext.get('extended_count', 0)) if ext is not None else 0,
                'extra_lambda_sum_N': float(np.sum(ext.get('extra_lambda', []))) if ext is not None else 0.0,
                'force_nonuniqueness': res.get('force_nonuniqueness'),
            },
            'tangential_ncp': {
                'enabled': tang is not None and not tang.empty,
                'slip_count': int((tang['stick_slip_state'] == 'slip').sum()) if tang is not None and not tang.empty else 0,
                'max_lambda_t_norm_N': float(tang['lambda_t_norm_N'].max()) if tang is not None and not tang.empty else 0.0,
            },
            'quality_flag': 'PASS' if sol.convergence_status == 'CONVERGED' else 'WARN'
        })
    return traces

def make_report_zip(stage_summary: pd.DataFrame, point_results: pd.DataFrame, validation: pd.DataFrame, traces: list[dict], mc_stats: pd.DataFrame | None = None, mc_samples: pd.DataFrame | None = None, physical_report: dict | None = None, interface_summary: pd.DataFrame | None = None, coupling_summary: pd.DataFrame | None = None) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('stage_summary.csv', stage_summary.to_csv(index=False).encode('utf-8-sig'))
        zf.writestr('point_results.csv', point_results.to_csv(index=False).encode('utf-8-sig'))
        zf.writestr('kcp_validation.csv', validation.to_csv(index=False).encode('utf-8-sig'))
        zf.writestr('contact_computation_trace.json', json.dumps(traces, ensure_ascii=False, indent=2))
        if interface_summary is not None and not interface_summary.empty:
            zf.writestr('interface_stage_summary.csv', interface_summary.to_csv(index=False).encode('utf-8-sig'))
        if coupling_summary is not None and not coupling_summary.empty:
            zf.writestr('cross_interface_coupling_blocks.csv', coupling_summary.to_csv(index=False).encode('utf-8-sig'))
        if mc_stats is not None and not mc_stats.empty:
            zf.writestr('monte_carlo_stats.csv', mc_stats.to_csv(index=False).encode('utf-8-sig'))
        if mc_samples is not None and not mc_samples.empty:
            zf.writestr('monte_carlo_samples.csv', mc_samples.to_csv(index=False).encode('utf-8-sig'))
        if physical_report is not None:
            overall = physical_report.get('overall', {})
            zf.writestr('physical_consistency_overall.json', json.dumps(overall, ensure_ascii=False, indent=2))
            for name in ['stage_summary', 'check_details', 'kcp_anomalies']:
                df = physical_report.get(name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    zf.writestr(f'physical_consistency_{name}.csv', df.to_csv(index=False).encode('utf-8-sig'))
    return buf.getvalue()


st.title('SMS 输入的异质叠层结构装配接触快速预测平台')
st.caption('多零件串并联集成升级：统一向量分块、跨接口耦合求解、逐接口状态与多路径KCP投影。')

with st.sidebar:
    st.header('数据与运行参数')
    packages = discover_input_packages(DATA_ROOT)
    use_manual_path = False
    if packages:
        label_to_path = {package_label(p): p for p in packages}
        selected_label = st.selectbox('标准输入包目录', list(label_to_path.keys()))
        data_dir = str(label_to_path[selected_label])
        st.caption(f'当前路径：{data_dir}')
        st.caption(f'自动识别类型：{detect_package_type(data_dir)}')
        use_manual_path = st.checkbox('手动输入其他目录', value=False)
    else:
        st.warning('data 文件夹下没有找到 package_manifest.json。')
        data_dir = str(DATA_ROOT / DEFAULT_PACKAGE_NAME)
        use_manual_path = True
    if use_manual_path:
        data_dir = st.text_input('手动目录路径', data_dir)
        st.caption(f'自动识别类型：{detect_package_type(data_dir)}')

    try:
        pkg = load_package(data_dir)
    except Exception as exc:
        st.error(f'数据包加载失败：{exc}')
        st.stop()
    precomputed_topology_mode = uses_precomputed_topology_operators(pkg)
    measurement_checkpoints = load_measurement_checkpoints(pkg)
    has_measurement_checkpoints = bool(
        [item for item in measurement_checkpoints if item.active_flag]
    )
    measurement_update_enabled = True
    measurement_override = None
    with st.expander("阶段实测后验更新", expanded=False):
        measurement_update_enabled = st.checkbox(
            "启用 measurement update",
            value=True,
            disabled=not has_measurement_checkpoints,
            key="measurement_update_enabled",
        )
        measurement_source_choice = st.radio(
            "测量来源",
            ["数据包", "运行时 CSV"],
            disabled=not has_measurement_checkpoints,
            key="measurement_source_choice",
            horizontal=True,
        )
        if (
            has_measurement_checkpoints
            and measurement_source_choice == "运行时 CSV"
        ):
            uploaded_measurements = st.file_uploader(
                "上传过程测量 CSV",
                type=["csv"],
                key="measurement_override_upload",
                help=(
                    "仅允许 measurement_id/checkpoint_id/value/"
                    "standard_uncertainty；不会写回 data 目录。"
                ),
            )
            if uploaded_measurements is not None:
                try:
                    candidate_override = pd.read_csv(uploaded_measurements)
                    measurement_override = (
                        validate_runtime_measurement_override(
                            pkg, candidate_override
                        )
                    )
                    st.success("运行时测量通过质量门；本次运行使用覆盖值。")
                except Exception as exc:
                    measurement_override = None
                    st.error(
                        "运行时 CSV 未通过质量门，已保留包内测量："
                        f"{exc}"
                    )
        elif not has_measurement_checkpoints:
            st.caption("当前数据包没有启用的 measurement checkpoint。")

    st.divider()
    if precomputed_topology_mode:
        st.info(
            '当前为预计算 topology_step 算子模式，本倍率不会重构 q/W/Cn，因此已禁用。'
        )
    sms_scale = slider_with_number(
        'SMS 形貌倍率', 0.10, 2.50, 1.00, 0.01, key='sms_scale',
        disabled=precomputed_topology_mode,
    )
    closure_scale = slider_with_number(
        '阶段闭合/载荷倍率', 0.10, 2.50, 1.00, 0.01, key='closure_scale',
        disabled=precomputed_topology_mode,
    )
    cn_scale = slider_with_number(
        '局部界面柔度 Cn 倍率', 0.10, 4.00, 1.00, 0.01, key='cn_scale',
        disabled=precomputed_topology_mode,
    )
    eps = st.number_input('互补容差 eps', value=1e-9, format='%.1e')
    st.button('重新运行', type='primary')

with st.sidebar.expander('数值替代模块', expanded=False):
    sub_enabled = st.checkbox(
        '启用数值替代模块参与求解', value=False,
        disabled=precomputed_topology_mode,
    )
    sub_mode_label = st.selectbox(
        'Cn装配方式', ['使用原始Cn', '用数值替代Cn替换', '原始Cn + 数值替代附加项'],
        disabled=precomputed_topology_mode,
    )
    sub_mode = {'使用原始Cn': 'base_only', '用数值替代Cn替换': 'replace', '原始Cn + 数值替代附加项': 'add'}[sub_mode_label]
    st.caption('建议先用“替换”检查数值模块本身，再用“叠加”验证是否存在重复柔度。')
    use_partition = st.checkbox('分区等效 Cn', value=True, disabled=precomputed_topology_mode)
    use_layer = st.checkbox('薄层/涂层柔度', value=True, disabled=precomputed_topology_mode)
    use_rough = st.checkbox('粗糙接触柔度', value=True, disabled=precomputed_topology_mode)
    use_indent = st.checkbox('局部压陷柔度', value=True, disabled=precomputed_topology_mode)
    use_locator = st.checkbox('定位器等效柔度', value=True, disabled=precomputed_topology_mode)
    use_clamp = st.checkbox('夹持头等效柔度', value=True, disabled=precomputed_topology_mode)
    use_joint = st.checkbox('连接区等效柔度', value=True, disabled=precomputed_topology_mode)
    use_release = st.checkbox('释放回弹增量', value=True, disabled=precomputed_topology_mode)
    scale_partition = st.number_input('分区Cn倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    scale_layer = st.number_input('薄层倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    scale_rough = st.number_input('粗糙接触倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    scale_indent = st.number_input('局部压陷倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    scale_fixture = st.number_input('定位/夹持/连接倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    scale_release = st.number_input('释放回弹倍率', value=1.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)
    if precomputed_topology_mode:
        sub_enabled = False

substitution_settings = NumericalSubstitutionSettings(
    enabled=sub_enabled, mode=sub_mode,
    use_partition=use_partition, use_layer=use_layer, use_rough=use_rough, use_indent=use_indent,
    use_locator=use_locator, use_clamp=use_clamp, use_joint=use_joint, use_release=use_release,
    scale_partition=scale_partition, scale_layer=scale_layer, scale_rough=scale_rough, scale_indent=scale_indent,
    scale_locator=scale_fixture, scale_clamp=scale_fixture, scale_joint=scale_fixture, scale_release=scale_release,
)


with st.sidebar.expander('v5 高级模块', expanded=False):
    st.markdown('**S03/S04 SMS更新与G映射**')
    sms_map_enabled = st.checkbox('用 SMS 点/KCM 实时重建 g0', value=False, disabled=precomputed_topology_mode)
    sms_map_method_label = st.selectbox('SMS映射方法', ['WLS/MAP低阶基', 'IDW散点插值'], disabled=precomputed_topology_mode)
    sms_map_method = {'WLS/MAP低阶基': 'wls_basis', 'IDW散点插值': 'idw'}[sms_map_method_label]
    sms_ridge = st.number_input('WLS/MAP正则λ', value=1e-6, format='%.1e', disabled=precomputed_topology_mode)

    st.markdown('**S17 法向-切向NCP**')
    tangent_enabled = st.checkbox('启用Ct/μ切向摩擦投影', value=False, disabled=precomputed_topology_mode)
    tangent_scale = st.number_input('切向自由滑移倍率', value=1.0, min_value=0.0, max_value=10.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)

    st.markdown('**S20 N-2-1扩展LCP**')
    oc_enabled = st.checkbox('启用扩展LCP参与阶段求解', value=False, disabled=precomputed_topology_mode)
    oc_coupling = st.number_input('扩展约束-接触耦合系数', value=0.05, min_value=0.0, max_value=0.5, step=0.01, format='%.2f', disabled=precomputed_topology_mode)
    oc_scale_gap = st.number_input('扩展约束闭合倍率', value=1.0, min_value=0.0, max_value=5.0, step=0.05, format='%.2f', disabled=precomputed_topology_mode)

    st.markdown('**S28 适用域与回退判断**')
    fallback_enabled = st.checkbox('启用适用域/回退质量门', value=True)

if precomputed_topology_mode:
    sms_map_enabled = False
    tangent_enabled = False
    oc_enabled = False

sms_mapping_settings = SMSMappingSettings(enabled=sms_map_enabled, method=sms_map_method, ridge_lambda=sms_ridge)
overconstraint_settings = OverConstraintSettings(enabled=oc_enabled, coupling_ratio=oc_coupling, scale_gap_effect=oc_scale_gap)
tangential_settings = TangentialNCPSettings(enabled=tangent_enabled, shear_scale=tangent_scale)
fallback_settings = FallbackSettings(enabled=fallback_enabled)

# The package quality gate runs before any formal solve.  Detailed FAIL rows
# marked blocking identify broken primary/foreign keys, layouts, matrix/LCP
# contracts or truthfulness declarations that make a solve unsafe.
package_validation = validate_package(pkg)
if has_blocking_failures(package_validation):
    st.error('数据包存在阻断性严重错误，正式求解已停止。请先修复下表所列对象。')
    blocking_mask = package_validation.get(
        'blocking', pd.Series(False, index=package_validation.index, dtype='boolean')
    ).astype('boolean').fillna(False)
    blocking_rows = package_validation[
        blocking_mask & package_validation['status'].astype(str).eq('FAIL')
    ]
    st.dataframe(zh_df(blocking_rows), width='stretch', hide_index=True)
    st.stop()

truth_statement = data_truthfulness_statement(pkg)
if '不代表真实工程预测结果' in truth_statement:
    st.warning('仅用于数值一致性与软件联调，不代表真实工程预测结果。')

# Runtime calculation. The app is small enough to run on every interaction.
result = run_all_stages(
    pkg,
    sms_scale=sms_scale,
    closure_scale=closure_scale,
    cn_scale=cn_scale,
    eps=eps,
    substitution_settings=substitution_settings,
    sms_mapping_settings=sms_mapping_settings,
    overconstraint_settings=overconstraint_settings,
    tangential_settings=tangential_settings,
    measurement_update_enabled=measurement_update_enabled,
    measurement_override=measurement_override,
)
stage_summary = stage_summary_table(result)
point_results = point_result_table(pkg, result)
interface_results = interface_stage_summary(pkg, result)
kcp = extract_kcp(pkg, result)
validation = compare_validation(kcp, pkg.validation_kcp)
validation_context_reasons = []
if substitution_settings.enabled:
    validation_context_reasons.append('启用了数值替代Cn')
if sms_mapping_settings.enabled:
    validation_context_reasons.append('启用了SMS实时重建')
if overconstraint_settings.enabled:
    validation_context_reasons.append('启用了扩展LCP')
if tangential_settings.enabled:
    validation_context_reasons.append('启用了切向摩擦投影')
if any(
    bool(getattr(item.get("measurement_update"), "posterior_accepted", False))
    for item in result.values()
):
    validation_context_reasons.append("已接受阶段实测后验更新")
validation_comparable = not validation_context_reasons
validation_context = (
    '当前为数据包基线配置，可进行KCP验证值对比。'
    if validation_comparable else
    '；'.join(validation_context_reasons) + '。数据包未声明与该组合对应的独立验证批次，因此验证值仅作参考，不参与物理可行性判定。'
)
physical_report = physical_consistency_report(
    pkg, result, kcp, validation, eps=eps,
    validation_comparable=validation_comparable,
    validation_context=validation_context,
)
traces = make_runtime_traces(pkg, result, substitution_settings, sms_mapping_settings, overconstraint_settings, tangential_settings)

# Avoid reusing Monte Carlo results from a previously selected package or substitution setup.
mc_context_key = json.dumps({
    'root': str(pkg.root),
    'substitution': substitution_settings.__dict__,
    'sms_mapping': sms_mapping_settings.__dict__,
    'overconstraint': overconstraint_settings.__dict__,
    'tangential': tangential_settings.__dict__,
    'sms_scale': sms_scale,
    'closure_scale': closure_scale,
    'cn_scale': cn_scale,
}, ensure_ascii=False, sort_keys=True)
if st.session_state.get('mc_context_key') != mc_context_key:
    st.session_state['mc_context_key'] = mc_context_key
    st.session_state.pop('mc_samples', None)
    st.session_state.pop('mc_stats', None)

summary = pkg.manifest.get('quality_summary', {})
cols = st.columns(6)
cols[0].metric('零件数', len(pkg.parts))
cols[1].metric('接口数', len(pkg.interfaces))
cols[2].metric('候选接触点', summary.get('candidate_count', len(pkg.contact_points)))
cols[3].metric('阶段数', len(pkg.stage_plan))
cols[4].metric('KCP数量', int((pkg.kcp_kcm['feature_role'] == 'KCP').sum()))
cols[5].metric('数据类型', pkg.package_type)

TABS = [
    '① 项目/输入包管理',
    '② 数据总览/Schema读取',
    '③ 手动录入与数据编辑',
    '④ SMS与初始间隙',
    '⑤ 接触域与候选点',
    '⑥ 界面参数库/数值替代',
    '⑦ topology_step / 四阶段兼容接触求解',
    '⑧ 物理一致性检查',
    '⑨ N-2-1扩展检查',
    '⑩ KCP预测与贡献',
    '⑪ Monte Carlo与敏感性',
    '⑫ 验证、报告与追溯',
    '⑬ 装配拓扑、阶段路径与状态传递',
    '⑭ 接口耦合诊断与对照试算',
    '⑮ 阶段实测后验更新与回代',
]
active_page = st.sidebar.radio(
    '功能环节',
    TABS,
    key='main_page_navigation',
    help='切换环节时仅渲染当前页面，避免控件交互后不同环节内容串页。',
)

if active_page == TABS[0]:
    st.subheader('项目与八类输入包管理')
    if is_multi_part_package(pkg):
        st.warning('当前四零件包是合成数值一致性集成基准，仅用于验证多零件串联、并联/闭环拓扑与统一耦合求解，不能作为真实工程验证结果。')
    with st.expander('数据包说明 / Manifest', expanded=False):
        st.json(pkg.manifest)

    checks = package_validation
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('**质量门检查**')
        st.dataframe(zh_df(checks), width='stretch', hide_index=True)
    with c2:
        st.markdown('**输入包文件完整性**')
        if pkg.package_type.startswith('V25'):
            required_files = [
                'I0/part.csv', 'I0/interface.csv', 'I_Gamma/contact_point.csv', 'I_Gamma/gap_field.csv',
                'I_stage/stage_input.csv', 'I_key/kcp_definition.csv', 'I_key/kcm_definition.csv',
                'I_red/condensed_operator.csv', 'parameter_library/interface_parameter.csv',
                'solver/interface_lcp_model.csv', 'solver/lcp_solution.csv', 'solver/contact_computation_trace.csv',
                'prediction/kcp_prediction_result.csv', 'validation/validation_result.csv',
            ]
            if is_multi_part_package(pkg):
                required_files += [
                    'I0/subassembly.csv', 'I0/subassembly_membership.csv', 'I0/joint_definition.csv',
                    'I0/stage_definition.csv', 'I_stage/part_stage_state.csv',
                    'I_stage/interface_stage_state.csv', 'I_stage/stage_transition_record.csv',
                    'prediction/deformation_contribution_ledger.csv', 'prediction/double_count_check_result.csv',
                    'matrices/vector_layout.csv', 'matrices/multi_part_matrices.npz',
                ]
            else:
                required_files += ['matrices/default_matrices.npz']
        else:
            required_files = [
                'I0/part_table.csv', 'I0/interface_table.csv', 'I0/assembly_sequence.csv',
                'I_Gamma/contact_points.csv', 'I_Gamma/gap_field.csv', 'I_Gamma/interface_parameter.csv',
                'I_stage/stage_plan.csv', 'I_stage/process_record.csv',
                'I_key/KCP_KCM_list.csv', 'I_red/condensed_operator.csv',
                'I_stat/distributions.json', 'validation/validation_kcp.csv', 'matrices/E1_matrices.npz',
            ]
        file_df = pd.DataFrame([{
            '文件': f,
            '状态': 'PASS' if (pkg.root / f).exists() else 'MISSING',
            '路径': str(pkg.root / f),
        } for f in required_files])
        st.dataframe(file_df, width='stretch', hide_index=True)

    st.markdown('**核心对象预览**')
    left, right = st.columns(2)
    with left:
        st.caption('Part / 零件表')
        st.dataframe(zh_df(pkg.parts), width='stretch', hide_index=True)
        st.caption('Interface / 结合面表')
        st.dataframe(zh_df(pkg.interfaces), width='stretch', hide_index=True)
    with right:
        st.caption('StageInput / 阶段输入')
        st.dataframe(zh_df(pkg.stage_plan), width='stretch', hide_index=True)
        st.caption('KCP/KCM定义')
        st.dataframe(zh_df(pkg.kcp_kcm), width='stretch', hide_index=True)

    if is_multi_part_package(pkg):
        st.markdown('**多零件拓扑与状态继承**')
        topo = topology_summary(pkg)
        topo_cols = st.columns(5)
        topo_cols[0].metric('连通分量', topo['connected_components'])
        topo_cols[1].metric('闭环秩', topo['cycle_rank'])
        topo_cols[2].metric('串联路径', '有' if topo['has_serial_path'] else '无')
        topo_cols[3].metric('并联/闭环路径', '有' if topo['has_closed_or_parallel_path'] else '无')
        topo_cols[4].metric('最大接口度', topo['max_part_degree'])
        topology_table = pkg.raw_tables.get('I0/assembly_topology.csv', pd.DataFrame())
        state_table = state_lineage_summary(pkg)
        topo_left, topo_right = st.columns([1.15, 1])
        with topo_left:
            st.caption('装配拓扑步骤（零件为节点、接口/连接为边）')
            st.dataframe(zh_df(topology_table), width='stretch', hide_index=True)
        with topo_right:
            st.caption('聚合状态父链与逐对象状态数量')
            st.dataframe(zh_df(state_table), width='stretch', hide_index=True)


if active_page == TABS[1]:
    st.subheader('V2.5 / E1 数据读取总览')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('自动识别类型', pkg.package_type)
    c2.metric('CSV/JSON/NPZ文件数', len(pkg.data_overview) if isinstance(pkg.data_overview, pd.DataFrame) else 0)
    c3.metric('标准化接触点数', len(pkg.contact_points))
    c4.metric('矩阵key数', len(pkg.matrices))

    st.markdown('**文件读取状态（CSV / JSON / NPZ）**')
    if isinstance(pkg.data_overview, pd.DataFrame) and not pkg.data_overview.empty:
        kind_filter = st.multiselect('文件类型筛选', sorted(pkg.data_overview['kind'].dropna().unique().tolist()), default=sorted(pkg.data_overview['kind'].dropna().unique().tolist()))
        status_filter = st.multiselect('读取状态筛选', sorted(pkg.data_overview['status'].dropna().unique().tolist()), default=sorted(pkg.data_overview['status'].dropna().unique().tolist()))
        overview_df = pkg.data_overview[pkg.data_overview['kind'].isin(kind_filter) & pkg.data_overview['status'].isin(status_filter)].copy()
        st.dataframe(overview_df, width='stretch', hide_index=True)
    else:
        st.warning('未生成数据总览。')

    st.markdown('**schema_adapter 输出的当前求解器最小输入**')
    min_rows = []
    for sid in pkg.stage_plan['stage_id'].astype(str).tolist():
        min_rows.append({
            'stage_id': sid,
            'contact_point_id': ';'.join(pkg.contact_points['candidate_id'].astype(str).tolist()),
            'g0_shape': str(pkg.matrices.get('g0', np.array([])).shape),
            'q_shape': str(pkg.matrices.get(f'q__{sid}', np.array([])).shape),
            'W_struct_shape': str(pkg.matrices.get(f'W_struct__{sid}', np.array([])).shape),
            'Cn_local_shape': str(pkg.matrices.get('Cn_local', np.array([])).shape),
            'W_total_shape': str((pkg.matrices.get(f'W_struct__{sid}', np.zeros((0, 0))) + pkg.matrices.get('Cn_local', np.zeros((0, 0)))).shape) if f'W_struct__{sid}' in pkg.matrices and 'Cn_local' in pkg.matrices else 'MISSING',
            'QA_shape': str(pkg.matrices.get('QA', np.array([])).shape),
        })
    st.dataframe(pd.DataFrame(min_rows), width='stretch', hide_index=True)

    with st.expander('矩阵 key / shape / 数值范围', expanded=True):
        mat_summary = []
        for key, arr in pkg.matrices.items():
            a = np.asarray(arr)
            mat_summary.append({
                'matrix_key': key,
                'shape': str(a.shape),
                'min': float(np.nanmin(a)) if a.size and np.issubdtype(a.dtype, np.number) else np.nan,
                'max': float(np.nanmax(a)) if a.size and np.issubdtype(a.dtype, np.number) else np.nan,
            })
        st.dataframe(pd.DataFrame(mat_summary), width='stretch', hide_index=True)

    if is_multi_part_package(pkg):
        st.markdown('**全局向量布局与跨接口耦合块**')
        st.dataframe(zh_df(vector_layout(pkg)), width='stretch', hide_index=True)
        coupling_stage = st.selectbox('检查阶段 W_struct 分块', pkg.stage_plan['stage_id'].tolist(), format_func=lambda s: stage_format(pkg, s), key='coupling_stage')
        coupling_df = coupling_block_summary(pkg, coupling_stage)
        st.dataframe(zh_df(coupling_df), width='stretch', hide_index=True)
        st.caption('非对角块的范数非零表示共享零件与装配闭环引起的跨接口影响；本软件直接求解完整全局矩阵，不会把各接口分别求解后拼接。')

    with st.expander('V2.5 源表快速查看', expanded=False):
        if getattr(pkg, 'raw_tables', None):
            table_names = sorted(pkg.raw_tables.keys())
            selected_table = st.selectbox('选择源表', table_names)
            st.dataframe(pkg.raw_tables[selected_table], width='stretch', hide_index=True)
        else:
            st.info('旧 E1 包已按标准化表读取，无额外 raw_tables。')

if active_page == TABS[2]:
    st.subheader('手动录入与数据编辑')
    if has_manual_input(pkg.root):
        st.success('当前数据包包含 manual_input_table.csv，可在页面内直接修改并重建矩阵包。')
        manual_df = read_manual_input(pkg.root)
        manual_zh = zh_df(manual_df)
        disabled_cols = [c for c in ['初始间隙g0/mm', '局部序号', '候选点ID'] if c in manual_zh.columns]
        edited_zh = st.data_editor(manual_zh, width='stretch', hide_index=True, num_rows='fixed', disabled=disabled_cols)
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button('保存并重建矩阵包', type='primary'):
                edited = unzh_df(edited_zh)
                save_manual_input(pkg.root, edited)
                outputs = rebuild_from_manual_input(pkg.root)
                st.success('已保存并重建：' + '；'.join(outputs.keys()))
                st.rerun()
        with col_b:
            st.info('保存后会同步更新 I_Gamma/gap_field.csv、I_stage/stage_plan.csv 和 matrices/E1_matrices.npz。')
    else:
        st.info('当前数据包没有 manual_input_table.csv。若要手动录入，建议基于 E1_manual_input_9pt 复制一份再修改。')

    st.markdown('**过程实测记录 / process_record.csv**')
    st.dataframe(zh_df(pkg.process_record), width='stretch', hide_index=True)

if active_page == TABS[3]:
    st.subheader('SMS形貌与初始间隙构建')
    sms_file = safe_read_csv(pkg.root / 'I_meas' / 'sms_point_or_node.csv')
    if sms_file is not None:
        st.markdown('**SMS点/节点偏差记录**')
        st.dataframe(zh_df(sms_file.head(200)), width='stretch', hide_index=True)
    elif pkg.package_type.startswith('V25'):
        v25_meas = safe_read_csv(pkg.root / 'I_meas' / 'measurement_record.csv')
        v25_field = safe_read_csv(pkg.root / 'sms_update' / 'sms_field.csv')
        if v25_meas is not None and not v25_meas.empty:
            sms_rows = v25_meas[v25_meas.get('update_target', pd.Series('', index=v25_meas.index)).astype(str).str.upper().eq('SMS')]
            st.markdown('**V2.5 MeasurementRecord中的SMS更新记录**')
            st.dataframe(zh_df(sms_rows.head(200)), width='stretch', hide_index=True)
        if v25_field is not None:
            st.markdown('**V2.5冻结SMSField**')
            st.dataframe(zh_df(v25_field.head(200)), width='stretch', hide_index=True)

    if sms_mapping_settings.enabled:
        rebuilt_sms = rebuild_gap_from_sms(pkg, sms_mapping_settings)
        if rebuilt_sms.get('mapping_mode') == 'PACKAGE_FROZEN_FALLBACK':
            st.warning('V2.5原始SMS点不足以同时重建结合面两侧，本次安全使用包内冻结SMS分量/g0；未把缺失侧默认为零。')
        elif rebuilt_sms.get('mapping_mode') == 'LIVE_REBUILD_PARTIAL_LEGACY':
            st.info('旧E1兼容模式：使用现有SMS点实时重建，未提供的一侧按旧算例约定视为名义侧。')
        st.markdown('**S03：WLS/MAP SMS更新结果**')
        st.dataframe(zh_df(rebuilt_sms['fit_summary']), width='stretch', hide_index=True)
        st.markdown('**S04：SMS → 接触点 G 映射 / 重建 g0**')
        st.dataframe(zh_df(rebuilt_sms['gap_table']), width='stretch', hide_index=True)
        st.markdown('**SMS映射质量门**')
        st.dataframe(zh_df(rebuilt_sms['quality']), width='stretch', hide_index=True)
        gap_df = rebuilt_sms['gap_table'].rename(columns={'g0_rebuilt_mm': 'g0_scaled_runtime', 'nominal_component_mm': 'nominal_component', 'sms_component_rebuilt_mm': 'sms_component'}).copy()
        gap_df['g0_scaled_runtime'] = rebuilt_sms['nominal_gap'] + sms_scale * rebuilt_sms['sms_component'] + rebuilt_sms['pose_component']
        gap_df = gap_df.merge(pkg.contact_points[['candidate_id', 'area_weight']], on='candidate_id', how='left')
        gap_display_cols = ['candidate_id', 'x_i0', 'y_i0', 'nominal_component', 'sms_component', 'g0_package_mm', 'g0_scaled_runtime', 'area_weight']
    else:
        gap_df = pkg.gap_field.merge(pkg.contact_points[['candidate_id', 'x_i0', 'y_i0', 'area_weight']], on='candidate_id', how='left')
        gap_df['g0_scaled_runtime'] = pkg.matrices['nominal_gap'] + sms_scale * pkg.matrices['sms_component'] + pkg.matrices.get('pose_component', 0.0)
        gap_display_cols = ['candidate_id', 'x_i0', 'y_i0', 'nominal_component', 'sms_component', 'values_g', 'g0_scaled_runtime', 'area_weight']
    st.markdown('**GapField：名义项 + SMS项 → 初始间隙 g0**')
    st.dataframe(zh_df(gap_df[gap_display_cols]), width='stretch', hide_index=True)

    chart = alt.Chart(gap_df).mark_circle(size=170).encode(
        x=alt.X('x_i0:Q', title='x / mm'),
        y=alt.Y('y_i0:Q', title='y / mm'),
        color=alt.Color('g0_scaled_runtime:Q', title='运行g0 / mm'),
        tooltip=[c for c in ['candidate_id', 'x_i0', 'y_i0', 'nominal_component', 'sms_component', 'values_g', 'g0_package_mm', 'g0_scaled_runtime'] if c in gap_df.columns]
    ).properties(height=430)
    st.altair_chart(chart, width='stretch')
    st.warning('负 g0 只表示未平衡前的几何干涉趋势，不等于真实压缩量；真实压力和压缩由 LCP 求解。')

if active_page == TABS[4]:
    st.subheader('接触域、候选点、面积权重与映射检查')
    cp = pkg.contact_points.copy()
    st.dataframe(zh_df(cp), width='stretch', hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('候选点数', len(cp))
    c2.metric('总面积权重/mm²', f"{cp['area_weight'].sum():.2f}")
    c3.metric('最小面积权重/mm²', f"{cp['area_weight'].min():.2f}")
    c4.metric('法向nz均值', f"{cp.get('normal_nz', pd.Series([np.nan])).mean():.3f}")

    mat_rows = []
    for key in ['G_gap_mapping', 'Bn_mapping', 'Bt_mapping', 'QA', 'Cn_local']:
        if key in pkg.matrices:
            arr = pkg.matrices[key]
            mat_rows.append({'matrix': key, 'shape': str(arr.shape), 'min': float(np.min(arr)), 'max': float(np.max(arr))})
    st.markdown('**矩阵维度与数值范围**')
    st.dataframe(pd.DataFrame(mat_rows), width='stretch', hide_index=True)

    chart = alt.Chart(cp).mark_circle(size=150).encode(
        x=alt.X('x_i0:Q', title='x / mm'), y=alt.Y('y_i0:Q', title='y / mm'),
        color=alt.Color('area_weight:Q', title='面积权重'),
        tooltip=['candidate_id', 'x_i0', 'y_i0', 'area_weight', 'edge_or_interior_flag']
    ).properties(height=420)
    st.altair_chart(chart, width='stretch')

if active_page == TABS[5]:
    st.subheader('界面参数库与数值替代模块')
    st.info('本页现在不只是显示参数表，而是把分区等效Cn、薄层/涂层、粗糙接触、局部压陷、定位/夹持/连接等效柔度和释放回弹增量实际组装进求解。左侧栏可切换“原始Cn / 替换 / 叠加”。')

    selected_stage_p = st.selectbox('选择阶段', pkg.stage_plan['stage_id'].tolist(), format_func=lambda s: stage_format(pkg, s), key='param_stage')
    q, W_total, W_struct, Cn, _components_unused, _sms_rebuild_unused = build_stage_vectors(pkg, selected_stage_p, sms_scale=sms_scale, closure_scale=closure_scale, cn_scale=cn_scale, substitution_settings=substitution_settings)
    comp_df = compute_substitution_components(pkg, selected_stage_p, substitution_settings)
    comp_df['Cn_runtime_mm_per_N'] = np.diag(Cn)
    summary_df = module_summary_table(comp_df)
    qg_df = quality_gate_substitution(pkg, comp_df, substitution_settings.mode if substitution_settings.enabled else 'base_only')

    top_a, top_b, top_c, top_d = st.columns(4)
    top_a.metric('数值替代状态', '启用' if substitution_settings.enabled else '关闭')
    top_b.metric('Cn装配方式', {'base_only': '原始Cn', 'replace': '数值替代替换', 'add': '原始+替代'}[substitution_settings.mode])
    top_c.metric('本阶段平均Cn/mm/N', f"{np.mean(np.diag(Cn)):.3e}")
    top_d.metric('释放回弹增量均值/mm', f"{comp_df['release_rebound_increment_mm'].mean():.4f}")

    sub_tabs = st.tabs(['参数源表', '模块计算结果', '柔度组成与质量门', '响应分解小工具'])
    with sub_tabs[0]:
        st.markdown('**原始界面参数库 / interface_parameter.csv**')
        st.dataframe(zh_df(pkg.interface_parameters), width='stretch', hide_index=True)
        st.markdown('**I_red / CondensedOperator：结构柔度来源**')
        st.dataframe(zh_df(pkg.condensed_operator), width='stretch', hide_index=True)
        tables = load_substitution_config(pkg)
        st.markdown('**I_substitution 数值替代输入表**')
        for name in ['partition_cn', 'layer_stack', 'rough_contact', 'local_indent', 'fixture_joint', 'release_rebound', 'validity']:
            df_src = tables[name]
            with st.expander(f'{name}.csv', expanded=(name in ['partition_cn', 'validity'])):
                if isinstance(df_src, pd.DataFrame) and not df_src.empty:
                    st.dataframe(zh_df(df_src), width='stretch', hide_index=True)
                else:
                    st.warning(f'当前数据包未提供 {name}.csv，程序将按零贡献处理。')

    with sub_tabs[1]:
        st.markdown('**每个候选点的数值替代分量**')
        display_cols = [
            'candidate_id', 'zone_id', 'stage_id', 'base_Cn_from_package_mm_per_N',
            'C_partition_mm_per_N', 'C_layer_mm_per_N', 'C_rough_mm_per_N', 'C_indent_mm_per_N',
            'C_locator_mm_per_N', 'C_clamp_mm_per_N', 'C_joint_mm_per_N',
            'Cn_substitution_mm_per_N', 'Cn_runtime_mm_per_N', 'Ct_equiv_mm_per_N', 'mu_eff', 'beta_r',
            'release_rebound_increment_mm', 'quality_flag'
        ]
        st.dataframe(zh_df(comp_df[display_cols]), width='stretch', hide_index=True)
        chart_comp = alt.Chart(comp_df).transform_fold(
            ['C_partition_mm_per_N', 'C_layer_mm_per_N', 'C_rough_mm_per_N', 'C_indent_mm_per_N', 'C_locator_mm_per_N', 'C_clamp_mm_per_N', 'C_joint_mm_per_N'],
            as_=['component', 'value']
        ).mark_bar().encode(
            x=alt.X('candidate_id:N', title='候选点'),
            y=alt.Y('value:Q', title='柔度分量 / mm/N'),
            color=alt.Color('component:N', title='模块'),
            tooltip=['candidate_id:N', 'zone_id:N', 'component:N', 'value:Q']
        ).properties(height=360)
        st.altair_chart(chart_comp, width='stretch')

    with sub_tabs[2]:
        st.markdown('**模块分量汇总**')
        st.dataframe(zh_df(summary_df), width='stretch', hide_index=True)
        st.markdown('**柔度组成检查**')
        diag_df = pd.DataFrame({
            'candidate_id': pkg.contact_points['candidate_id'],
            'W_struct_diag_mm_per_N': np.diag(W_struct),
            'Cn_diag_mm_per_N': np.diag(Cn),
            'W_total_diag_mm_per_N': np.diag(W_total),
        })
        st.dataframe(zh_df(diag_df), width='stretch', hide_index=True)
        st.markdown('**数值替代质量门 / 防重复柔度提示**')
        st.dataframe(zh_df(qg_df), width='stretch', hide_index=True)
        st.caption('若使用“原始Cn + 数值替代附加项”，必须确认原始Cn没有已经包含粗糙、薄层、局部压陷等同一物理来源；否则应改用“数值替代替换”。')

    with sub_tabs[3]:
        st.markdown('**局部样件/局部FE响应分解小工具**')
        a, b, c, d, e = st.columns(5)
        u_total = a.number_input('总位移 u_total/mm', value=0.0800, step=0.001, format='%.5f')
        u_fixture = b.number_input('夹具位移/mm', value=0.0100, step=0.001, format='%.5f')
        u_sensor = c.number_input('传感器偏置/mm', value=0.0020, step=0.001, format='%.5f')
        u_substrate = d.number_input('基体变形/mm', value=0.0300, step=0.001, format='%.5f')
        force = e.number_input('法向载荷/N', value=200.0, step=10.0, format='%.2f')
        u_interface = u_total - u_fixture - u_sensor - u_substrate
        c_est = u_interface / force if force != 0 else np.nan
        st.metric('估计局部界面柔度 Cn_local = u_interface / F', f'{c_est:.6e} mm/N')
        st.info('试算后可把结果写入 I_substitution/partition_cn.csv、local_indent.csv，或写入 interface_parameter.csv。')

if active_page == TABS[6]:
    st.subheader('确定性 topology_step / 兼容四阶段 LCP 求解')
    st.dataframe(zh_df(stage_summary), width='stretch', hide_index=True)
    release_rows = stage_summary[stage_summary['stage_name'].astype(str).str.upper().eq('RELEASE')]
    join_rows = stage_summary[stage_summary['stage_name'].astype(str).str.upper().eq('JOIN')]
    final_row = release_rows.iloc[-1] if not release_rows.empty else stage_summary.iloc[-1]
    final_join = join_rows.iloc[-1] if not join_rows.empty else stage_summary.iloc[-1]
    metric_cols = st.columns(4)
    metric_cols[0].metric('最终RELEASE主动接触点', int(final_row['active_count']))
    metric_cols[1].metric('最终RELEASE最小间隙/mm', f"{final_row['gap_min_mm']:.5f}")
    metric_cols[2].metric('最终JOIN最大压力/MPa', f"{final_join['pressure_max_MPa']:.5f}")
    metric_cols[3].metric('最大互补残差', f"{stage_summary['complementarity_residual'].max():.2e}")

    selected_stage = st.selectbox('查看 topology_step 接触点结果', list(result), format_func=lambda s: stage_format(pkg, s), key='solve_stage')
    df_stage = point_results[point_results['topology_step_id'].astype(str).eq(str(result[selected_stage].get('topology_step_id', selected_stage)))]
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.dataframe(zh_df(df_stage), width='stretch', hide_index=True)
    with c2:
        chart2 = alt.Chart(df_stage).mark_circle(size=170).encode(
            x=alt.X('x_i0:Q', title='x / mm'),
            y=alt.Y('y_i0:Q', title='y / mm'),
            color=alt.Color('pressure_p_n_MPa:Q', title='pressure / MPa'),
            tooltip=['candidate_id', 'q_free_gap_mm', 'gap_g_mm', 'lambda_n_N', 'pressure_p_n_MPa', 'active_flag']
        ).properties(height=430)
        st.altair_chart(chart2, width='stretch')

    if is_multi_part_package(pkg) and not interface_results.empty:
        st.markdown('**逐接口阶段状态（由同一次全局耦合解分块汇总）**')
        interface_stage_df = interface_stage_state_table(pkg, result, selected_stage)
        st.dataframe(zh_df(interface_stage_df), width='stretch', hide_index=True)
        interface_chart = alt.Chart(interface_stage_df).mark_bar().encode(
            x=alt.X('interface_id:N', title='接口'),
            y=alt.Y('lambda_sum_N:Q', title='接触力合计 / N'),
            color=alt.Color('active_count:Q', title='主动点数'),
            tooltip=['interface_id', 'contact_point_count', 'active_count', 'lambda_sum_N', 'pressure_max_MPa', 'gap_min_mm'],
        ).properties(height=300)
        st.altair_chart(interface_chart, width='stretch')

    trace_list = result[selected_stage]['solution'].active_set_trace
    st.markdown('**主动集迭代轨迹**')
    if trace_list:
        st.dataframe(pd.DataFrame(trace_list), width='stretch', hide_index=True)
    else:
        st.caption('该阶段初始主动集已满足互补条件或无需迭代调整。')

    tangential_df = result[selected_stage].get('tangential_ncp')
    if tangential_df is not None and not tangential_df.empty:
        st.markdown('**S17 法向-切向 NCP / Ct-μ 摩擦投影结果**')
        t_left, t_right = st.columns([1.2, 1])
        with t_left:
            st.dataframe(zh_df(tangential_df), width='stretch', hide_index=True)
        with t_right:
            st.dataframe(zh_df(tangential_summary_table(tangential_df)), width='stretch', hide_index=True)
            tchart = alt.Chart(tangential_df).mark_circle(size=150).encode(
                x=alt.X('x_i0:Q', title='x / mm'),
                y=alt.Y('y_i0:Q', title='y / mm'),
                color=alt.Color('stick_slip_state:N', title='粘/滑'),
                size=alt.Size('lambda_t_norm_N:Q', title='|λt|/N'),
                tooltip=['candidate_id', 'lambda_t_norm_N', 'friction_limit_N', 'stick_slip_state', 'cone_residual']
            ).properties(height=320)
            st.altair_chart(tchart, width='stretch')


if active_page == TABS[7]:
    st.subheader('物理一致性与验证状态')
    st.caption('先判断LCP解是否物理可行，再解释接触形态；KCP产品容差和独立验证基准单独显示，避免把验证偏差误认为求解器失败。')
    pr_overall = physical_report['overall']
    status_cols = st.columns(4)
    status_cols[0].metric('① LCP物理可行性', pr_overall.get('physics_status', 'UNKNOWN'), help='只检查收敛、非负间隙、非负接触力、互补性和平衡重构。')
    status_cols[1].metric('② 接触形态提示', pr_overall.get('contact_status', 'UNKNOWN'), help='无接触或全接触为WARN，表示需要结合工况解释，并非求解失败。')
    status_cols[2].metric('③ KCP产品容差', pr_overall.get('kcp_tolerance_status', 'UNKNOWN'), help='只判断预测值是否有限、是否落在KCP容差带内。')
    status_cols[3].metric('④ KCP验证基准', pr_overall.get('kcp_validation_status', 'UNKNOWN'), help='只有当前运行配置与验证数据配置一致时，验证偏差才参与最终判定。')

    physics_status = pr_overall.get('physics_status')
    tolerance_status = pr_overall.get('kcp_tolerance_status')
    validation_status = pr_overall.get('kcp_validation_status')
    contact_status = pr_overall.get('contact_status')
    if physics_status == 'FAIL':
        st.error('LCP物理可行性FAIL：求解结果不能直接采用。请优先检查q、W、Cn、QA、阶段载荷、接触域和求解收敛状态。')
    elif tolerance_status == 'FAIL':
        st.error('LCP解可以计算，但至少一个KCP预测值无效或超出产品容差；不能将本次KCP作为合格结果。')
    elif validation_status == 'FAIL':
        st.error('LCP解满足物理可行性，但与同配置独立验证值存在显著偏差；这是验证FAIL，不是求解器失败。')
    elif contact_status == 'WARN' or tolerance_status == 'WARN' or validation_status == 'WARN':
        st.warning('计算链已完成，但存在需要工程解释的接触形态、KCP容差或验证偏差提示。请查看下方“需要关注”页签。')
    else:
        st.success('LCP解满足一级物理可行性检查，当前KCP产品容差和可比验证检查未发现异常。')

    if validation_status == 'REFERENCE_ONLY':
        st.info('KCP验证基准：仅供参考。' + str(pr_overall.get('validation_context', '当前运行配置改变，未提供同配置验证批次。')))

    is_min_placeholder = pkg.package_type == 'V25_DEFAULT_MIN_CASE' and len(pkg.contact_points) <= 1
    if is_min_placeholder:
        st.info('当前是V2.5单点最小占位包：它主要用于检查Schema和软件连通性。若四阶段均为无接触、接触力为0且LCP物理状态PASS，这是预期结果，不代表真实结构验证。')

    diag_cols = st.columns(4)
    diag_cols[0].metric('最大互补残差/N·mm', f"{pr_overall.get('max_complementarity_residual_Nmm', np.nan):.3e}")
    diag_cols[1].metric('最大间隙违反量/mm', f"{pr_overall.get('max_gap_violation_mm', np.nan):.3e}")
    diag_cols[2].metric('最大力违反量/N', f"{pr_overall.get('max_force_violation_N', np.nan):.3e}")
    diag_cols[3].metric('综合决策状态', pr_overall.get('overall_status', 'UNKNOWN'))

    phys_stage = physical_report['stage_summary']
    phys_checks = physical_report['check_details']
    kcp_anomaly = physical_report['kcp_anomalies']
    overview_tab, attention_tab, kcp_tab, method_tab = st.tabs(['阶段总览', '需要关注', 'KCP检查', '阈值与公式'])

    with overview_tab:
        st.dataframe(zh_df(phys_stage), width='stretch', hide_index=True)
        st.caption('接触形态：NO_CONTACT=全部开放；ALL_CONTACT=全部接触；MIXED_CONTACT=接触区与开放区并存。前两者是解释性WARN，不自动等同于物理FAIL。')
        if not phys_stage.empty:
            long_phys = phys_stage[['stage_id', 'gap_violation_mm', 'force_violation_N', 'complementarity_residual_Nmm', 'equilibrium_residual_mm']].melt('stage_id', var_name='metric', value_name='value')
            chart = alt.Chart(long_phys).mark_bar().encode(
                x=alt.X('stage_id:N', title='阶段'),
                y=alt.Y('value:Q', title='残差/违反量'),
                color=alt.Color('metric:N', title='指标'),
                tooltip=['stage_id', 'metric', alt.Tooltip('value:Q', format='.3e')]
            ).properties(height=300)
            st.altair_chart(chart, width='stretch')

    with attention_tab:
        attention = phys_checks[phys_checks['status'].isin(['WARN', 'FAIL'])]
        if attention.empty:
            st.success('阶段物理检查与接触形态均无WARN/FAIL。')
        else:
            st.dataframe(zh_df(attention), width='stretch', hide_index=True)
        with st.expander('查看全部检查项', expanded=False):
            st.dataframe(zh_df(phys_checks), width='stretch', hide_index=True)

    with kcp_tab:
        if validation_status == 'REFERENCE_ONLY':
            st.info('下表“产品容差状态”仍有效；“验证偏差状态”仅展示参考差异，不参与本次综合判定。')
        kcp_cols = [c for c in ['kcp_id', 'feature_type', 'stage_id', 'predicted_value', 'unit', 'lower_tol', 'upper_tol', 'tolerance_status', 'validation_status', 'validation_sigma', 'tolerance_detail', 'validation_detail'] if c in kcp_anomaly.columns]
        st.dataframe(zh_df(kcp_anomaly[kcp_cols]), width='stretch', hide_index=True)

    with method_tab:
        settings = physical_report['settings']
        threshold_df = pd.DataFrame([
            {'检查项': '间隙非负', '公式/要求': 'g ≥ 0', '阈值': settings.gap_tolerance_mm, '单位': 'mm'},
            {'检查项': '接触力非负', '公式/要求': 'λ ≥ 0', '阈值': settings.force_tolerance_N, '单位': 'N'},
            {'检查项': '互补性', '公式/要求': '|g·λ| ≈ 0', '阈值': settings.complementarity_tolerance_Nmm, '单位': 'N·mm'},
            {'检查项': '平衡重构', '公式/要求': 'g = q + Wλ', '阈值': settings.equilibrium_tolerance_mm, '单位': 'mm'},
        ])
        st.dataframe(threshold_df, width='stretch', hide_index=True)
        st.markdown('''
- **LCP物理可行性**只判断求解结果是否满足基本数学与物理约束。
- **接触形态提示**用于解释0个主动点或全部主动点，不单独证明结果错误。
- **KCP产品容差**回答“预测结果是否满足设计要求”。
- **KCP验证基准**回答“预测结果是否与同配置独立实测/高保真结果一致”。改变高级模块后，如果没有对应验证批次，只能标记为参考比较。
''')

if active_page == TABS[8]:
    st.subheader('S20：N-2-1 / 多点过约束扩展 LCP')
    if pkg.package_type.startswith('V25'):
        d12_files = {
            'overconstraint_locator_group.csv': pkg.root / 'I_stage' / 'overconstraint_locator_group.csv',
            'locator_element.csv': pkg.root / 'I_stage' / 'locator_element.csv',
            'locator_compliance.csv': pkg.root / 'I_stage' / 'locator_compliance.csv',
            'overconstraint_contact_model.csv': pkg.root / 'solver' / 'overconstraint_contact_model.csv',
            'extended_lcp_solution.csv': pkg.root / 'solver' / 'extended_lcp_solution.csv',
            'active_set_stability_trace.csv': pkg.root / 'solver' / 'active_set_stability_trace.csv',
            'force_nonuniqueness_report.csv': pkg.root / 'solver' / 'force_nonuniqueness_report.csv',
            'connection_lock_history.csv': pkg.root / 'I_stage' / 'connection_lock_history.csv',
        }
    else:
        d12_files = {
            'overconstraint_locator_group.csv': pkg.root / 'I_stage' / 'overconstraint_locator_group.csv',
            'locator_element.csv': pkg.root / 'I_stage' / 'locator_element.csv',
            'locator_compliance.csv': pkg.root / 'I_stage' / 'locator_compliance.csv',
            'overconstraint_contact_model.json': pkg.root / 'I_pred' / 'overconstraint_contact_model.json',
            'connection_lock_history.csv': pkg.root / 'I_pred' / 'connection_lock_history.csv',
            'tangential_free_slip.csv': pkg.root / 'I_stage' / 'tangential_free_slip.csv',
        }
    d12_df = pd.DataFrame([{'D12对象': k, '状态': 'FOUND' if v.exists() else 'MISSING', '路径': str(v)} for k, v in d12_files.items()])
    st.dataframe(d12_df, width='stretch', hide_index=True)

    selected_stage_ext = st.selectbox('选择扩展LCP topology_step', list(result), format_func=lambda s: stage_format(pkg, s), key='n21_stage')
    ext = result[selected_stage_ext].get('extended_lcp')
    if ext is not None and ext.get('enabled', False):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric('普通接触点', ext.get('contact_count', len(pkg.contact_points)))
        e2.metric('扩展约束点', ext.get('extended_count', 0))
        e3.metric('扩展约束反力合计/N', f"{float(np.sum(ext.get('extra_lambda', []))):.4f}")
        e4.metric('扩展LCP维度', ext['W_ext'].shape[0])
        st.markdown('**扩展约束元素求解结果**')
        st.dataframe(zh_df(extended_solution_table(selected_stage_ext, ext)), width='stretch', hide_index=True)
        st.markdown('**主动集稳定性 ρ_chi 与振荡检查**')
        st.dataframe(zh_df(active_set_stability_trace(ext['solution'])), width='stretch', hide_index=True)
        if result[selected_stage_ext].get('force_nonuniqueness'):
            st.markdown('**接触力非唯一性初步检查**')
            st.json(result[selected_stage_ext]['force_nonuniqueness'])
    else:
        st.info('扩展LCP未启用，或当前数据包未提供 D12 对象。请在左侧 v5 高级模块中启用“扩展LCP参与阶段求解”。')

    if any(p.exists() for p in d12_files.values()):
        with st.expander('查看 D12 原始数据表', expanded=False):
            for name, path in d12_files.items():
                if path.exists() and path.suffix == '.csv':
                    st.markdown(f'**{name}**')
                    st.dataframe(zh_df(pd.read_csv(path)), width='stretch', hide_index=True)
                elif path.exists() and path.suffix == '.json':
                    st.markdown(f'**{name}**')
                    st.json(json.loads(path.read_text(encoding='utf-8')))
    st.caption('当前实现将零件-零件接触、定位器、夹持头和连接区统一拼接为 q_ext / W_ext 并求解扩展互补；耦合项仍为简化核函数，真实工程应由 B_ext 与 W_struct_ext 标定。')

if active_page == TABS[9]:
    st.subheader('KCP预测与贡献分解')
    mc_samples_state = st.session_state.get('mc_samples')
    has_mc_samples = isinstance(mc_samples_state, pd.DataFrame) and not mc_samples_state.empty
    source_options = ['当前基准计算'] + (['Monte Carlo样本'] if has_mc_samples else [])
    kcp_source = st.radio('KCP结果来源', source_options, horizontal=True, key='kcp_result_source')

    display_kcp = kcp.copy()
    display_result = result
    selected_sample_row = None
    if kcp_source == 'Monte Carlo样本':
        valid_samples = mc_samples_state[mc_samples_state['status'].astype(str).eq('PASS')].copy()
        if valid_samples.empty:
            st.warning('当前Monte Carlo批次没有可用的PASS样本，请返回Monte Carlo页面重新运行。')
        else:
            sample_numbers = valid_samples['sample_no'].astype(int).tolist()
            selected_sample_no = st.selectbox(
                '选择Monte Carlo样本',
                sample_numbers,
                format_func=lambda n: f'Sample {int(n):04d}',
                key='kcp_mc_sample_no',
            )
            selected_sample_row = valid_samples[valid_samples['sample_no'].astype(int) == int(selected_sample_no)].iloc[0]
            input_cols = st.columns(4)
            input_cols[0].metric('样本编号', f'{int(selected_sample_no):04d}')
            input_cols[1].metric('SMS倍率', f"{float(selected_sample_row['sms_scale']):.5f}")
            input_cols[2].metric('闭合/载荷倍率', f"{float(selected_sample_row['closure_scale']):.5f}")
            input_cols[3].metric('Cn倍率', f"{float(selected_sample_row['cn_scale']):.5f}")

            recompute_sample = st.checkbox(
                '重新计算该样本并显示完整四阶段结果',
                value=False,
                key='kcp_recompute_mc_sample',
                help='关闭时直接读取Monte Carlo已保存的KCP；开启时用该样本倍率重新运行四阶段求解。',
            )
            if recompute_sample:
                display_result = run_all_stages(
                    pkg,
                    sms_scale=float(selected_sample_row['sms_scale']),
                    closure_scale=float(selected_sample_row['closure_scale']),
                    cn_scale=float(selected_sample_row['cn_scale']),
                    eps=eps,
                    substitution_settings=substitution_settings,
                    sms_mapping_settings=sms_mapping_settings,
                    overconstraint_settings=overconstraint_settings,
                    tangential_settings=tangential_settings,
                )
                display_kcp = extract_kcp(pkg, display_result)
                st.success('已按该样本的随机输入重新完成四阶段求解。')
            else:
                for idx, row in display_kcp.iterrows():
                    kcp_id = str(row['kcp_id'])
                    if kcp_id in selected_sample_row.index:
                        display_kcp.at[idx, 'predicted_value'] = float(selected_sample_row[kcp_id])

            baseline_map = kcp.set_index('kcp_id')['predicted_value'].to_dict()
            display_kcp['baseline_value'] = display_kcp['kcp_id'].map(baseline_map)
            display_kcp['delta_vs_baseline'] = display_kcp['predicted_value'] - display_kcp['baseline_value']
            percentile_values = []
            tolerance_status = []
            for _, row in display_kcp.iterrows():
                kcp_id = str(row['kcp_id'])
                distribution = pd.to_numeric(mc_samples_state.get(kcp_id), errors='coerce').dropna() if kcp_id in mc_samples_state.columns else pd.Series(dtype=float)
                value = float(row['predicted_value'])
                percentile_values.append(float((distribution <= value).mean()) if len(distribution) else np.nan)
                lower = pd.to_numeric(pd.Series([row.get('lower_tol', np.nan)]), errors='coerce').iloc[0]
                upper = pd.to_numeric(pd.Series([row.get('upper_tol', np.nan)]), errors='coerce').iloc[0]
                outside = (pd.notna(lower) and value < float(lower)) or (pd.notna(upper) and value > float(upper))
                tolerance_status.append('FAIL' if outside else 'PASS')
            display_kcp['mc_percentile'] = percentile_values
            display_kcp['tolerance_status'] = tolerance_status

            if recompute_sample:
                with st.expander('查看该样本完整四阶段结果', expanded=False):
                    sample_stage_summary = stage_summary_table(display_result)
                    sample_point_results = point_result_table(pkg, display_result)
                    st.dataframe(zh_df(sample_stage_summary), width='stretch', hide_index=True)
                    sample_stage_id = st.selectbox(
                        '查看该样本的阶段接触点结果',
                        sample_stage_summary['stage_id'].tolist(),
                        format_func=lambda s: stage_format(pkg, s),
                        key='kcp_mc_recomputed_stage',
                    )
                    st.dataframe(
                        zh_df(sample_point_results[sample_point_results['stage_id'] == sample_stage_id]),
                        width='stretch',
                        hide_index=True,
                    )
    if not has_mc_samples:
        st.info('尚无Monte Carlo样本。请先在“⑪ Monte Carlo与敏感性”页面运行一次Monte Carlo，之后即可在此切换样本。')

    st.dataframe(zh_df(display_kcp), width='stretch', hide_index=True)
    if is_multi_part_package(pkg):
        ledger = contribution_ledger_summary(pkg)
        ledger_cols = st.columns(4)
        ledger_cols[0].metric('贡献账本状态', ledger.get('status', 'NOT_AVAILABLE'))
        ledger_cols[1].metric('贡献记录数', ledger.get('record_count', 0))
        ledger_cols[2].metric('重复唯一键', ledger.get('duplicate_count', 0))
        ledger_cols[3].metric('重构误差', f"{float(ledger.get('reconstruction_error', np.nan)):.3e}")
        with st.expander('查看多路径贡献记录与防重复检查', expanded=False):
            contribution_records = pkg.raw_tables.get('prediction/contribution_record.csv', pd.DataFrame())
            double_count = pkg.raw_tables.get('prediction/double_count_check_result.csv', pd.DataFrame())
            st.caption('每个零件SMS登记一次；每个接口的阶段增量登记一次。')
            group_results = ledger.get('group_results', pd.DataFrame())
            if isinstance(group_results, pd.DataFrame) and not group_results.empty:
                st.caption('按 sample_id、prediction_id 和 target_kcp_id 独立重构。')
                st.dataframe(zh_df(group_results), width='stretch', hide_index=True)
            st.dataframe(zh_df(contribution_records), width='stretch', hide_index=True)
            st.dataframe(zh_df(double_count), width='stretch', hide_index=True)
    st.markdown('**与验证数据对比**')
    display_validation = compare_validation(display_kcp, pkg.validation_kcp)
    st.dataframe(zh_df(display_validation), width='stretch', hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    mae = display_validation['abs_error'].mean()
    rmse = float(np.sqrt(np.mean(display_validation['error'] ** 2)))
    max_err = display_validation['abs_error'].max()
    pass_ratio = display_validation['within_uncertainty_2sigma'].mean()
    c1.metric('MAE', f'{mae:.5f}')
    c2.metric('RMSE', f'{rmse:.5f}')
    c3.metric('Max Error', f'{max_err:.5f}')
    c4.metric('2σ内比例', f'{pass_ratio:.0%}')

    val_melt = display_validation[['kcp_id', 'predicted_value', 'measured_value']].melt('kcp_id', var_name='kind', value_name='value')
    chart = alt.Chart(val_melt).mark_bar().encode(
        x=alt.X('kcp_id:N', title='KCP'),
        y=alt.Y('value:Q', title='数值'),
        color=alt.Color('kind:N', title='类型'),
        xOffset='kind:N',
        tooltip=['kcp_id', 'kind', 'value']
    ).properties(height=360)
    st.altair_chart(chart, width='stretch')

    st.markdown('**简化贡献说明**')
    contrib = pd.DataFrame([
        {'贡献项': 'SMS几何直接项', '本版来源': 'nominal_component + sms_component → g0', '是否已计算': '是'},
        {'贡献项': '接触附加项', '本版来源': 'q + Wλ → g, λ, p, w', '是否已计算': '是'},
        {'贡献项': '工艺阶段项', '本版来源': 'u_free__stage / closure_scale', '是否已计算': '是'},
        {'贡献项': '界面参数项', '本版来源': 'Cn_local / cn_scale', '是否已计算': '是'},
        {'贡献项': 'N-2-1扩展约束项', '本版来源': 'D12 / 扩展LCP', '是否已计算': '是' if overconstraint_settings.enabled else '未启用'},
    ])
    st.dataframe(contrib, width='stretch', hide_index=True)

if active_page == TABS[10]:
    st.subheader('Monte Carlo统计预测与单因素敏感性')
    if precomputed_topology_mode:
        st.info(
            '当前为预计算 topology_step 算子模式，本倍率不会重构 q/W/Cn，因此已禁用。'
            '本轮不对该模式执行基于这些倍率的 Monte Carlo 或单因素敏感性扫描。'
        )
        st.dataframe(
            topology_step_execution_table(result)[
                ['topology_step_id', 'operator_source', 'parameter_effective', 'parameter_mode']
            ],
            width='stretch',
            hide_index=True,
        )
    else:
        defaults = distribution_defaults(pkg)
        st.markdown('**分布参数**')
        col1, col2, col3, col4 = st.columns(4)
        n_mc = col1.number_input('样本数 N_MC', min_value=10, max_value=2000, value=100, step=10)
        seed = col2.number_input('随机种子', min_value=0, value=20260708, step=1)
        mc_sms_std = col3.number_input('SMS倍率标准差', min_value=0.0, value=float(defaults['sms_scale'][1]), step=0.01, format='%.3f')
        mc_closure_std = col4.number_input('闭合倍率标准差', min_value=0.0, value=float(defaults['closure_scale'][1]), step=0.01, format='%.3f')
        col5, col6 = st.columns(2)
        mc_cn_std = col5.number_input('Cn倍率标准差', min_value=0.0, value=float(defaults['cn_scale'][1]), step=0.01, format='%.3f')
        run_mc = col6.button('运行 Monte Carlo', type='primary')

        if run_mc or 'mc_samples' not in st.session_state:
            means_stds = {
                'sms_scale': (sms_scale, mc_sms_std),
                'closure_scale': (closure_scale, mc_closure_std),
                'cn_scale': (cn_scale, mc_cn_std),
            }
            samples, stats = run_monte_carlo(pkg, int(n_mc), int(seed), means_stds, eps=eps, substitution_settings=substitution_settings, sms_mapping_settings=sms_mapping_settings, overconstraint_settings=overconstraint_settings, tangential_settings=tangential_settings)
            st.session_state['mc_samples'] = samples
            st.session_state['mc_stats'] = stats
        else:
            samples, stats = st.session_state['mc_samples'], st.session_state['mc_stats']

        st.markdown('**统计结果**')
        st.dataframe(zh_df(stats), width='stretch', hide_index=True)
        kcp_cols = [c for c in samples.columns if c.startswith('KCP_')]
        selected_kcp = st.selectbox('查看KCP分布', kcp_cols, key='mc_kcp') if kcp_cols else None
        if selected_kcp:
            hist = alt.Chart(samples).mark_bar().encode(
                x=alt.X(f'{selected_kcp}:Q', bin=alt.Bin(maxbins=24), title=selected_kcp),
                y=alt.Y('count()', title='样本数'),
                tooltip=[alt.Tooltip('count()', title='样本数')]
            ).properties(height=320)
            st.altair_chart(hist, width='stretch')

        with st.expander('查看 Monte Carlo 样本明细', expanded=False):
            st.dataframe(zh_df(samples), width='stretch', hide_index=True)
            st.download_button(
                '下载逐次Monte Carlo KCP预测CSV',
                zh_df(samples).to_csv(index=False).encode('utf-8-sig'),
                file_name='monte_carlo_samples_zh.csv',
                mime='text/csv',
            )

        st.markdown('**单因素敏感性扫描**')
        s1, s2, s3 = st.columns(3)
        var = s1.selectbox('扫描变量', ['sms_scale', 'closure_scale', 'cn_scale'], format_func=lambda v: {'sms_scale': 'SMS倍率', 'closure_scale': '闭合/载荷倍率', 'cn_scale': 'Cn倍率'}[v])
        v_min = s2.number_input('最小值', value=0.70, step=0.05, format='%.2f')
        v_max = s3.number_input('最大值', value=1.30, step=0.05, format='%.2f')
        values = np.linspace(float(v_min), float(v_max), 9).round(4).tolist()
        sweep = one_factor_sweep(pkg, var, values, {'sms_scale': sms_scale, 'closure_scale': closure_scale, 'cn_scale': cn_scale}, eps=eps, substitution_settings=substitution_settings, sms_mapping_settings=sms_mapping_settings, overconstraint_settings=overconstraint_settings, tangential_settings=tangential_settings)
        st.dataframe(zh_df(sweep), width='stretch', hide_index=True)
        if kcp_cols:
            kcp_for_sweep = st.selectbox('敏感性曲线KCP', kcp_cols, key='sweep_kcp')
            line = alt.Chart(sweep).mark_line(point=True).encode(
                x=alt.X('value:Q', title='变量值'),
                y=alt.Y(f'{kcp_for_sweep}:Q', title=kcp_for_sweep),
                tooltip=['variable', 'value', kcp_for_sweep]
            ).properties(height=320)
            st.altair_chart(line, width='stretch')

if active_page == TABS[11]:
    st.subheader('验证、报告导出与计算追溯')
    st.markdown('**ValidationResult**')
    st.dataframe(zh_df(validation), width='stretch', hide_index=True)
    bar = alt.Chart(validation).mark_bar().encode(
        x=alt.X('kcp_id:N', title='KCP'),
        y=alt.Y('abs_error:Q', title='绝对误差'),
        tooltip=['kcp_id', 'predicted_value', 'measured_value', 'abs_error', 'unit']
    ).properties(height=340)
    st.altair_chart(bar, width='stretch')

    if fallback_settings.enabled:
        sms_quality_for_fallback = None
        if sms_mapping_settings.enabled:
            try:
                sms_quality_for_fallback = rebuild_gap_from_sms(pkg, sms_mapping_settings)['quality']
            except Exception:
                sms_quality_for_fallback = pd.DataFrame([{'check_item': 'SMS映射', 'status': 'FAIL', 'detail': 'SMS映射执行失败'}])
        fallback_table = evaluate_validity_and_fallback(pkg, result, fallback_settings, sms_quality_for_fallback)
        st.markdown('**S28 适用域与替代-回退判断**')
        st.dataframe(zh_df(fallback_table), width='stretch', hide_index=True)

    st.markdown('**剩余不足与需要补充的数据/接口**')
    st.dataframe(zh_df(remaining_limitations_table()), width='stretch', hide_index=True)

    st.markdown('**V2.5 / 数据包内追溯对象读取展示**')
    source_trace_tabs = st.tabs(['ContactComputationTrace', 'LCPSolution', 'KCPPredictionResult'])
    with source_trace_tabs[0]:
        src = pkg.raw_tables.get('solver/contact_computation_trace.csv') if getattr(pkg, 'raw_tables', None) else None
        if src is None:
            src = pkg.raw_tables.get('validation/contact_computation_trace.csv') if getattr(pkg, 'raw_tables', None) else None
        if src is not None and not src.empty:
            st.dataframe(zh_df(src), width='stretch', hide_index=True)
        else:
            st.info('当前数据包未提供可直接读取的 ContactComputationTrace 源表。')
    with source_trace_tabs[1]:
        src = pkg.raw_tables.get('solver/lcp_solution.csv') if getattr(pkg, 'raw_tables', None) else None
        if src is not None and not src.empty:
            st.dataframe(zh_df(src), width='stretch', hide_index=True)
        else:
            st.info('当前数据包未提供 solver/lcp_solution.csv。')
    with source_trace_tabs[2]:
        src = pkg.raw_tables.get('prediction/kcp_prediction_result.csv') if getattr(pkg, 'raw_tables', None) else None
        if src is not None and not src.empty:
            st.dataframe(zh_df(src), width='stretch', hide_index=True)
        else:
            st.info('当前数据包未提供 prediction/kcp_prediction_result.csv。')

    st.markdown('**ContactComputationTrace（本次运行动态生成）**')
    st.json(traces)

    mc_stats_for_zip = st.session_state.get('mc_stats') if isinstance(st.session_state.get('mc_stats'), pd.DataFrame) else None
    mc_samples_for_zip = st.session_state.get('mc_samples') if isinstance(st.session_state.get('mc_samples'), pd.DataFrame) else None
    coupling_export = pd.concat(
        [coupling_block_summary(pkg, sid) for sid in result],
        ignore_index=True,
    ) if is_multi_part_package(pkg) else pd.DataFrame()
    report_zip = build_runtime_report_zip(
        pkg, result, validation, traces,
        mc_stats=mc_stats_for_zip, mc_samples=mc_samples_for_zip,
        physical_report=physical_report,
    )
    measurement_report_tables, measurement_report_traces = (
        measurement_update_report_tables(pkg, result)
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button('下载KCP验证CSV（中文表头）', zh_df(validation).to_csv(index=False).encode('utf-8-sig'),
                       file_name='runtime_kcp_validation_zh.csv', mime='text/csv')
    d2.download_button('下载追溯JSON', json.dumps(traces, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name='runtime_contact_computation_trace.json', mime='application/json')
    d3.download_button('下载完整运行报告ZIP', report_zip,
                       file_name=f'runtime_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip', mime='application/zip')

    if st.button('保存报告到本地 reports 文件夹'):
        run_dir = REPORT_ROOT / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        run_dir.mkdir(parents=True, exist_ok=True)
        stage_summary.to_csv(run_dir / 'stage_summary.csv', index=False, encoding='utf-8-sig')
        point_results.to_csv(run_dir / 'point_results.csv', index=False, encoding='utf-8-sig')
        validation.to_csv(run_dir / 'kcp_validation.csv', index=False, encoding='utf-8-sig')
        (run_dir / 'contact_computation_trace.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
        if mc_stats_for_zip is not None:
            mc_stats_for_zip.to_csv(run_dir / 'monte_carlo_stats.csv', index=False, encoding='utf-8-sig')
        if mc_samples_for_zip is not None:
            mc_samples_for_zip.to_csv(run_dir / 'monte_carlo_samples.csv', index=False, encoding='utf-8-sig')
        if not interface_results.empty:
            interface_results.to_csv(run_dir / 'interface_stage_summary.csv', index=False, encoding='utf-8-sig')
        if not coupling_export.empty:
            coupling_export.to_csv(run_dir / 'cross_interface_coupling_blocks.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([topology_summary(pkg)]).to_csv(run_dir / 'topology_summary.csv', index=False, encoding='utf-8-sig')
        assembly_path_summary(pkg).to_csv(run_dir / 'assembly_path_summary.csv', index=False, encoding='utf-8-sig')
        stage_transition_runtime_table(result).to_csv(run_dir / 'stage_transition_runtime.csv', index=False, encoding='utf-8-sig')
        coupling_ablation_export(pkg, result).to_csv(run_dir / 'coupling_ablation_comparison.csv', index=False, encoding='utf-8-sig')
        runtime_state_lineage(result).to_csv(run_dir / 'state_lineage.csv', index=False, encoding='utf-8-sig')
        package_validation.to_csv(run_dir / 'validation_summary.csv', index=False, encoding='utf-8-sig')
        topology_step_execution_table(result).to_csv(run_dir / 'topology_step_execution.csv', index=False, encoding='utf-8-sig')
        validate_topology_steps(pkg).to_csv(run_dir / 'topology_step_validation.csv', index=False, encoding='utf-8-sig')
        topology_step_execution_table(result)[[
            'sample_id', 'topology_id', 'topology_step_id', 'step_order', 'result_subassembly_id',
            'active_part_ids', 'active_interface_ids', 'active_joint_ids', 'active_boundary_ids', 'active_load_ids',
        ]].to_csv(run_dir / 'active_subassembly_history.csv', index=False, encoding='utf-8-sig')
        topology_step_state_lineage_table(result).to_csv(run_dir / 'topology_step_state_lineage.csv', index=False, encoding='utf-8-sig')
        topology_step_operator_usage_table(result).to_csv(run_dir / 'topology_step_operator_usage.csv', index=False, encoding='utf-8-sig')
        topology_step_contact_summary_table(result).to_csv(run_dir / 'topology_step_contact_summary.csv', index=False, encoding='utf-8-sig')
        connection_lock_history_table(result).to_csv(run_dir / 'connection_lock_history.csv', index=False, encoding='utf-8-sig')
        release_history_table(result).to_csv(run_dir / 'release_history.csv', index=False, encoding='utf-8-sig')
        for report_name, report_table in measurement_report_tables.items():
            report_table.to_csv(
                run_dir / report_name,
                index=False,
                encoding='utf-8-sig',
            )
        (run_dir / 'measurement_update_trace.json').write_text(
            json.dumps(
                measurement_report_traces,
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        (run_dir / 'data_truthfulness_statement.txt').write_text(truth_statement, encoding='utf-8')
        (run_dir / 'physical_consistency_overall.json').write_text(json.dumps(physical_report.get('overall', {}), ensure_ascii=False, indent=2), encoding='utf-8')
        for name in ['stage_summary', 'check_details', 'kcp_anomalies']:
            df = physical_report.get(name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(run_dir / f'physical_consistency_{name}.csv', index=False, encoding='utf-8-sig')
        st.success(f'已保存到：{run_dir}')


if active_page == TABS[12]:
    st.subheader('装配拓扑、阶段路径与状态传递')
    st.caption('二维装配关系来自 I0/part.csv、interface.csv、joint_definition.csv 与接触域表；不维护手工拓扑，不表示三维CAD或机器人轨迹。')
    topo = topology_summary(pkg)
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric('连通分量', int(topo['connected_components']))
    t2.metric('串联路径', '有' if topo['has_serial_path'] else '无')
    t3.metric('并联/闭环', f"{'有' if topo['has_closed_or_parallel_path'] else '无'} / 秩{topo['cycle_rank']}")
    t4.metric('共享零件', int(topo['shared_part_count']))
    t5.metric('共享柔性零件', int(topo['shared_flexible_part_count']))

    route_table = topology_step_table(pkg)
    st.markdown('**topology_step 工艺路线表**')
    st.dataframe(zh_df(route_table), width='stretch', hide_index=True)
    if not route_table.empty:
        timeline = alt.Chart(route_table).mark_circle(size=180).encode(
            x=alt.X('step_order:Q', title='step_order'),
            y=alt.Y('assembly_cycle_id:N', title='assembly_cycle'),
            color=alt.Color('operation_type:N', title='operation_type'),
            tooltip=['topology_step_id', 'step_order', 'assembly_cycle_id', 'operation_type', 'result_subassembly_id'],
        ).properties(height=220, title='确定性工艺路线时间轴')
        st.altair_chart(timeline, width='stretch')

    top_left, top_right = st.columns([1, 1])
    topology_stage = top_left.selectbox(
        'topology_step 选择', list(result), format_func=lambda value: stage_format(pkg, value), key='topology_stage'
    )
    kcp_options = ['（不高亮KCP）'] + kcp['kcp_id'].astype(str).tolist()
    selected_kcp_path = top_right.selectbox('KCP贡献路径高亮', kcp_options, key='topology_kcp')
    highlight = {'parts': set(), 'interfaces': set(), 'stages': set()}
    if selected_kcp_path != '（不高亮KCP）':
        highlight = kcp_contribution_path(pkg, selected_kcp_path)
        st.caption(
            f"高亮零件：{'; '.join(sorted(highlight['parts'])) or '无显式来源'}；"
            f"接口：{'; '.join(sorted(highlight['interfaces'])) or '无显式来源'}；"
            f"阶段：{'; '.join(sorted(highlight['stages'])) or '未声明'}"
        )
    selection = st.plotly_chart(
        topology_plot(pkg, result, topology_stage, highlight),
        width='stretch',
        key='topology_plot_selection',
        on_select='rerun',
        selection_mode='points',
    )
    interface_ids = pkg.interfaces.get('interface_id', pd.Series(dtype=str)).astype(str).tolist()
    clicked_interface = None
    try:
        selected_points = selection.selection.points
        if selected_points and selected_points[0].get('customdata') in interface_ids:
            clicked_interface = selected_points[0].get('customdata')
    except (AttributeError, KeyError, TypeError):
        clicked_interface = None
    if clicked_interface:
        st.session_state['topology_interface_detail'] = clicked_interface
    default_interface = st.session_state.get('topology_interface_detail', interface_ids[0] if interface_ids else '')
    if default_interface not in interface_ids and interface_ids:
        default_interface = interface_ids[0]
    selected_interface = st.selectbox(
        '接口详情（可点击图中接口白色标记，也可在此选择）', interface_ids,
        index=interface_ids.index(default_interface) if default_interface in interface_ids else 0,
        key='topology_interface_selector',
    ) if interface_ids else None
    if selected_interface:
        detail = interface_stage_state_table(pkg, result, topology_stage)
        detail = detail[detail['interface_id'].astype(str).eq(str(selected_interface))]
        st.markdown('**接口阶段详情**')
        st.dataframe(zh_df(detail), width='stretch', hide_index=True)

    selected_result = result[topology_stage]
    selected_spec = selected_result['topology_step_spec']
    selected_state = selected_result['stage_state']
    st.markdown('**步骤执行与追溯详情**')
    detail_columns = st.columns(4)
    detail_columns[0].metric('assembly_cycle', selected_state.assembly_cycle_id or '未声明')
    detail_columns[1].metric('operation_type', selected_state.operation_type)
    detail_columns[2].metric('solve_status', selected_result['solve_status'])
    detail_columns[3].metric('LCP 调用次数', selected_result['lcp_call_count'])
    st.dataframe(pd.DataFrame([{
        'topology_step_id': topology_stage,
        'parent_topology_step_id': selected_state.parent_topology_step_id or '',
        'parent_state_id': selected_state.parent_stage_state_id or '',
        'input_subassembly_id': selected_spec.input_subassembly_id or '',
        'result_subassembly_id': selected_state.current_subassembly_id,
        'added_part_ids': ';'.join(selected_spec.added_part_ids),
        'activated_interface_ids': ';'.join(selected_spec.activated_interface_ids),
        'deactivated_interface_ids': ';'.join(selected_spec.deactivated_interface_ids),
        'activated_boundary_ids': ';'.join(selected_spec.activated_boundary_ids),
        'deactivated_boundary_ids': ';'.join(selected_spec.deactivated_boundary_ids),
        'activated_load_ids': ';'.join(selected_spec.activated_load_ids),
        'removed_load_ids': ';'.join(selected_spec.removed_load_ids),
        'activated_joint_ids': ';'.join(selected_spec.activated_joint_ids),
        'active_interface_ids': ';'.join(selected_state.active_interface_ids),
        'operator_set_id': selected_state.operator_set_id,
        'operator_source': selected_state.operator_source,
        'measurement_checkpoint_id': selected_state.measurement_checkpoint_id,
        'predicted_state_id': selected_state.predicted_state_id,
        'posterior_state_id': (
            selected_result.get('posterior_stage_state').stage_state_id
            if selected_result.get('posterior_stage_state') is not None else ''
        ),
        'effective_state_id': selected_state.effective_state_id,
        'state_role': selected_state.state_role,
        'measurement_update_status': selected_state.measurement_update_status,
        'posterior_accepted': selected_state.posterior_accepted,
        'covariance_trace': (
            float(np.trace(selected_state.state_covariance))
            if np.asarray(selected_state.state_covariance).size else np.nan
        ),
        'state_correction_norm': float(
            np.linalg.norm(selected_state.state_correction_vector)
        ),
        'rollback_status': (
            selected_state.rollback_record_id
            or 'NO_ROLLBACK'
        ),
        'complementarity_residual': selected_result['solution'].residuals.get('complementarity_residual', 0.0),
    }]), width='stretch', hide_index=True)
    history_left, history_right = st.columns(2)
    with history_left:
        st.markdown('**JOIN 锁定历史**')
        locks = connection_lock_history_table(result)
        st.dataframe(zh_df(locks), width='stretch', hide_index=True) if not locks.empty else st.caption('当前步骤链尚无锁定记录。')
    with history_right:
        st.markdown('**RELEASE 继承记录**')
        releases = release_history_table(result)
        st.dataframe(zh_df(releases), width='stretch', hide_index=True) if not releases.empty else st.caption('当前步骤链尚无释放记录。')

    path_table = assembly_path_summary(pkg)
    path_tab, transition_tab, lineage_tab = st.tabs(['装配路径识别', '阶段前后状态变化', '运行时父状态链'])
    with path_tab:
        if path_table.empty:
            st.info('当前拓扑未识别到长度≥2的串联路径、并联替代路径或闭环。')
        else:
            st.dataframe(zh_df(path_table), width='stretch', hide_index=True)
    with transition_tab:
        runtime_transition = stage_transition_runtime_table(result)
        st.dataframe(zh_df(runtime_transition), width='stretch', hide_index=True)
        stage_order = list(result)
        pos = stage_order.index(topology_stage)
        current_if = interface_stage_state_table(pkg, result, topology_stage)
        if pos > 0:
            previous_if = interface_stage_state_table(pkg, result, stage_order[pos - 1])
            before_after = previous_if.merge(current_if, on='interface_id', suffixes=('_before', '_after'))
            for metric in ('active_count', 'lambda_sum_N', 'pressure_max_MPa', 'gap_min_mm'):
                before_after[f'{metric}_change'] = before_after[f'{metric}_after'] - before_after[f'{metric}_before']
            st.markdown(f'**{stage_order[pos - 1]} → {topology_stage} 接口状态变化**')
            st.dataframe(zh_df(before_after), width='stretch', hide_index=True)
        else:
            st.info('首阶段相对于初始聚合参考态建立运行时状态。')
    with lineage_tab:
        st.dataframe(zh_df(runtime_state_lineage(result)), width='stretch', hide_index=True)
        st.info('fallback_flag=true 表示 q/U_FREE/W_struct 仍使用包内预计算输入；接触解、接口汇总、阶段增量和父状态引用由本次运行生成。')


if active_page == TABS[13]:
    st.subheader('接口耦合诊断与可视化')
    st.success('正式计算模式：所有阶段均使用完整 W_total = W_struct + Cn 的全局统一 LCP；跨接口块保持不变。')
    st.warning('“跨接口块置零”仅是诊断对照，永远不标记为正式工程结果。')
    coupling_stage = st.selectbox(
        '查看阶段 W_struct', list(result), format_func=lambda value: stage_format(pkg, value), key='coupling_page_stage'
    )
    layout = vector_layout(pkg)
    W_view = np.asarray(result[coupling_stage]['W_struct'], dtype=float)
    heatmap = go.Figure(go.Heatmap(
        z=W_view, colorscale='RdBu', zmid=0.0,
        colorbar={'title': 'W_struct / mm·N⁻¹'},
        hovertemplate='row=%{y}<br>col=%{x}<br>W=%{z:.4e}<extra></extra>',
    ))
    for _, block in layout.iterrows():
        boundary = int(block['end_index']) + 0.5
        heatmap.add_vline(x=boundary, line_width=2, line_color='black')
        heatmap.add_hline(y=boundary, line_width=2, line_color='black')
    heatmap.update_layout(height=620, title=f'W_struct 全局热力图 | {stage_format(pkg, coupling_stage)}', xaxis_title='全局列索引', yaxis_title='全局行索引')
    st.plotly_chart(heatmap, width='stretch')
    coupling_blocks = coupling_block_summary(pkg, coupling_stage)
    st.markdown('**VectorLayout接口块诊断**')
    st.dataframe(zh_df(coupling_blocks), width='stretch', hide_index=True)
    st.plotly_chart(coupling_network_plot(pkg, coupling_stage), width='stretch')

    st.markdown('**耦合影响对照试算**')
    threshold_percent = st.number_input('明显变化警告阈值/%', min_value=0.0, max_value=1000.0, value=5.0, step=1.0, key='coupling_warning_threshold')
    run_ablation = st.button('运行诊断：将跨接口 W_struct 块置零', type='secondary', key='run_coupling_ablation')
    ablation_key = f"{pkg.root}|{coupling_stage}|{threshold_percent}"
    if run_ablation:
        st.session_state['coupling_ablation_key'] = ablation_key
        st.session_state['coupling_ablation_result'] = coupling_ablation_comparison(
            pkg, result, coupling_stage, float(threshold_percent) / 100.0
        )
    if st.session_state.get('coupling_ablation_key') == ablation_key:
        ablation = st.session_state.get('coupling_ablation_result')
        if isinstance(ablation, dict):
            summary_ablation = ablation['summary']
            if bool(summary_ablation.iloc[0]['warning_flag']):
                st.error('去除跨接口耦合后结果变化超过阈值或活动集发生变化；正式结果对接口耦合敏感。')
            else:
                st.info('该阶段诊断变化未超过当前阈值；正式结果仍继续使用完整耦合矩阵。')
            st.dataframe(zh_df(summary_ablation), width='stretch', hide_index=True)
            compare_tabs = st.tabs(['lambda/g/活动集', '最大压力与KCP'])
            with compare_tabs[0]:
                st.dataframe(zh_df(ablation['point_comparison']), width='stretch', hide_index=True)
            with compare_tabs[1]:
                st.dataframe(zh_df(ablation['kcp_comparison']), width='stretch', hide_index=True)


if active_page == TABS[14]:
    st.subheader("阶段实测后验更新与回代")
    st.warning(
        "本页只更新数据包定义的低维阶段状态，不辨识 Cn/Ct/mu/"
        "beta_r、连接刚度或 SMS，也不执行在线 FE 更新。"
    )
    if not has_measurement_checkpoints:
        st.info(
            "当前数据包未配置阶段实测后验更新，原 topology_step "
            "预测流程保持不变。"
        )
    else:
        if (
            str(
                pkg.manifest.get(
                    "measurement_data_nature",
                    pkg.manifest.get("data_nature", ""),
                )
            )
            == "SYNTHETIC_NUMERICAL_CONSISTENCY_CASE"
        ):
            st.info(
                "当前测量为合成数值一致性数据，仅用于验证后验更新、"
                "物理重求和状态传递。"
            )
        checkpoint_records = pd.DataFrame([
            item.to_record() for item in measurement_checkpoints
        ])
        st.markdown("**checkpoint 路线表**")
        st.dataframe(
            checkpoint_records,
            width="stretch",
            hide_index=True,
        )
        checkpoint_ids = [
            item.checkpoint_id
            for item in measurement_checkpoints
            if item.active_flag
        ]
        selected_checkpoint_id = st.selectbox(
            "checkpoint 选择",
            checkpoint_ids,
            key="measurement_checkpoint_selector",
        )
        measurement_tables, measurement_traces = (
            measurement_update_report_tables(pkg, result)
        )
        selected_updates = [
            item.get("measurement_update")
            for item in result.values()
            if (
                item.get("measurement_update") is not None
                and item["measurement_update"].checkpoint_id
                == selected_checkpoint_id
            )
        ]
        if not measurement_update_enabled:
            st.info(
                "measurement update 已禁用；本次运行保留 prior-only 路径。"
            )
        elif not selected_updates:
            st.error(
                "checkpoint 已配置，但本次运行没有产生更新记录。"
            )
        else:
            update = selected_updates[0]
            trace = update.trace
            prior_residual = float(
                trace.get("prior_residual_norm", float("nan"))
            )
            posterior_residual = float(
                trace.get("posterior_residual_norm", float("nan"))
            )
            posterior_linearized_residual = float(
                trace.get(
                    "posterior_linearized_residual_norm",
                    float("nan"),
                )
            )
            prior_trace = float(
                trace.get("P_prior_trace", float("nan"))
            )
            posterior_trace = float(
                trace.get("P_posterior_trace", float("nan"))
            )
            metric_columns = st.columns(10)
            metric_values = (
                ("checkpoint ID", update.checkpoint_id),
                ("使用测量数", len(update.measurement_ids)),
                ("更新状态维数", len(update.eta_posterior)),
                ("prior physical residual", f"{prior_residual:.6g}"),
                (
                    "linearized residual",
                    f"{posterior_linearized_residual:.6g}",
                ),
                (
                    "post-LCP physical residual",
                    f"{posterior_residual:.6g}",
                ),
                (
                    "weighted physical residual",
                    f"{float(trace.get('weighted_residual_posterior_physical', float('nan'))):.6g}",
                ),
                ("covariance trace reduction",
                 f"{prior_trace - posterior_trace:.6g}"),
                ("NIS", f"{update.nis:.6g}"),
                ("re-solve LCP", update.resolve_lcp_call_count),
            )
            for column, (label, value) in zip(
                metric_columns, metric_values
            ):
                column.metric(label, value)
            status = str(trace.get(
                "status",
                "POSTERIOR_ACCEPTED"
                if update.posterior_accepted
                else "POSTERIOR_REJECTED_ROLLBACK",
            ))
            if update.posterior_accepted:
                st.success(status)
            elif update.rollback_record is None:
                st.info(status)
            else:
                st.error(status)
            st.caption(
                f"测量来源：{update.measurement_source}；"
                f"effective_state_id={update.effective_state_id}；"
                f"接受依据={trace.get('acceptance_basis', trace.get('decision_basis', ''))}；"
                f"协方差来源={trace.get('measurement_covariance_source', '')}"
            )
            st.markdown("**线性预测与实际 post-LCP 物理预测**")
            if update.measurement_ids:
                physical_observation_table = pd.DataFrame({
                    "measurement_id": update.measurement_ids,
                    "z_predicted_prior_physical":
                        update.z_predicted_prior_physical,
                    "z_predicted_posterior_linearized":
                        update.z_predicted_posterior_linearized,
                    "z_predicted_posterior_physical":
                        update.z_predicted_posterior_physical,
                    "residual_prior_physical":
                        update.residual_prior_physical,
                    "residual_posterior_linearized":
                        update.residual_posterior_linearized,
                    "residual_posterior_physical":
                        update.residual_posterior_physical,
                    "linearization_error": update.linearization_error,
                    "physical_residual_improved":
                        update.physical_residual_improved,
                })
                st.dataframe(
                    physical_observation_table,
                    width="stretch",
                    hide_index=True,
                )
            st.dataframe(
                pd.DataFrame(update.observation_records),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "posterior residual 默认指 LCP 回代后的实际物理残差；"
                "linearized residual 仅用于诊断线性化误差。"
            )

            st.markdown("**预测状态与更新决策**")
            summary = measurement_tables[
                "measurement_update_summary.csv"
            ]
            st.dataframe(summary, width="stretch", hide_index=True)

            observation_tab, state_tab, resolve_tab = st.tabs([
                "测量与创新",
                "状态修正",
                "物理重求",
            ])
            with observation_tab:
                st.dataframe(
                    measurement_tables["measurement_innovation.csv"],
                    width="stretch",
                    hide_index=True,
                )
            with state_tab:
                st.dataframe(
                    measurement_tables[
                        "predicted_posterior_comparison.csv"
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    measurement_tables["stage_covariance_trace.csv"],
                    width="stretch",
                    hide_index=True,
                )
            with resolve_tab:
                source = result[update.source_topology_step_id]
                prior_solution = source["solution"]
                posterior_solution = update.solution
                physical_row = pd.DataFrame([{
                    "source_topology_step":
                        update.source_topology_step_id,
                    "operator_set_id":
                        update.resolve_requirement.source_operator_set_id,
                    "active_interface_ids": ";".join(
                        source.get("active_interface_ids", [])
                    ),
                    "global_dimension": len(source["q"]),
                    "q_base_norm": float(
                        np.linalg.norm(update.q_operator_base)
                    ),
                    "q_source_effective_norm": float(
                        np.linalg.norm(update.q_source_effective)
                    ),
                    "eta_increment_norm": float(
                        np.linalg.norm(update.eta_increment)
                    ),
                    "q_update_increment_norm": float(
                        np.linalg.norm(update.q_update_increment)
                    ),
                    "q_base_semantics": update.q_base_semantics,
                    "linearization_error_norm": float(
                        np.linalg.norm(update.linearization_error)
                    ),
                    "physical_residual_improved":
                        update.physical_residual_improved,
                    "lambda_sum_prior": float(
                        np.sum(prior_solution.lambda_n)
                    ),
                    "lambda_sum_posterior": float(
                        np.sum(posterior_solution.lambda_n)
                    ) if posterior_solution is not None else np.nan,
                    "active_count_prior":
                        len(prior_solution.active_indices),
                    "active_count_posterior": (
                        len(posterior_solution.active_indices)
                        if posterior_solution is not None else 0
                    ),
                    "complementarity_residual":
                        update.physical_residuals.get(
                            "complementarity_residual", np.nan
                        ),
                    "resolve_status":
                        update.resolve_requirement.quality_flag,
                }])
                st.dataframe(
                    physical_row, width="stretch", hide_index=True
                )

            st.markdown("**PREDICTED → POSTERIOR → NEXT_PREDICTED 状态链**")
            lineage = topology_step_state_lineage_table(result)
            relevant_ids = {
                update.predicted_state_id,
                update.posterior_state_id,
            }
            route_keys = list(result)
            checkpoint_position = route_keys.index(
                update.topology_step_id
            )
            if checkpoint_position + 1 < len(route_keys):
                next_step = result[route_keys[checkpoint_position + 1]]
                next_state = next_step.get("predicted_stage_state")
                if next_state is not None:
                    relevant_ids.add(next_state.stage_state_id)
            if "stage_state_id" in lineage.columns:
                lineage = lineage[
                    lineage["stage_state_id"].astype(str).isin(
                        relevant_ids
                    )
                ]
            st.dataframe(lineage, width="stretch", hide_index=True)

            rollback = measurement_tables[
                "update_rollback_record.csv"
            ]
            st.markdown("**回滚记录**")
            if rollback.empty:
                st.caption("本次后验已接受，无回滚记录。")
            else:
                st.dataframe(
                    rollback, width="stretch", hide_index=True
                )

            downstream = topology_step_execution_table(result)
            downstream = downstream[
                pd.to_numeric(
                    downstream["step_order"], errors="coerce"
                )
                > pd.to_numeric(
                    result[update.topology_step_id][
                        "topology_step_spec"
                    ].step_order,
                    errors="coerce",
                )
            ]
            st.markdown("**后续步骤影响**")
            st.dataframe(
                downstream[[
                    column for column in (
                        "topology_step_id", "parent_stage_state_id",
                        "state_role", "effective_state_id",
                        "measurement_update_status",
                        "covariance_trace", "state_correction_norm",
                        "q_correction_norm",
                    ) if column in downstream.columns
                ]],
                width="stretch",
                hide_index=True,
            )
