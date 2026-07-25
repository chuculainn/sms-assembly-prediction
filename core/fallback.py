from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import SMSPackage, get_stage_ids
from .numerical_substitution import load_substitution_config, _stage_load_estimate


@dataclass
class FallbackSettings:
    enabled: bool = True
    warn_on_validity: bool = True
    fail_to_recommend_hifi: bool = True
    max_complementarity_residual: float = 1e-4
    max_gap_violation: float = 1e-7
    max_active_set_iterations_warn: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'FallbackSettings':
        if data is None:
            return cls()
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _status_rank(status: str) -> int:
    return {'PASS': 0, 'INFO': 0, 'WARN': 1, 'FAIL': 2}.get(status, 1)


def evaluate_validity_and_fallback(
    pkg: SMSPackage,
    result: dict[str, dict],
    settings: FallbackSettings | dict[str, Any] | None = None,
    sms_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    settings = FallbackSettings.from_dict(settings) if not isinstance(settings, FallbackSettings) else settings
    rows: list[dict[str, Any]] = []

    # Module validity table: load ranges for numerical-substitution modules.
    tables = load_substitution_config(pkg)
    validity = tables.get('validity', pd.DataFrame())
    if isinstance(validity, pd.DataFrame) and not validity.empty:
        for _, r in validity.iterrows():
            module = str(r.get('module_name', r.get('module_id', 'module')))
            stages = str(r.get('valid_stage_ids', 'ALL'))
            load_min = float(r.get('valid_load_min_N', -np.inf))
            load_max = float(r.get('valid_load_max_N', np.inf))
            for sid in get_stage_ids(pkg):
                if stages not in ('ALL', '*') and sid not in stages.split('|'):
                    continue
                load = _stage_load_estimate(pkg, sid)
                ok = load_min <= load <= load_max
                rows.append({
                    'check_item': f'适用域:{module}:{sid}',
                    'status': 'PASS' if ok else 'WARN',
                    'detail': f'load={load:.2f} N, valid=[{load_min:.2f},{load_max:.2f}] N',
                    'fallback_action': '快速模型可用' if ok else '回退/补充局部FE或样件标定以扩展D_valid',
                    'required_data': 'module_validity.csv中的载荷、压力、材料、表面状态适用域',
                })
    else:
        rows.append({
            'check_item': 'module_validity.csv', 'status': 'WARN',
            'detail': '未提供数值替代模块适用域',
            'fallback_action': '默认只作演示计算，工程预测前需补适用域',
            'required_data': '每个模块的D_valid：材料组合、压力/载荷范围、温度、表面状态、验证误差',
        })

    # Solver residual gates. NOT_REQUIRED is a valid state event, not a failed
    # or skipped LCP.
    result_by_step = {
        str(res.get("topology_step_id", key)): res for key, res in result.items()
    }
    for sid, res in result.items():
        sol = res['solution']
        solve_status = str(res.get("solve_status", sol.convergence_status)).upper()
        if solve_status == "NOT_REQUIRED":
            state = res.get("stage_state")
            action = str(
                getattr(state, "mechanical_state_action", "")
                or res.get("mechanical_state_action", "")
            )
            reason = str(
                getattr(state, "not_required_reason", "")
                or res.get("not_required_reason", "")
            )
            parent_step_id = str(getattr(state, "parent_topology_step_id", "") or "")
            inherited_ok = True
            if action == "INHERIT_PARENT_UNCHANGED":
                parent_result = result_by_step.get(parent_step_id)
                inherited_ok = parent_result is not None and all(
                    np.allclose(
                        np.asarray(res[name], dtype=float),
                        np.asarray(parent_result[name], dtype=float),
                        equal_nan=True,
                    )
                    for name in ("lambda_full", "gap_full", "pressure", "local_compression")
                )
            valid_action = action in {"INITIALIZE_EMPTY", "INHERIT_PARENT_UNCHANGED"}
            status = "PASS" if valid_action and inherited_ok else "FAIL"
            rows.append({
                "check_item": f"机械状态:{sid}",
                "status": status,
                "detail": (
                    f"solve_status=NOT_REQUIRED; reason={reason}; "
                    f"mechanical_state_action={action}; inherited_unchanged={inherited_ok}"
                ),
                "fallback_action": (
                    "无需LCP；保留状态链和机械状态语义"
                    if status == "PASS" else
                    "修复NOT_REQUIRED父机械状态继承"
                ),
                "required_data": "parent_state_id、active_index_mask、父机械响应",
            })
            continue
        gap_v = sol.residuals.get('gap_violation', 0.0)
        comp = sol.residuals.get('complementarity_residual', 0.0)
        it = sol.iteration_count
        status = 'PASS'
        action = '快速模型可用'
        if sol.convergence_status != 'CONVERGED' or comp > settings.max_complementarity_residual or gap_v > settings.max_gap_violation:
            status = 'FAIL'
            action = '回退高保真FE或检查q/W/Cn符号与半正定性'
        elif it > settings.max_active_set_iterations_warn:
            status = 'WARN'
            action = '主动集迭代较多，建议检查接触点密度或用高保真校核代表样本'
        rows.append({
            'check_item': f'LCP残差:{sid}',
            'status': status,
            'detail': f"status={sol.convergence_status}, comp={comp:.3e}, gap_v={gap_v:.3e}, iter={it}",
            'fallback_action': action,
            'required_data': 'q_vector、W_total、Cn_local、防重复柔度证明、代表性高保真对比',
        })

    # SMS mapping quality gate.
    if sms_quality is not None and not sms_quality.empty:
        for _, r in sms_quality.iterrows():
            status = str(r.get('status', 'WARN'))
            rows.append({
                'check_item': f"SMS映射:{r.get('check_item', '')}",
                'status': status,
                'detail': r.get('detail', ''),
                'fallback_action': 'PASS则可用；WARN/FAIL时应补点云、KCM或改用包内冻结g0',
                'required_data': 'FREE/KNOWN_SUPPORT状态SMS点云、KCM、sigma/R_m、坐标变换、补偿记录',
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    max_rank = max(_status_rank(s) for s in out['status'].astype(str))
    final = 'PASS' if max_rank == 0 else ('WARN' if max_rank == 1 else 'FAIL')
    recommendation = {
        'PASS': '快速模型可用于当前演示/适用域内预测',
        'WARN': '可运行但需标注不确定性；建议补充样件/局部FE或检查适用域',
        'FAIL': '不建议作为预测结果；应回退局部高保真FE/样件实验或重新测量输入',
    }[final]
    out.loc[len(out)] = {
        'check_item': '总体回退决策',
        'status': final,
        'detail': recommendation,
        'fallback_action': recommendation,
        'required_data': '见各行 required_data',
    }
    return out


def remaining_limitations_table() -> pd.DataFrame:
    rows = [
        {'模块': 'CAD/FE自动接触域生成', '剩余不足': '当前仍主要读取已有ContactDomain；没有直接读取CATIA/Abaqus几何自动采样。', '需要数据/接口': 'CAD面ID、FE节点/单元集、master/slave对应、法向规则、面积权重收敛准则。'},
        {'模块': 'Schur/Craig-Bampton/POD建库', '剩余不足': '当前读取W_struct；没有从全阶K自动分区、凝聚和二次降阶。', '需要数据/接口': '全阶刚度K、DOF分区、界面DOF、边界约束、材料/铺层版本、全阶FE校核结果。'},
        {'模块': 'N-2-1扩展LCP', '剩余不足': '已能拼接locator/clamp/joint扩展点并求解，但耦合项仍为简化核函数，不等价于真实全局扩展B_ext。', '需要数据/接口': '真实B_ext、W_struct_ext、定位器/夹持头/孔边有限刚度标定、连续卡板离散收敛、高保真反力基准。'},
        {'模块': '法向-切向NCP', '剩余不足': '已实现局部摩擦锥投影，但法向与切向未完全耦合迭代。', '需要数据/接口': 'Bt映射、切向自由滑移q_t、Ct/μ分区参数、粘滑历史、摩擦试验/局部FE对比。'},
        {'模块': 'CG-FFT/BEM粗糙接触', '剩余不足': '当前粗糙接触仍是等效柔度公式，不是微观粗糙面求解器。', '需要数据/接口': '表面粗糙度高度图、功率谱/相关长度、微观材料参数、边界压力、CG-FFT/BEM校核结果。'},
        {'模块': '孔边连接细节', '剩余不足': '当前C_joint只作等效柔度，孔壁承压、紧固件沉降、微滑移历史仍简化。', '需要数据/接口': '孔位/孔径/边距、紧固件参数、扭矩-轴力、孔边局部FE、连接锁定历史。'},
        {'模块': '自动高保真回退', '剩余不足': '已给出回退建议，但不会自动调用Abaqus/CG-FFT求解。', '需要数据/接口': 'Abaqus/求解器命令接口、局部子模型模板、边界继承记录、任务队列和结果解析脚本。'},
        {'模块': '真实验证闭环', '剩余不足': '内置数据仍为合成演示，不能代表真实垂尾翼盒验证精度。', '需要数据/接口': '真实SMS、过程力/位移/预紧、装配后KCP、样件试验、局部高保真对比数据。'},
    ]
    return pd.DataFrame(rows)
