# 后验状态驱动的虚拟 SMS 滚动预测：实现约束

> 文档性质：在确定性 `topology_step` 与阶段实测后验更新之上的增量实现合同  
> 适用范围：从已接受后验状态出发，对未来零件的显式虚拟 SMS 样本执行剩余装配路线  
> 数据真实性：显式合成样本只用于数值一致性、软件联调和算法自洽，不允许工程精度或概率声明

## 1. 解释顺序与规范关系

1. 当前任务合同优先于旧文档中“尚未实现滚动预测”的阶段性范围描述。
2. `docs/reference/00_data_structure_v25_full.pdf` 规定通用对象、字段、KCP 聚合与防重复规则。
3. `docs/reference/01_data_structure_v25_multipart.pdf` 在第 2、3、6、7、8、11、14 章及附录 A 的多零件条款内增量覆盖完整说明书。
4. 公共 `VectorLayout`、跨接口 `W_struct` 非对角块和统一全局 LCP 不得因滚动预测退化。
5. 当前代码采用预计算步骤算子；本轮只通过显式 `G_eta` 与 `G_SMS` 修正 `q`，不声称在线重建 FE/Schur 算子。
6. 当前 KCP 实现及数据包声明的 `aggregation_policy` 保持不变；最终绝对态与阶段增量不得混用。

## 2. 滚动切断点与源状态

滚动计划必须显式给出：

```text
source_checkpoint_id
source_topology_step_id
source_posterior_state_id
prediction_start_step_id
prediction_end_step_id_optional
```

正式路线的 `source_state_policy` 为 `REQUIRE_ACCEPTED_POSTERIOR`。源状态必须同时满足：

- checkpoint、topology 与 source step 外键闭合；
- `posterior_accepted=true`；
- `state_role=POSTERIOR`；
- `quality_flag=PASS`；
- 位于 `prediction_start_step_id` 之前；
- 当前子装配体、活动接口、连接锁定与释放历史可解析；
- 参数、SMS、`W_struct`、`Cn` 与 MatrixManifest 哈希与后验接受时一致。

缺少已接受后验时必须阻断，不得静默回退到 predicted。`PREDICTED_CUTOFF_BASELINE` 只允许作为显式对照分支，并与正式路线共享除 source state 外的全部输入。

## 3. 历史状态与未来零件 SMS 边界

1. 已装子装配体从源后验状态继续，不重新初始化为自由态 SMS。
2. 已装零件的历史 SMS 不进入 future SMS correction。
3. 只有 `future_part_ids` 中、源子装配体尚未包含的零件可以读取虚拟 SMS。
4. 未来零件在 `first_effective_topology_step_id` 首次加入时生效。
5. 生效后在该样本的后续有效步骤持续作用，直到显式 `last_effective_topology_step_id`。
6. 加入前不得要求映射或产生修正；加入后缺失映射必须阻断。
7. 显式零作用必须由 MatrixManifest 中的 `EXPLICIT_ZERO_NO_EFFECT` 映射表达，不能以缺失矩阵代替。
8. 运行时不得修改或写回原始 SMS、后验状态或数据包。

## 4. 显式虚拟 SMS 样本

第一版只支持：

```text
sample_nature = EXPLICIT_SYNTHETIC_VIRTUAL_SMS_LIBRARY
generation_method = DETERMINISTIC_EXPLICIT_VALUES
mapping_semantics = DELTA_FROM_OPERATOR_REFERENCE
scenario_type = DETERMINISTIC_BASELINE
probability_interpretation_allowed = false
engineering_claim_allowed = false
```

每个 part 的系数向量按 `VirtualSMSComponent.component_order` 显式装配：

```text
Delta alpha_p^(m) = alpha_p^(m) - alpha_p,ref
```

禁止依赖 CSV 自然行序、随机抽样、从 `P_post` 或 SMS 协方差采样、未来载荷随机化、界面参数随机化、分布拟合、Sobol/Shapley、Pf 或概率置信区间。

## 5. q 分解与求解

每个未来步骤必须显式保存：

```text
q_operator_base
q_posterior_state_correction
q_virtual_sms_correction_by_part
q_future_process_correction
q_effective
```

并满足：

```text
q_effective
  = q_operator_base
  + G_eta eta_current
  + sum_p G_SMS,p,k Delta alpha_p
  + Delta q_process,k
```

第一版正式包的 `Delta q_process,k` 为显式零向量。每个步骤都从自己的 operator base 构造 `q_effective`，不得复用上一求解步骤的有效 q。当前后验完整 `eta` 每步只应用一次；同一 `part/sample/step` 的 SMS 修正只应用一次。

`G_SMS,p,k` 必须满足：

- 行数等于公共 contact `VectorLayout` 全局维数；
- 列数等于 part 的 SMS layout 维数；
- row/column layout、key、shape、dtype、operator set、part、reference SMS、mapping semantics 与 MatrixManifest 一致；
- 有明确 `derivation_source` 与真实性标签。

正式接触求解保持：

```text
W_total = W_struct + Cn
q_active = q_effective[active_indices]
W_active = W_total[np.ix_(active_indices, active_indices)]
0 <= lambda ⟂ q_active + W_active lambda >= 0
```

同一步全部活动接口只调用一次全局 LCP。禁止逐接口或逐未来零件求解后拼接；必须保留跨接口非对角块。

## 6. 从中间状态继续 topology_step

滚动执行必须复用 `core/topology_step.py` 的规范化步骤和 `_execute_step` 核心逻辑。入口只执行 start/end 范围内步骤，并满足：

- 第一未来状态的父状态是所选 source posterior/predicted state；
- 不重新执行历史步骤；
- 不重新生成历史 JOIN/RELEASE 记录；
- 历史锁定与释放只读继承；
- 未来非求解步骤仍完整继承机械状态；
- 每个样本生成独立 state ID、结果、账本、trace、质量门和可变状态；
- 样本之间只共享只读输入矩阵；
- 正序、逆序、单独和批量执行结果一致；
- 单样本失败按策略隔离，不污染其他样本。

第一版滚动路线不再执行新的 measurement checkpoint；正式 04 包仅使用一个历史 accepted posterior checkpoint。

## 7. KCP 与贡献防重复

1. 每个成功样本执行完整剩余路线后调用当前正式 KCP 入口。
2. 保留数据包既有 `aggregation_policy`，不得静默切换绝对态/增量策略。
3. 未来 SMS 对接触状态的间接作用通过 `q -> LCP -> final state -> KCP` 进入。
4. 若 KCP 路径声明 future SMS 直接几何项，则该项按 `part/sample/KCP/source_id` 只登记一次。
5. 若最终绝对状态已经包含同一几何来源，不再叠加直接项。
6. 每个样本使用独立 `ContributionLedger`，唯一键与 `DoubleCountCheckResult` 必须 PASS/WARN 且无致命冲突。
7. KCP 输出必须有限、带单位、容差状态、聚合策略、账本 ID、double-count 状态与质量状态。

## 8. 汇总与真实性

多样本只允许输出显式样本集合的：

- count、sample mean、sample std、min、max；
- empirical P05/P50/P95；
- tolerance exceedance count；
- `EMPIRICAL_SAMPLE_FRACTION_NOT_FAILURE_PROBABILITY`；
- contact mode counts；
- success/failure counts。

页面、报告和 CLI 必须明确：

```text
probability_interpretation_allowed=false
engineering_claim_allowed=false
```

不得把 predicted cutoff 描述为真实基准，不得把显式样本频率描述为 failure probability。

## 9. 验证数据与不可变性

`VALIDATE` 数据只用于运行后评价，不得进入 source state、SMS、q、LCP、KCP 或样本选择。运行前后比较：

- source posterior/predicted state hash；
- historical topology result、JOIN lock 与 RELEASE history hash；
- parameter、SMS input、`W_struct`、`Cn` 与 MatrixManifest hash。

任何正式输入对象变化均为阻断失败。合成 04 oracle 必须独立构造 `Delta alpha`、`G_SMS @ Delta alpha`、q、活动集 LCP、最终状态和 KCP，不调用 production rolling runner、production topology runner、production KCP aggregation entry 或 production virtual SMS application function。

## 10. 失败分类与发布门

至少区分：

- invalid SMS sample；
- missing mapping；
- layout mismatch；
- source posterior invalid；
- LCP failure；
- physical consistency failure；
- KCP failure；
- double-count failure；
- unexpected exception。

运行时可使用 `FAIL_SAMPLE_CONTINUE` 隔离样本；正式发布检查采用 `FAIL_RUN_IF_ANY_FORMAL_SAMPLE_FAILS`。正式 04 包全部样本必须 PASS。

## 11. 明确未实现

- Monte Carlo 与任何随机数生成；
- 从后验协方差抽样；
- 分布拟合、Pf、概率置信区间；
- Sobol/Shapley；
- 在线 FE/Schur、完整摩擦 NCP；
- 真实 KCP 工程验证；
- 工艺优化、返修、条件分支、并行调度或自动路线重规划。
## 最终提交前补充约束（2026-07-27）

1. `final_state_includes_direct_sms_geometry` 是执行语义，不是说明性字段：
   `false` 执行 `base + direct` 并登记一次 direct 账本；`true` 执行 `base`，
   不登记 `FUTURE_SMS_DIRECT_GEOMETRY`，trace 必须记录真实布尔值和
   `DIRECT_SMS_ALREADY_INCLUDED_IN_FINAL_STATE`。
2. 该字段必须严格解析为布尔值，并与 rolling aggregation policy 兼容；
   非法值、缺失值和同一零件冲突声明为执行前 blocking FAIL。
3. source checkpoint 必须与 plan 的 topology 和 source step 一致；运行时
   measurement update、accepted posterior 与实际 source state 的 checkpoint、
   topology step、state ID、role、accepted/rollback/quality 状态必须闭合。
4. `POSTERIOR_ONLY` 的 baseline 状态为 `NOT_APPLICABLE`，attempt/success/failure
   均为 0，comparison 使用固定 schema 空表，不得伪造 baseline failure。
5. UI 与报告必须安全处理全部 formal 样本失败的空集合，仍展示权威 FAIL、
   样本失败表和质量门，不展示伪 KCP 或伪描述性统计。
