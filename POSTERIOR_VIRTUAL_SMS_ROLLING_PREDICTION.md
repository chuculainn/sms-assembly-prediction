# 后验状态驱动的虚拟 SMS 滚动预测

## 目标与边界

本功能在某个 measurement checkpoint 接受后验状态后，从 rolling cutoff 的下一装配步骤继续确定性执行。已有子装配体继承 accepted posterior 的几何、接触、连接锁定与协方差状态；尚未加入的 future parts 使用显式虚拟 SMS 样本。正式入口为：

```python
run_posterior_virtual_sms_rolling_prediction(
    package,
    topology_result,
    rolling_plan_id,
    virtual_sms_sample_ids=None,
)
```

UI、CLI、测试和报告都调用该入口，不另行拼装数值链。

## 输入对象

`I_pred` 中的 rolling plan 定义 source checkpoint、source topology step、prediction start/end、future parts、显式样本集、确定性未来工艺场景、KCP 集与聚合策略。虚拟 SMS 库由 sample set、sample、component 和 coefficients 四类表构成：

- `coefficient_source=REFERENCE_SMS` 的唯一样本定义 operator reference；
- 任意样本的 `Delta alpha = alpha_sample - alpha_reference`；
- component 按数据中的 `component_order` 连续装配，不假设固定维数；
- 样本为显式确定性值，不从后验协方差或 SMS 协方差抽样。

`future_sms_assignment.csv` 约束样本从何时作用，`sms_operator_mapping.csv` 将 future part、operator set 和 `G_SMS` 唯一关联。MatrixManifest 必须声明公共接触行布局和 SMS 列布局。

## 数值链

每个未来正式求解步使用：

```text
q_effective
  = q_operator_base
  + q_posterior_state_correction
  + q_virtual_sms_correction
  + q_future_process_correction

q_virtual_sms_correction(part)
  = G_SMS(part, operator_set) @ Delta alpha(part)
```

`q_operator_base` 始终来自当前步骤自己的冻结算子，不使用上一阶段的 effective q 作为新基线。accepted posterior 的低维状态通过既有 `G_q` 只应用一次；future SMS 每个 part/sample/step 只应用一次。reference sample 的 `Delta alpha` 与 virtual SMS correction 均为零。

每个样本从 source posterior 深复制出独立 state branch，仅执行 cutoff 之后的步骤。所有活动接口继续使用公共 `VectorLayout`，保留 `W_struct` 跨接口非对角块，并在每个正式求解步一次性求解：

```text
W_total = W_struct + Cn
0 <= lambda ⟂ W_total lambda + q_effective >= 0
```

不允许逐接口求解后拼接。非求解步骤仍传递状态；JOIN 形成的锁定历史在 RELEASE 按数据语义继承。

## KCP、账本与对照

每个成功样本通过正式 `extract_kcp` 入口计算 KCP，再按冻结 aggregation policy 加入 future SMS direct geometry 项。贡献账本以来源唯一键判重，并重构最终 KCP；duplicate count 或 reconstruction error 超限会使 double-count gate 失败。

审计收尾后，样本、运行、报告、CLI 和第 16 页统一读取同一权威状态。任何 KCP double-count、KCP quality、SMS application 完整性或输入不可变性失败都会使正式 run 为 `FAIL`；其他样本仍继续执行以保留失败隔离证据，但不能把部分成功误报为整体 PASS。

正式结果来自 source posterior。可选的 predicted-cutoff baseline 对相同样本、相同未来步骤和相同算子再运行一次，二者的唯一基线差异是 cutoff source state。报告输出逐样本 KCP、posterior-minus-predicted、接触模式和失败隔离记录。

`baseline_comparison_policy` 实际控制是否运行 baseline。正式 04 使用必需对照策略；posterior-only 策略不运行 baseline。两类失败分别统计，baseline 不进入正式 `sample_count/success_count/failure_count`。

## 生效区间、映射角色与治理

- assignment 的 first/last 必须存在、顺序正确并位于 rolling 区间；first 必须等于 future part 的首次加入步骤，SMS 持续到 rolling end。
- 每个 sample/part/适用求解步必须恰有一条 application trace；漏项、提前出现或 `application_count != 1` 均为 FAIL。
- 普通 `EFFECTIVE_MAPPING` 必须为非零矩阵；只有 mapping 与 MatrixManifest 同时声明 `EXPLICIT_ZERO_NO_EFFECT` 时才允许零矩阵，且 application trace 仍存在。
- `kcp_set_id`、aggregation/baseline/failure policy、全部正式行的 `quality_flag`、reference SMS 和 coefficient unit 都是执行前阻断门，不只是报告字段。
- 完整 `topology_result`（包括 prediction start 与未来原始步骤）和 package 输入在运行前后采用稳定规范化哈希比较；任何变化使 run FAIL。

## 描述性汇总与真实性

样本汇总只报告 count、mean、sample std、min/max、经验 p05/p50/p95、超差样本数和显式样本占比。该占比固定标记为：

```text
EMPIRICAL_SAMPLE_FRACTION_NOT_FAILURE_PROBABILITY
```

本轮没有 Monte Carlo、随机数、分布拟合、后验抽样、Sobol/Shapley 或 Pf。`probability_interpretation_allowed=false`，`engineering_claim_allowed=false`。

## 04 合成最小数据包

`data/04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE` 从已审计的 03 包确定性构建，source 为 accepted posterior，未来加入一个零件，包含 2 维 SMS layout 和 5 个显式样本：

- reference `[0.0, 0.0]`
- mode-1 positive `[1.0, 0.0]`
- mode-1 negative `[-1.0, 0.0]`
- mode-2 positive `[0.0, 1.0]`
- combined `[0.75, -0.5]`

包内 oracle 使用独立 active-set 枚举和直接 KCP 公式，不调用生产 rolling/topology/KCP runner。它仅证明数据结构、数值链和软件实现自洽，不代表真实工程精度。

## 报告与界面

完整报告新增 15 张 rolling CSV 与 `rolling_prediction_trace.json`，覆盖计划、样本、系数、assignment、mapping、逐步执行、SMS 应用、状态谱系、KCP、对照、汇总、接触模式、失败和质量门。

CLI 对旧包输出 count 0 和 `ROLLING_FINAL_STATUS=NOT_APPLICABLE`；配置 plan 的包按唯一 `virtual_sms_sample_id` 统计正式样本，并分别输出 posterior、baseline、application、double-count 和 immutability 计数。配置错误使用退出码 1，运行/物理/KCP/防重复/不可变失败使用退出码 2。Streamlit 第 16 页直接显示权威 run status、两类样本计数和各专项质量门。

## 当前限制

- future process 第一版只支持显式确定性 q correction；
- 不在线重建全阶 FE/Schur 算子；
- 不做状态—参数联合估计或接口参数随机化；
- 不做概率预测、可靠性分析或生产级工程验证；
- 04 仅为最小集成基准，核心没有零件数、接口数、接触点数或向量维数上限。

## 最终提交前定向修复语义

- `final_state_includes_direct_sms_geometry=false` 时，正式 KCP 为
  `base_kcp_value + direct_sms_contribution`，direct 来源在贡献账本中只登记一次。
- `final_state_includes_direct_sms_geometry=true` 时，最终绝对状态已包含该直接几何来源，
  正式 KCP 只使用 `base_kcp_value`，不再叠加 direct contribution，账本动作记录为
  `DIRECT_SMS_ALREADY_INCLUDED_IN_FINAL_STATE`。该字段只接受严格的 `true/false`，
  非法值、同一零件冲突声明或不兼容 aggregation policy 均在执行前阻断。
- rolling source 必须闭合为同一条
  `plan -> checkpoint -> measurement update -> accepted posterior -> actual source state`
  链。checkpoint topology/step、update ID 与 posterior ID、实际 source role/checkpoint/state
  任一不一致均为 blocking FAIL，不允许回退到 predicted state。
- `POSTERIOR_ONLY` 不运行 predicted-cutoff baseline；三个 baseline 计数均为 0，
  状态为 `NOT_APPLICABLE`，comparison 返回固定 schema 空表，不生成
  `PREDICTED_FAILED`。
- 第 16 页和 rolling 报告支持 0 个成功样本、0 条 KCP、0 条贡献账本和
  0 条 comparison；空结果显示明确提示，不生成零值伪 KCP 或描述性统计。
