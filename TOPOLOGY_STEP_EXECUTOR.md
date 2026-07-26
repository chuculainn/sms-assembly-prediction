# topology_step 确定性多轮装配执行器

## 已实现范围

执行入口为：

```python
run_topology_steps(package, topology_id=None, sample_id="SAMPLE_001")
```

真实路线优先读取 `I0/assembly_topology.csv`，逐行适配为 `TopologyStepSpec`，再按 `step_order + topology_step_id` 稳定排序。多值字段支持分号分隔，在执行前解析为去重且保持输入顺序的 ID 元组。`run_all_stages(...)` 仍是兼容入口，但内部调用同一执行器。

每一步读取其 `parent_topology_step_id` 对应的 `StageState`，深复制零件、接口和锁定状态，然后应用本步增删集合。已有零件继承上一状态，不重新读取自由零件 SMS；只有 `added_part_ids` 中首次出现的零件创建 `initialization_source=ASSEMBLY_BEFORE_SMS`、`initialization_count=1` 的 PartState。每步产生新的 state ID，父对象不会被后续步骤覆盖，最终状态可回放至 TS000。

`solve_required=false` 分为两种确定语义：

- 无父状态 INIT：`solve_status=NOT_REQUIRED`、`mechanical_state_action=INITIALIZE_EMPTY`，初始化空机械状态；
- 有父状态的 MEASURE/INSPECT 等事件：`mechanical_state_action=INHERIT_PARENT_UNCHANGED`，完整继承 lambda、gap、pressure、local_compression、contact mode、contact_structural_response、活动接口/mask 与连接状态，只生成新的 state ID、父引用和事件元数据，不调用 LCP。

带父状态的非求解步骤若声明零件、接口、边界、载荷或连接的机械集合变化，会在执行前阻断，避免“未求解却改变机械状态”。

## 统一耦合 LCP

当前活动接口的全局索引是公共 VectorLayout 各接口闭区间的并集：

```python
q_active = q_full[active_indices]
W_active = W_total_full[np.ix_(active_indices, active_indices)]
W_total_full = W_struct_full + Cn_full
```

每个 `solve_required=true` 步骤仅调用一次 `solve_lcp_active_set(q_active, W_active)`。没有逐接口求解、结果拼接或非对角块清零。求解后 `lambda_active/gap_active` 回填公共布局；inactive 的 `lambda_full=0`、`gap_full=NaN`，并另存 `active_index_mask`，因此 inactive 不会被误解释为零间隙活动接触。

每步保存 `ContactComputationTrace`、完整/活动向量、活动/非活动接口、算子 key、残差、主动集、压力、局部压缩和 LCP 调用次数。真实路线的来源固定标记为 `PRECOMPUTED_TOPOLOGY_STEP_OPERATOR`，准确表示读取了数据包预计算算子，并不表示在线全阶 FE 重建。

## JOIN、RELEASE 与状态传递

一个 JOIN topology_step 生成一条唯一 ConnectionLockHistory；该记录可同时包含多个 `joint_ids`，并保存 locked reference、预紧输入来源、连接刚度 ID 与锁定时接触模式。锁定历史 ID 进入后续所有状态。

RELEASE 删除路线声明的边界和载荷，读取此前全部 lock history，并按以下顺序得到新连接集合：

1. 继承父状态 `active_joint_ids`；
2. 应用本步 `activated_joint_ids`；
3. 优先应用本步 `deactivated_joint_ids`；
4. 对其余 joint 应用 JointDefinition 的 `retention_rule`；
5. 保存 retained、removed 和 `active_joint_ids_after_step`，供后续步骤继承。

当前支持包内实际规则 `RETAIN_THROUGH_RELEASE`，并支持提交前收尾测试所需的 `REMOVE_AT_RELEASE`。未知或空白规则是阻断错误，不默认保留。RELEASE 保存一条 ReleaseHistoryRecord，并对释放后的全部活动接口重新执行一次统一 LCP。锁定历史独立于 JointDefinition，后者只提供连接定义和保留规则。

## 专项质量门

23 项求解前专项门包括：步骤 ID/顺序、父链解析/无环/无未来引用、零件与接口外键、活动接口端点成员关系、同一步多接口激活、原始平行接口不丢失、边界/载荷/连接引用、结果子装配体、算子解析、legacy 标记、VectorLayout 连续覆盖、MatrixManifest key/shape/dtype/row-column layout、活动交叉块实际范数、retention rule、非求解机械集合不变、状态父链和合成数据真实性标记。严重结构错误抛出 `TopologyStepValidationError`，不会仅警告后继续正式求解。

旧数据包历史 StageInput 中存在早于 canonical BoundaryItem/LoadItem 定义的 stage-scoped 标签。仅在 `adapter_source=LEGACY_STAGE_ADAPTER` 时保留既有兼容语义；真实 topology_step 表仍严格执行外键阻断。

## 合成基准路线

`data/02_TOPOLOGY_STEP_MIN_CASE` 包含 TS000、TS101–TS104、TS201–TS204、TS301–TS304。TS301 同时激活 G_AD 与 G_DB，并与既有 G_AB、G_BC 组成一个 12 维问题，只调用一次全局 LCP。每个求解步骤提供可追溯的 q、W_struct、Cn、W_total、operator_set_id、VectorLayout、MatrixManifest 和独立主动集枚举 oracle。

该包设置 `data_nature=SYNTHETIC_NUMERICAL_CONSISTENCY_CASE` 与 `engineering_claim_allowed=false`。它只证明软件执行链和数值自洽，不代表真实结构、工艺或 KCP 精度。

生成器 `scripts/build_topology_step_fixture.py` 可重复完整重建该包；校验器由当前正式模板同步，不复制旧包 Python 或旧验证结果。包内字段字典覆盖实际 CSV schema 和 24 个 TopologyStepSpec 字段；对象映射登记输入、算子、oracle、连接历史、执行报告与验证结果。当前自校验为 22/22 PASS，MatrixManifest 与 NPZ 均为 156 个唯一 key。

## 预计算算子参数与 CLI 语义

真实路线使用预计算步骤算子，本轮不在线重构 q/W/Cn。因此 Streamlit 禁用 sms/closure/cn 倍率和会改变算子的高级控件，并隐藏基于这些无效倍率的 topology Monte Carlo/敏感性入口。运行 trace、execution table 与 operator usage 报告均记录 `parameter_effective=false` 和 `PRECOMPUTED_TOPOLOGY_STEP_OPERATOR_DISABLED`；用户输入值不会保存成已应用参数。Legacy 路径继续使用既有运行时倍率。

CLI 的稳定退出码为 0=PASS/允许 WARN，1=阻断校验失败（包括包内校验器失败），2=运行时/求解/总体物理失败，3=未预期异常。尾部固定输出：

```text
FINAL_STATUS=PASS|FAIL
BLOCKING_FAIL_COUNT=<integer>
PHYSICAL_FAIL_COUNT=<integer>
```

fallback 与物理一致性报告不对 `NOT_REQUIRED` 要求 LCP 收敛、互补或平衡残差；它们检查状态链、action/reason、活动 mask 和中间继承一致性。`NOT_REQUIRED` 不是 SKIP，也不是求解失败。

## Legacy 兼容

缺少真实 topology_step 表时，适配器从 StageDefinition/StageInput 生成四个 `LEGACY_TS_*`。旧的 `run_stage` 求解和高级诊断字段保持不变，结果记录：

```text
operator_source=LEGACY_STAGE_OPERATOR
fallback_flag=true
fallback_reason=LEGACY_STAGE_COMPATIBILITY
```

这条路线只表示旧四阶段兼容回退，不宣称是真实多轮工艺路线。

## 当前未实现

- 后续虚拟 SMS 滚动预测和未来场景 Monte Carlo；
- 在线全阶 K/FE Schur 凝聚；
- 完整法向—切向摩擦 NCP、几何/材料非线性；
- JSS/J-T 自动构建、真实 KCP 独立验证；
- 条件分支、返修、并行调度、路线优化和生产级数字孪生平台。

阶段实测后验更新已作为增量层接入：checkpoint 同时保留 PREDICTED/POSTERIOR，状态修正以显式 G_q 进入 q，并对当前活动接口执行一次统一全局 LCP 重求。详细合同、数据治理、回滚和真实性边界见 `STAGE_MEASUREMENT_POSTERIOR_UPDATE.md`。
