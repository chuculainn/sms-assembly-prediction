# 论文结构冻结：`topology_step` 执行器实现约束

> 文档性质：论文冻结主线向软件实现层的约束合同  
> 适用任务：真实工艺路线表驱动的确定性多轮装配执行器  
> 当前阶段：仅实现 `topology_step` 执行主链  
> 明确不包含：阶段实测后验更新、后续虚拟 SMS 滚动预测、Monte Carlo 与真实工程验证

---

## 1. 文档目的

本文件用于把论文结构冻结后已经确认的研究场景、术语边界、物理逻辑和实现范围转换为 Codex 可执行的软件约束。

本文件不是论文正文，也不是对现有技术说明书和数据结构说明书的替代。其主要作用是：

1. 规定 `topology_step` 执行器必须表达的真实装配流程；
2. 防止软件继续沿用“仅执行一次固定四阶段”的简化逻辑；
3. 约束多零件、多接口、子装配体和连接历史的状态传递；
4. 明确当前哪些能力已经实现、哪些属于本轮目标、哪些仍不得宣称已经实现；
5. 作为后续阶段实测后验更新和虚拟 SMS 滚动预测的前置接口合同。

---

## 2. 研究场景冻结

论文面向已经完成部分制造或装配的异质叠层结构，研究在给定名义模型、装配拓扑、零件 SMS、工艺边界、载荷、界面参数和结构算子的条件下，如何计算多步骤装配过程中的接触状态演化及最终装配精度。

真实装配过程不能仅表示为全结构统一经历一次：

```text
LOCATE → CLAMP → JOIN → RELEASE
```

而应表示为多个真实工艺操作按顺序组成的装配路线。例如：

```text
建立初始子装配体
→ 加入零件 B
→ 定位 B
→ 夹持 B
→ 连接 B
→ 释放部分工装
→ 加入零件 C
→ 再次定位、夹持、连接与释放
→ 加入零件 D，并同时形成多个结合面
→ 完成最终装配
```

本轮软件升级应首先实现上述真实工艺路线的表驱动执行，不实现阶段实测后验更新和未来虚拟 SMS 滚动预测。

---

## 3. 核心术语冻结

### 3.1 `topology`

`topology` 表示一整套装配计划，包括：

- 零件节点；
- 结合面与连接边；
- 子装配体；
- 工艺步骤；
- 零件加入顺序；
- 接口激活顺序；
- 边界、载荷和连接变化；
- 状态继承关系。

### 3.2 `topology_step`

`topology_step` 表示装配计划中的一次唯一、可追溯的原子工艺事件。

每个步骤至少应明确：

- 当前输入子装配体；
- 新加入或移除的零件；
- 激活或失活的接口；
- 激活或撤除的边界；
- 新施加、调整或撤除的载荷；
- 激活、锁定、保留或释放的连接；
- 使用的预计算结构算子；
- 父步骤和父状态；
- 是否需要重新进行接触求解；
- 当前步骤完成后的结果子装配体。

### 3.3 `assembly_cycle`

`assembly_cycle` 表示围绕同一新增零件或同一局部装配任务形成的一组连续操作，可包含：

```text
ADD_PART / LOCATE / CLAMP / JOIN / RELEASE
```

`assembly_cycle_id` 只用于分组，不得替代 `topology_step_id`。

### 3.4 `stage`

`stage` 表示定位、夹持、连接、释放等工艺类型或力学阶段语义。

`topology_step` 与 `stage` 不是同一概念：

- `topology_step` 是装配路线中的唯一事件；
- `stage` 是该事件的工艺类别；
- 同一种 `stage` 可以在整条路线中重复出现多次；
- 不得假设全文只有一组 LOCATE、CLAMP、JOIN、RELEASE。

### 3.5 当前子装配体

当前子装配体是执行到某个 `topology_step` 时已经加入且仍属于装配系统的全部零件、接口和连接的集合。

当前子装配体必须继承前一步骤的装配状态，不得在后续步骤重新退化为各零件的自由态 SMS。

### 3.6 活动接口

活动接口是当前步骤中参与统一接触求解的接口。

活动接口集合可能包括：

- 本步骤新激活接口；
- 前序步骤已经激活且仍保持接触或连接影响的接口；
- 多条串联、并联或闭环路径中的相关接口。

失活接口不应进入当前 LCP，但其历史、锁定状态或贡献记录仍需保留。

---

## 4. 当前软件状态与本轮目标分级

### 4.1 当前已经实现

当前软件已经具备：

- 多零件拓扑表达；
- 串联、并联和闭环关系识别；
- 同一阶段活动接口的全局耦合 LCP；
- 跨接口结构柔度非对角块；
- 聚合、零件和接口状态对象；
- 状态父链；
- 简化 JOIN 锁定与 RELEASE 继承；
- 数据包质量门；
- 接触计算追溯；
- KCP 贡献账本；
- 旧 E1、V2.5 和四零件数据包兼容。

### 4.2 本轮必须实现

本轮只实现：

- 从数据包读取真实工艺路线表；
- 将路线标准化为有序 `TopologyStepSpec`；
- 按 `step_order` 执行多轮装配步骤；
- 动态更新当前子装配成员；
- 动态激活和撤除接口、边界、载荷与连接；
- 每个步骤对全部活动接口统一求解一次全局 LCP；
- 继承父状态、接触模式和连接历史；
- 在 JOIN 保存锁定历史；
- 在 RELEASE 读取锁定历史并重新平衡；
- 保存逐步骤结果、质量门和完整追溯；
- 将旧四阶段数据包自动适配为四个 legacy topology steps。

### 4.3 当前允许保留的简化

本轮允许继续使用数据包预计算的：

- `q`；
- `U_FREE`；
- `W_struct`；
- `C_n`；
- `W_total`；
- 连接刚度或锁定参考；
- 预期 LCP oracle。

本轮不要求在线从全阶 FE 自动重新生成这些量。

每个步骤必须声明算子来源，例如：

```text
PRECOMPUTED_TOPOLOGY_STEP_OPERATOR
LEGACY_STAGE_OPERATOR
```

若采用旧阶段算子兼容回退，必须记录：

```text
fallback_flag = true
fallback_reason = LEGACY_STAGE_COMPATIBILITY
```

### 4.4 本轮尚未实现、不得宣称已经实现

本轮禁止将以下内容描述为已完成：

- 阶段实测后验更新；
- 后续虚拟 SMS 滚动预测；
- 多样本未来场景抽样；
- 全阶 K 矩阵自动 Schur 凝聚；
- 完整法向—切向摩擦 NCP；
- 坐标链和单位归一化自动执行；
- JSS/J-T 自动构建；
- 真实 KCP 独立验证；
- 接触域离散收敛；
- 几何或材料非线性；
- 工艺路线自动优化；
- 生产级实时数字孪生。

---

## 5. 工艺路线数据合同

### 5.1 主表

优先读取：

```text
I0/assembly_topology.csv
```

主表一行代表一个唯一工艺步骤。

建议规范字段：

```text
topology_id
topology_step_id
step_order
parent_topology_step_id
assembly_cycle_id
operation_type
stage_id
input_subassembly_id
result_subassembly_id
added_part_ids
removed_part_ids
activated_interface_ids
deactivated_interface_ids
activated_boundary_ids
deactivated_boundary_ids
activated_load_ids
removed_load_ids
activated_joint_ids
deactivated_joint_ids
operator_set_id
solve_required
reference_state_id
measurement_checkpoint_id
notes
```

第一版允许多值字段使用分号分隔，但进入计算前必须转换为集合或有序列表。

### 5.2 配套关联表

若仓库中已有或后续需要规范化，可使用：

```text
I0/subassembly.csv
I0/subassembly_membership.csv
I0/joint_definition.csv
I0/stage_definition.csv

I_stage/stage_input.csv
I_stage/boundary_item.csv
I_stage/load_item.csv

matrices/vector_layout.csv
matrices/matrix_manifest.csv
```

也可逐步增加：

```text
I0/topology_step_part.csv
I0/topology_step_interface.csv
I_stage/topology_step_boundary.csv
I_stage/topology_step_load.csv
I_stage/topology_step_joint.csv
```

本轮不得为追求字段完美而破坏现有数据包兼容性，应优先使用 schema adapter。

---

## 6. `TopologyStepSpec` 内部对象约束

软件内部应形成统一规范对象，建议包含：

```python
TopologyStepSpec = {
    "topology_id": str,
    "topology_step_id": str,
    "step_order": int,
    "parent_topology_step_id": str | None,
    "assembly_cycle_id": str | None,
    "operation_type": str,
    "stage_id": str | None,
    "input_subassembly_id": str | None,
    "result_subassembly_id": str,
    "added_part_ids": list[str],
    "removed_part_ids": list[str],
    "activated_interface_ids": list[str],
    "deactivated_interface_ids": list[str],
    "activated_boundary_ids": list[str],
    "deactivated_boundary_ids": list[str],
    "activated_load_ids": list[str],
    "removed_load_ids": list[str],
    "activated_joint_ids": list[str],
    "deactivated_joint_ids": list[str],
    "operator_set_id": str | None,
    "solve_required": bool,
    "reference_state_id": str,
    "measurement_checkpoint_id": str | None,
    "notes": str,
}
```

第一版只支持单条确定性线性工艺路线，不实现条件分支、返修、并行事件调度或动态重规划。

---

## 7. 执行逻辑冻结

主执行逻辑必须从：

```python
for stage in [LOCATE, CLAMP, JOIN, RELEASE]:
    solve_stage(stage)
```

升级为：

```python
for step in ordered_topology_steps:
    execute_topology_step(step)
```

每一步至少执行：

1. 读取父步骤后验状态；
2. 复制上一有效状态作为本步预测起点；
3. 更新当前子装配体成员；
4. 加入或移除零件；
5. 激活或撤除接口；
6. 激活或撤除边界；
7. 激活、调整或撤除载荷；
8. 激活、锁定、保留或撤除连接；
9. 解析 `operator_set_id`；
10. 构建活动接口索引；
11. 组装活动接口 `q`；
12. 提取活动接口对应的完整 `W_struct` 子矩阵；
13. 组装 `W_total = W_struct + C_n`；
14. 对全部活动接口统一调用一次 LCP；
15. 回填公共全局布局；
16. 更新聚合、逐零件和逐接口状态；
17. 更新接触模式暖启动；
18. 保存锁定或释放历史；
19. 保存接触计算追溯；
20. 将本步后验状态传给下一步骤。

---

## 8. 全局耦合求解不可破坏的物理约束

### 8.1 禁止逐接口求解

同一步所有活动接口必须组成统一全局接触问题：

\[
0 \leq \lambda^{(k)}
\perp
g^{(k)}
=
q^{(k)} + W_{\mathrm{total}}^{(k)}\lambda^{(k)}
\geq 0
\]

不得：

- 为每个接口分别调用一次 LCP；
- 将各接口结果简单拼接；
- 先算某一结合面完整四阶段，再算另一结合面；
- 删除活动接口之间的非对角柔度块。

### 8.2 同一步多接口激活

一个新零件在同一步形成多个接口时，必须同时激活。

例如零件 D 同时形成：

```text
G_AD
G_DB
```

必须在同一 `topology_step` 中统一求解，不能拆成两个先后步骤来规避耦合。

### 8.3 公共向量布局

整条装配路线使用统一 `VectorLayout`。

活动索引由当前活动接口对应块的并集形成：

```python
active_indices = union(interface_layout_indices)
q_active = q_full[active_indices]
W_active = W_total_full[np.ix_(active_indices, active_indices)]
```

求解后保存：

```text
active_interface_ids
inactive_interface_ids
active_index_mask
lambda_active
gap_active
lambda_full
gap_full
```

失活接口不得仅凭 `lambda=0`、`gap=0` 解释，应有独立活动掩码。

---

## 9. 状态传递约束

每一步状态至少应包含：

```text
sample_id
topology_id
topology_step_id
parent_topology_step_id
parent_state_id
assembly_cycle_id
operation_type
current_subassembly_id
active_part_ids
active_interface_ids
active_joint_ids
active_boundary_ids
active_load_ids
vector_layout_id
operator_set_id
lambda
gap
pressure
local_compression
contact_mode
contact_structural_response
connection_lock_history_ids
solve_status
operator_source
fallback_flag
quality_flag
```

父状态链必须：

- 可解析；
- 无环；
- 不引用未来步骤；
- 可完整回放；
- 不被后续步骤覆盖。

纯 INIT、MEASURE 或其他不需要力学求解的步骤允许：

```text
solve_status = NOT_REQUIRED
```

但仍必须生成状态快照。

---

## 10. 新零件 SMS 与已有子装配体状态的边界

本轮虽然不实现阶段实测后验更新，但必须遵守论文冻结的状态语义：

- 新加入零件读取其装配前 SMS；
- 已经加入并形成子装配体的零件继续继承当前装配状态；
- 既有子装配体不能在每一轮装配开始时重新初始化为自由态 SMS；
- 当前步骤的初始几何应由父状态与新零件 SMS 共同构成；
- 禁止重复加入上一阶段已经计入的制造形貌或弹性变形。

---

## 11. JOIN 与 RELEASE 约束

### 11.1 JOIN

JOIN 步骤应：

- 激活指定连接；
- 保存 `joint_ids`；
- 保存 `locked_reference`；
- 保存 `preload_actual` 或其输入来源；
- 保存 `joint_stiffness`；
- 保存锁定时接触模式；
- 生成唯一 `ConnectionLockHistory`；
- 将锁定历史传给后续步骤。

### 11.2 RELEASE

RELEASE 步骤应：

- 撤除指定边界；
- 撤除指定载荷；
- 按 `retention_rule` 保留连接；
- 读取此前 JOIN 的锁定历史；
- 对当前全部活动接口重新全局求解；
- 保存 `ReleaseHistoryRecord`；
- 不得将子装配体重新初始化为自由零件。

---

## 12. 质量门冻结

`topology_step` 专项质量门至少包括：

1. `topology_step_id` 唯一；
2. 同一路线 `step_order` 不重复；
3. 父步骤可解析；
4. 父步骤链无环；
5. 首步骤不引用未来状态；
6. 新加入零件存在于 Part 表；
7. 活动接口存在于 Interface 表；
8. 活动接口两端零件已在当前子装配体中，或在本步同时加入；
9. 同一步允许多个接口激活；
10. 平行接口不得丢失；
11. 边界、载荷和连接引用可解析；
12. 结果子装配体可构造；
13. JOIN/RELEASE 保留规则可解析；
14. `solve_required=true` 时算子可解析；
15. 矩阵 key、布局和维度一致；
16. 活动子矩阵保留全部交叉块；
17. LCP 满足非负性、平衡和互补残差；
18. 状态父链完整；
19. 合成数据保持 `engineering_claim_allowed=false`。

严重错误必须阻断正式求解。

---

## 13. 旧数据包兼容约束

未提供真实工艺路线的旧数据包应自动转换为：

```text
LEGACY_TS_LOCATE
LEGACY_TS_CLAMP
LEGACY_TS_JOIN
LEGACY_TS_RELEASE
```

旧入口可以保留，但内部应统一调用新执行器。

兼容转换后必须保证：

- 旧数据包仍能运行；
- 原数值结果保持一致；
- 原测试不降低阈值；
- 原页面和报告不无故删除；
- 兼容回退被明确标记；
- 不把 legacy fallback 描述为真实多轮工艺执行。

---

## 14. 合成多步骤基准包约束

应新增一个独立合成基准包，例如：

```text
02_TOPOLOGY_STEP_MIN_CASE
```

建议路线：

```text
TS000 INIT：初始零件 A

TS101 LOCATE：加入 B，激活 G_AB
TS102 CLAMP：活动接口 G_AB
TS103 JOIN：锁定 AB 连接
TS104 RELEASE：撤除定位/夹持，保留 AB

TS201 LOCATE：加入 C，激活 G_BC，保留 G_AB
TS202 CLAMP：活动接口 G_AB、G_BC
TS203 JOIN：锁定 BC
TS204 RELEASE：保留 AB、BC

TS301 LOCATE：加入 D，同时激活 G_AD、G_DB
TS302 CLAMP：四接口统一活动
TS303 JOIN：激活 AD、DB 连接
TS304 RELEASE：撤除夹具，保留全部连接
```

基准包必须：

- 使用 4 零件、4 接口、公共 12 维布局；
- 每个求解步骤提供可追溯预计算算子；
- 证明同一步多接口只触发一次统一 LCP；
- 提供 oracle 或可复核解；
- 提供完整父状态链和连接历史；
- 明确为合成数值一致性数据；
- 不用于工程精度声明。

---

## 15. 页面与报告边界

本轮不增加新的实测融合或滚动预测页面。

在现有“装配拓扑、阶段路径与状态传递”页面中增加：

- 工艺路线表；
- `topology_step` 时间轴；
- 当前步骤；
- 输入和结果子装配体；
- 新加入零件；
- 活动接口；
- 边界和载荷变化；
- 连接变化；
- 算子来源；
- 父状态；
- LCP 质量门；
- 锁定与释放历史；
- 点击步骤查看详细状态。

运行报告建议增加：

```text
topology_step_execution.csv
topology_step_validation.csv
active_subassembly_history.csv
topology_step_state_lineage.csv
topology_step_operator_usage.csv
topology_step_contact_summary.csv
connection_lock_history.csv
release_history.csv
```

---

## 16. 自动测试最低要求

新增测试至少覆盖：

- 步骤唯一性、排序和父链；
- 无效父步骤阻断；
- 接口两端零件存在性；
- 同一步多接口激活；
- 同一步只调用一次全局 LCP；
- 活动接口交叉块保留；
- 平行接口不丢失；
- 动态子装配成员正确；
- 失活接口不进入 LCP；
- 活动掩码正确；
- JOIN 锁定历史；
- RELEASE 保留连接；
- 多轮相同 stage 类型重复执行；
- legacy 自动转换；
- legacy 数值等价；
- 算子缺失阻断；
- fallback 明确标记；
- 合成真实性标签；
- 报告文件完整；
- 页面可选择步骤；
- 不硬编码零件、接口和步骤数量；
- 新旧数据包 CLI 均通过。

---

## 17. 本轮验收条件

本轮只有同时满足以下条件，才可认定 `topology_step` 执行器完成：

1. 软件实际按路线表逐步执行，而不是只显示时间轴；
2. 同一步所有活动接口统一调用一次全局 LCP；
3. 多轮 LOCATE/CLAMP/JOIN/RELEASE 可重复出现；
4. 子装配体成员随步骤真实变化；
5. 新零件 SMS 与已有子装配体状态语义不混淆；
6. JOIN 锁定历史可跨后续步骤传递；
7. RELEASE 保留连接并重新求解；
8. 每一步算子来源可追溯；
9. 每一步状态和父链可回放；
10. 旧数据包结果保持一致；
11. 新合成多步骤基准包通过；
12. Python 3.11 全量回归通过；
13. Streamlit HTTP 健康检查通过；
14. 文档准确区分已实现、简化实现和尚未实现。

---

## 18. 后续阶段接口预留

本轮可为以下后续功能预留字段或接口，但不得实现空壳结果：

```text
measurement_checkpoint_id
assimilation_status
prediction_start_step_id
future_sms_source
rolling_scenario_id
```

下一轮软件任务才实现：

> 阶段实测后验更新。

再下一轮才实现：

> 后续虚拟 SMS 统计滚动预测。

---

## 19. 资料解释优先级

Codex 执行时应采用以下优先级：

1. 当前任务提示词；
2. 本实现约束文件；
3. 当前仓库代码、数据和测试；
4. 多零件串并联同步修订稿；
5. 正式数据结构说明书；
6. 技术说明书；
7. 理论说明书；
8. 其他说明资料。

如不同资料存在冲突，应：

- 不自行编造解释；
- 优先保持现有稳定功能和数据兼容；
- 在最终报告中列出冲突；
- 采用最小、明确、可测试的实现；
- 等待人工确认后再扩大范围。

---

## 20. 软件真实性声明

本轮完成后，软件可表述为：

> 已实现由装配工艺步骤表驱动的确定性多轮装配执行，并在每个步骤中对当前全部活动接口进行统一耦合接触求解，支持子装配状态继承、连接锁定与释放历史追溯。

仍不得表述为：

> 已实现阶段实测驱动的在线数字孪生滚动预测，或已获得真实飞机装配精度验证结果。

