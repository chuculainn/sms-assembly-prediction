# 多零件串并联升级说明

## 本次已实现

### topology_step 确定性多轮升级

- `run_all_stages(...)` 保持为兼容入口，内部统一调用 `run_topology_steps(...)`。真实路线以 `topology_step_id` 为结果键，不再用重复的 stage 类型充当步骤身份。
- `I0/assembly_topology.csv` 经 schema adapter 解析为 `TopologyStepSpec`；分号多值字段在执行前转换为 ID 元组。
- 每一步从不可变父状态快照出发，动态增删零件、接口、边界、载荷和连接；已有零件继承装配状态，仅新加入零件读取其装配前 SMS，初始化次数保持 1。
- 活动索引由公共 VectorLayout 中全部活动接口块的并集形成，使用 `np.ix_` 从完整 `W_total` 提取子矩阵，保留所有活动接口交叉块；每个 `solve_required=true` 步骤恰好调用一次全局 LCP。
- JOIN 每步生成一条唯一 `ConnectionLockHistory`，即使同时锁定多个 joint；RELEASE 读取并继承此前全部锁定历史，按 `retention_rule` 保留连接后重新统一求解。
- 无父状态 INIT 生成 `INITIALIZE_EMPTY`；有父状态的非求解事件生成 `INHERIT_PARENT_UNCHANGED`，保留父机械向量、接触模式、结构响应、活动 mask 和连接状态，不重新调用 LCP。fallback/物理一致性门把 `NOT_REQUIRED` 作为合法状态事件，而不是 SKIP 或 LCP 失败。
- RELEASE 的连接集合按“父集合 → 本步激活 → 显式停用 → 对其余 joint 应用 retention_rule”计算。显式停用优先；支持 `RETAIN_THROUGH_RELEASE` 与 `REMOVE_AT_RELEASE`，未知值阻断。
- 没有真实 topology_step 表的旧包显式转换为 `LEGACY_TS_LOCATE/CLAMP/JOIN/RELEASE`，数值结果与原 `run_stage` 路径等价，并标明兼容回退。
- 新增 `02_TOPOLOGY_STEP_MIN_CASE` 合成一致性包和独立主动集枚举 LCP oracle；没有修改原六个数据包。
- 新包生成器每次从模板完整重建校验器、字段字典、对象映射、oracle 与验证附件，不复制旧包统计。当前包内自校验 22/22 PASS，MatrixManifest/NPZ 均为 156 个 key。
- 预计算 topology_step 算子模式禁用不会重构 q/W/Cn 的倍率及高级重构控件，报告记录 `parameter_effective=false`；Legacy 模式的有效倍率保留。

1. 自动识别 `multi_part_matrices.npz`，数据包类型为 `V25_MULTI_PART`。
2. 从多个 GapField 按 `contact_domain_id + local_index` 组装全局接触向量。
3. 优先读取 `StageDefinition`，避免 AssemblyTopology 中同一阶段的多个拓扑步骤被误当成重复求解阶段。
4. 读取 `VectorLayout`，保留每个接口在全局向量中的分块边界。
5. 使用完整全局 `W_struct + Cn` 统一求解所有接口；跨接口非对角块不被丢弃。
6. 输出逐接触点、逐接口和全局阶段三级结果，并在运行追溯中记录各接口主动集。
7. 使用 `J_INTERFACE_ALL` 将实时局部压缩投影到多路径 KCP；单位倍率下与数据包合成 oracle 一致。
8. 检查装配图连通性、串联路径、闭环/并联路径、向量布局连续性、跨接口块、状态父链和贡献账本唯一性。
9. 报告 ZIP 新增 `interface_stage_summary.csv` 与 `cross_interface_coupling_blocks.csv`。
10. 保留旧 E1、V2.5 默认包和高级模块的兼容路径。
11. 通用校验器逐对象输出 PASS/WARN/FAIL、文件、字段、对象ID、修复建议和阻断标志；严重错误在正式求解前停止。
12. 新增 NetworkX/Plotly “装配拓扑、阶段路径与状态传递”独立页面，支持阶段边状态、接口点击详情和KCP路径高亮。
13. 新增 `W_struct` 热力图、接口块指标、耦合网络和跨接口块置零诊断；诊断结果永久标记为非正式工程结果。
14. `run_all_stages` 按数据定义顺序建立运行时 `StageState` 父链，JOIN锁定历史在RELEASE显式继承。
15. 报告ZIP新增拓扑、路径、运行时转移、接口汇总、交叉块、消融、父链、校验和真实性九类交付物。
16. `assembly_graph` 使用 `nx.MultiGraph`，以 `interface_id` 作为 edge key，完整保留同一零件对之间的多个平行接口；路径、闭环和Plotly显示均不会静默覆盖接口。
17. KCP贡献路径按 `contribution_record.csv.target_kcp_id` 精确过滤，兼容单个KCP ID和分号分隔的多个KCP ID。
18. `ContributionLedger` 按 prediction、sample 和 KCP 作用域分别执行唯一键检查、贡献向量重构和预测值对比，不跨样本累计。
19. `StageState` 将接触力引起的结构柔性响应输出为 `contact_structural_response`，其数值为 `W_struct @ lambda`。
20. `contact_structural_response` 只表示接触坐标下的结构柔性响应，不代表全阶阶段位移或完整零件自由度响应。

## 当前实现边界

- topology_step 路线是单条、确定性、线性有序路线；没有条件分支、返修、并行调度、实测触发切换或动态重规划。
- 每步的 `q/U_FREE/W_struct/Cn` 来自数据包预计算算子，软件没有在线调用全阶 FE，也没有从全阶 K 自动 Schur 凝聚。
- 本轮没有实现阶段实测后验更新、后续虚拟 SMS 滚动预测或未来场景 Monte Carlo；仓库既有 Monte Carlo 原型保持兼容，但不属于本执行器的新能力。
- 预计算 topology_step 模式不运行基于无效倍率的 Monte Carlo/单因素敏感性；本轮没有实现步骤算子在线重构。
- `02_TOPOLOGY_STEP_MIN_CASE` 是合成数值一致性案例，不是工程验证数据；不得据此宣称 KCP 预测精度或生产级数字孪生能力。

- 软件已建立运行时父状态对象、阶段增量、边界/载荷变化和连接锁定继承；但各阶段 `q/U_FREE/W_struct` 仍来自包内预计算输入，尚未从上一阶段全阶结构状态自动重新凝聚下一阶段算子。界面和导出均设置 `fallback_flag=true`。
- `PartStageState` 的包内零件位移作为可追溯预计算状态读取；运行时真正递推的是父状态引用、`contact_structural_response = W_struct @ lambda`、间隙增量、全局LCP解、接口状态与锁定历史，不宣称完成全阶阶段位移或零件自由度积分。
- `W_struct` 仍由数据包提供；尚未从 CAD/FE 刚度矩阵自动 Schur 凝聚生成。
- 切向摩擦、扩展约束、非线性和回弹仍沿用原型中的简化或等效实现；需要真实局部 FE、样件或标定数据后才能形成工程级闭环。
- 四零件数据为合成数值一致性基准，只可用于软件联调和算法自洽验证。

## 兼容解释记录

- 完整V2.5基础包的 `assembly_topology.csv` 使用 `topology_id + assembly_step` 复合唯一键；多零件修订包使用新增 `topology_step_id` 主键。校验器通过适配规则同时支持两者，不修改原始CSV。
- 原始 `contact_point.csv` 主键为 `point_id`，求解适配层内部规范化为 `candidate_id`。校验器检查原始 `point_id`，运行时继续使用兼容字段 `candidate_id`。
- 多零件修订稿的贡献来源唯一键按 `sample_id + source_class + source_id + origin_stage_id + increment_definition_id` 检查；基础V2.5说明书按“同一KCP路径”判重，因此旧转换包适配时额外包含 `target_kcp_id`，避免把同一来源对不同KCP的合法记录误判为重复。
- 任意交叉接口块为零时通用校验输出 WARN 而非自动 FAIL，因为某些数据驱动拓扑允许物理解耦；四零件基准仍由专门测试要求六对交叉块全部非零。

## 规模扩展方式

软件没有将4个零件、4个接口或每接口3个点写死。扩展案例时应同步提供：

- 任意数量的 Part、Interface、ContactDomain 与 ContactPoint；
- 覆盖全部接触点且连续无重叠的 VectorLayout；
- 与全局接触维数一致的各阶段 `Q`、`W_STRUCT`、`W_TOTAL` 和 `CN`；
- 对应的 StageDefinition、状态快照、连接历史和 KCP 投影矩阵/贡献账本。
