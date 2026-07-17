# 多零件串并联升级说明

## 本次已实现

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

## 当前实现边界

- 四零件包中的阶段状态继承已经通过包内 `q/U_FREE` 和状态快照进入求解与追溯；软件尚未从全阶结构、连接锁定历史自动重新生成下一阶段凝聚算子。
- 软件读取并检查 `PartStageState`、`InterfaceStageState`、`StageTransitionRecord`，但尚未将所有位姿/结构位移字段作为独立动态状态递推变量重新积分。
- `W_struct` 仍由数据包提供；尚未从 CAD/FE 刚度矩阵自动 Schur 凝聚生成。
- 切向摩擦、扩展约束、非线性和回弹仍沿用原型中的简化或等效实现；需要真实局部 FE、样件或标定数据后才能形成工程级闭环。
- 四零件数据为合成数值一致性基准，只可用于软件联调和算法自洽验证。

## 规模扩展方式

软件没有将4个零件、4个接口或每接口3个点写死。扩展案例时应同步提供：

- 任意数量的 Part、Interface、ContactDomain 与 ContactPoint；
- 覆盖全部接触点且连续无重叠的 VectorLayout；
- 与全局接触维数一致的各阶段 `Q`、`W_STRUCT`、`W_TOTAL` 和 `CN`；
- 对应的 StageDefinition、状态快照、连接历史和 KCP 投影矩阵/贡献账本。
