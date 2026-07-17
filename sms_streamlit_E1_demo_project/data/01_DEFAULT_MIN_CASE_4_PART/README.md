# 01_DEFAULT_MIN_CASE - 多零件串并联最小案例

本案例依据《数学模型数据结构说明书 V2.5》及其多零件串并联同步修订稿构建。

## 拓扑

- 串联路径：`P_PANEL_A - G_PANEL_RIB - P_RIB_B - G_RIB_SPAR - P_SPAR_C`。
- 并联路径：A与B之间同时存在直接路径 `G_PANEL_RIB`，以及桥接路径 `G_PANEL_BRACKET + G_RIB_BRACKET`。
- 4个接口各3个候选接触点，统一排列为12维接触向量。
- 四阶段为 LOCATE、CLAMP、JOIN、RELEASE；所有阶段均统一求解4个接口。

## 关键修正

增加 Subassembly、SubassemblyMembership、JointDefinition、StageDefinition、PartStageState、InterfaceStageState 和 VectorLayout；补齐三条阶段转移、连接锁定与释放历史；W_struct使用带非零交叉块的12x12对称正定矩阵。

## 使用边界

这是合成数值一致性案例，只能用于软件联调和自动测试。所有矩阵、载荷、SMS和KCP结果均不得作为真实工程或论文验证结论。

运行 `python validation/validate_package.py .` 可检查拓扑、引用、矩阵、LCP互补性、跨接口耦合、状态传递和贡献账本。
