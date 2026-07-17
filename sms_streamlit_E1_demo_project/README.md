# SMS 装配接触快速预测平台（多零件串并联集成版）

本项目是 Streamlit 原型软件。本次升级在原有 E1 与 V2.5 单案例兼容基础上，增加可配置多零件、多接口统一向量布局、跨接口耦合 LCP、逐接口状态汇总和多路径 KCP 投影。四零件包只是最小集成基准，不是理论或软件规模上限。

## 运行

```bash
cd sms_streamlit_E1_demo_project
pip install -r requirements.txt
streamlit run app.py
```

## 数据包

软件启动后在侧边栏选择数据包目录。当前内置：

- `E1_min_closed_loop`：旧版 E1 闭环演示包。
- `E1_manual_input_9pt`：旧版 E1 手动录入包，保留 `manual_input_table.csv` 读取和重建矩阵功能。
- `01_DEFAULT_MIN_CASE`：V2.5 默认最小包，自动识别为 `V25_DEFAULT_MIN_CASE`。
- `01_DEFAULT_MIN_CASE_4_PART`：4零件/4接口/12接触点串并联集成基准，自动识别为 `V25_MULTI_PART`。
- `E1_min_closed_loop_V25_DEFAULT_CASE`：由旧 E1 闭环包重构成的 V2.5 测试包。
- `E1_manual_input_9pt_V25_DEFAULT_CASE`：由旧 E1 9 点手动包重构成的 V2.5 测试包。

## 关键功能

- `core/schema_adapter.py`：V2.5 数据结构到当前求解器最小输入的适配层。
- `core/data_loader.py`：自动识别 `E1_LEGACY` / `V25_DEFAULT_MIN_CASE`。
- `core/multi_part.py`：拓扑、向量分块、跨接口耦合、逐接口状态和贡献账本检查。
- 多零件包按 `matrices/vector_layout.csv` 解释全局向量，完整求解含非零交叉块的全局 `W_struct`，不会把接口分别求解后拼接。
- 数据总览页：显示 CSV/JSON/NPZ 读取状态、行数、字段和矩阵 key。
- 质量检查：必要目录/文件、CSV 表头、NPZ key、g0/q/W/Cn/QA 维度检查。
- 物理一致性检查：显示互补残差、最小间隙、最小接触力、主动接触点数量和 KCP 异常提示。
- 追溯展示：读取并展示 `ContactComputationTrace`、`LCPSolution`、`KCPPredictionResult`，同时保留运行时动态追溯 JSON。

## 命令行检查

```bash
python scripts/cli_check.py data/01_DEFAULT_MIN_CASE
python scripts/cli_check.py data/01_DEFAULT_MIN_CASE_4_PART
python scripts/cli_check.py data/E1_manual_input_9pt
python scripts/cli_check.py data/E1_min_closed_loop_V25_DEFAULT_CASE
python scripts/cli_check.py data/E1_manual_input_9pt_V25_DEFAULT_CASE
```

`01_DEFAULT_MIN_CASE` 是占位连通性测试包，不用于论文数值结论。物理一致性检查是一级质量门，用于判断本次 LCP 解是否满足基本可行性；它不替代真实 FE 验证、样件验证或完整 D_valid 适用域判定。

`01_DEFAULT_MIN_CASE_4_PART` 同样是合成数值一致性数据，只证明数据结构、统一耦合求解、状态对象读取和贡献账本接口可联通，不能作为真实结构的精度或工程结论。

详细升级范围与剩余边界见 `MULTI_PART_UPGRADE.md`。

## 自动测试与审计

```bash
python scripts/run_automated_tests.py
```

测试结果写入 `test_reports/latest/`。当前功能边界、真实性判定和 V6-MVP 完成条件见 `AUDIT_V6_MVP.md`。

## v5.4.1兼容修复

- V2.5启用SMS重建时，适配`I_meas/measurement_record.csv`中的SMS识别/校准记录。
- 原始点不足以重建结合面两侧时，安全使用`GapField/SMSField`中的冻结SMS分量，不再报`KeyError: part_id`，也不把缺失侧静默置零。
- V2.5缺少可用`I_substitution`参数而选择`replace`时，保留包内`Cn_local`并输出WARN，不再用全零Cn替换。
- 新增3个V2.5数据包与全部高级开关的组合回归测试。

## v5.4.2界面优化

- 将物理一致性总状态拆分为LCP物理可行性、接触形态提示、KCP产品容差和KCP验证基准。
- 无接触/全接触状态保留WARN提示，但不再等同于LCP物理FAIL。
- 高级模块改变运行配置而数据包未提供同配置验证批次时，KCP验证标记为`REFERENCE_ONLY`。
- 对V2.5单点最小占位包增加用途说明，并将异常项、KCP检查、阈值公式分为独立页签。
