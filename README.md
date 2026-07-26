# SMS 装配接触快速预测平台（topology_step 多轮装配版）

本项目是 Streamlit 原型软件。在原有 E1、V2.5 与多零件串并联能力之上，当前版本增加由 `I0/assembly_topology.csv` 驱动的确定性 `topology_step` 多轮执行器。每一步动态更新子装配体、活动接口、边界、载荷和连接，并对该步全部活动接口只求解一次保留交叉块的全局 LCP。四零件案例只是最小集成基准，不是理论或软件规模上限。

## 运行

```bash
# 在当前仓库根目录（包含 app.py）执行，不需要额外嵌套目录
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

推荐 Python 3.11；本仓库当前验收使用 `D:\anaconda\envs\thesis\python.exe`。当前依赖包含 Streamlit、Pandas、NumPy、Altair、NetworkX 与 Plotly。

## 数据包

软件启动后在侧边栏选择数据包目录。当前内置：

- `E1_min_closed_loop`：旧版 E1 闭环演示包。
- `E1_manual_input_9pt`：旧版 E1 手动录入包，保留 `manual_input_table.csv` 读取和重建矩阵功能。
- `01_DEFAULT_MIN_CASE`：V2.5 默认最小包，自动识别为 `V25_DEFAULT_MIN_CASE`。
- `01_DEFAULT_MIN_CASE_4_PART`：4零件/4接口/12接触点串并联集成基准，自动识别为 `V25_MULTI_PART`。
- `02_TOPOLOGY_STEP_MIN_CASE`：13 个 topology_step、3 个装配循环、4零件/4接口/12接触点的确定性多轮合成基准；`engineering_claim_allowed=false`。
- `E1_min_closed_loop_V25_DEFAULT_CASE`：由旧 E1 闭环包重构成的 V2.5 测试包。
- `E1_manual_input_9pt_V25_DEFAULT_CASE`：由旧 E1 9 点手动包重构成的 V2.5 测试包。

## 关键功能

- `core/schema_adapter.py`：V2.5 数据结构到当前求解器最小输入的适配层。
- `core/data_loader.py`：自动识别 `E1_LEGACY` / `V25_DEFAULT_MIN_CASE` / `V25_MULTI_PART`。
- `core/multi_part.py`：拓扑、向量分块、跨接口耦合、逐接口状态和贡献账本检查。
- `core/package_validator.py`：通用多零件主外键、布局、矩阵/LCP、状态父链、账本和真实性三级校验；阻断性 FAIL 会在正式求解前停止运行。
- `core/stage_state.py`：运行时 `StageState` 与 `LOCATE -> CLAMP -> JOIN -> RELEASE` 父状态链；明确区分包内预计算输入和运行时重算量。
- `core/topology_step.py`：工艺路线 schema adapter、23 项专项质量门、动态活动集、逐步状态继承、统一 LCP、JOIN 锁定与 RELEASE 保留/移除的执行主链。
- `core/reporting.py`：统一生成拓扑、路径、阶段递推、耦合诊断、状态链、校验和真实性报告。
- 多零件包按 `matrices/vector_layout.csv` 解释全局向量，完整求解含非零交叉块的全局 `W_struct`，不会把接口分别求解后拼接。
- 数据总览页：显示 CSV/JSON/NPZ 读取状态、行数、字段和矩阵 key。
- 质量检查：必要目录/文件、CSV 表头、NPZ key、g0/q/W/Cn/QA 维度检查。
- 物理一致性检查：显示互补残差、最小间隙、最小接触力、主动接触点数量和 KCP 异常提示。
- 追溯展示：读取并展示 `ContactComputationTrace`、`LCPSolution`、`KCPPredictionResult`，同时保留运行时动态追溯 JSON。
- “装配拓扑、阶段路径与状态传递”页面：显示 topology_step 路线表与时间轴，可按唯一步骤选择重复出现的 LOCATE/CLAMP/JOIN/RELEASE，查看父状态、算子来源、活动接口及 JOIN/RELEASE 历史。
- “接口耦合诊断与对照试算”页面：显示 `W_struct` 热力图、VectorLayout 分块指标、接口耦合网络，并提供跨接口块置零的非正式诊断试算。

## 四零件包与新页面使用

1. 启动后在侧边栏“标准输入包目录”选择 `01_DEFAULT_MIN_CASE_4_PART`；软件应显示类型 `V25_MULTI_PART`。
2. 进入“⑬ 装配拓扑、阶段路径与状态传递”，选择任意数据定义阶段；图中白色接口标记可点击，表格显示接触点数、活动点、总接触力、最大压力、最小间隙、父接口状态和锁定历史。
3. 选择 KCP 后，相关零件、接口和阶段来源会高亮；下方可查看串联、并联、闭环路径及运行时父状态链。
4. 进入“⑭ 接口耦合诊断与对照试算”切换阶段，查看完整 `W_struct` 热力图和每个接口块的范数、相对强度、最大绝对值与零块标记。
5. 正式结果始终使用完整耦合矩阵。点击“运行诊断：将跨接口 W_struct 块置零”只生成 `DIAGNOSTIC_NOT_FORMAL_ENGINEERING_RESULT`，超过阈值时显示红色警告。
6. 在“⑫ 验证、报告与追溯”下载完整运行报告 ZIP；其中包含 `topology_summary.csv`、`assembly_path_summary.csv`、`stage_transition_runtime.csv`、`interface_stage_summary.csv`、`cross_interface_coupling_blocks.csv`、`coupling_ablation_comparison.csv`、`state_lineage.csv`、`validation_summary.csv` 和 `data_truthfulness_statement.txt`。

## topology_step 多轮路线使用

1. 选择 `02_TOPOLOGY_STEP_MIN_CASE`，进入“⑬ 装配拓扑、阶段路径与状态传递”。路线从 TS000 依次执行到 TS304；同一种 stage 类型可在不同 `assembly_cycle_id` 中重复出现。
2. TS301 在同一步加入 P_D、同时激活 G_AD 与 G_DB；连同既有 G_AB、G_BC，以 12 维 `W_active = W_total[np.ix_(active_indices, active_indices)]` 只调用一次全局 LCP。
3. 每个求解步骤明确记录 `operator_source=PRECOMPUTED_TOPOLOGY_STEP_OPERATOR`。没有真实 topology_step 表的旧包经适配层生成四个 `LEGACY_TS_*`，并记录 `fallback_flag=true` 与 `fallback_reason=LEGACY_STAGE_COMPATIBILITY`。
4. 报告另含 `topology_step_execution.csv`、`topology_step_validation.csv`、`active_subassembly_history.csv`、`topology_step_state_lineage.csv`、`topology_step_operator_usage.csv`、`topology_step_contact_summary.csv`、`connection_lock_history.csv` 和 `release_history.csv`。
5. 无父状态的 INIT 使用 `NOT_REQUIRED + INITIALIZE_EMPTY`；有父状态的 MEASURE/INSPECT 等非求解事件使用 `NOT_REQUIRED + INHERIT_PARENT_UNCHANGED`，完整继承 lambda、gap、pressure、local_compression、接触模式、结构响应、活动 mask 与连接状态，不调用 LCP。
6. RELEASE 先应用显式 `deactivated_joint_ids`，再对其余活动 joint 执行 `RETAIN_THROUGH_RELEASE` 或 `REMOVE_AT_RELEASE`；未知规则为阻断错误，ReleaseHistory 同时记录 retained、removed 与释放后活动集合。
7. `PRECOMPUTED_TOPOLOGY_STEP_OPERATOR` 模式不会按运行时倍率重构 q/W/Cn，因此相关倍率和高级重构控件被禁用，基于这些倍率的 Monte Carlo/敏感性入口隐藏；执行报告记录 `parameter_effective=false`。Legacy 路径原有有效倍率行为保持不变。

## 命令行检查

```bash
python scripts/cli_check.py data/01_DEFAULT_MIN_CASE
python scripts/cli_check.py data/01_DEFAULT_MIN_CASE_4_PART
python scripts/cli_check.py data/02_TOPOLOGY_STEP_MIN_CASE
python scripts/cli_check.py data/E1_manual_input_9pt
python scripts/cli_check.py data/E1_min_closed_loop_V25_DEFAULT_CASE
python scripts/cli_check.py data/E1_manual_input_9pt_V25_DEFAULT_CASE
```

CLI 机器可读尾部固定输出 `FINAL_STATUS`、`BLOCKING_FAIL_COUNT` 和 `PHYSICAL_FAIL_COUNT`。退出码语义为：0=PASS/允许 WARN，1=阻断校验失败，2=运行时/求解/总体物理失败，3=未预期异常。数据包自带校验器失败也计入退出码 1。

`01_DEFAULT_MIN_CASE` 是占位连通性测试包，不用于论文数值结论。物理一致性检查是一级质量门，用于判断本次 LCP 解是否满足基本可行性；它不替代真实 FE 验证、样件验证或完整 D_valid 适用域判定。

`01_DEFAULT_MIN_CASE_4_PART` 同样是合成数值一致性数据，只证明数据结构、统一耦合求解、状态对象读取和贡献账本接口可联通，不能作为真实结构的精度或工程结论。界面持续显示：“仅用于数值一致性与软件联调，不代表真实工程预测结果。”

详细升级范围与剩余边界见 `MULTI_PART_UPGRADE.md` 和 `TOPOLOGY_STEP_EXECUTOR.md`。

## 自动测试与审计

```bash
python scripts/run_automated_tests.py
```

测试结果写入 `test_reports/latest/`。当前功能边界、真实性判定和 V6-MVP 完成条件见 `AUDIT_V6_MVP.md`。

当前提交前收尾验收统计为：134 tests、128 PASS、0 FAIL、6 SKIP。其中 `test_topology_step_executor.py` 37 项、`test_topology_step_closeout.py` 35 项，共 72 项 topology_step 专项测试；原 99 项测试全部保留。`02_TOPOLOGY_STEP_MIN_CASE` 的独立自校验为 22/22 PASS，MatrixManifest 与 NPZ 均为 156 个 key。

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

## 阶段实测后验更新

- 新增 measurement checkpoint 驱动的低维线性高斯更新、Joseph 协方差、显式 `G_q` 状态回代和一次统一全局 LCP 重求。
- 同时保留 PREDICTED/POSTERIOR 状态；成功后验传递到后续 `topology_step`，失败则生成回滚记录并保留预测状态。
- 新增确定性 `03_STAGE_MEASUREMENT_UPDATE_MIN_CASE`、独立后验/LCP oracle、本地验证器、CLI 计数、完整报告产物与 Streamlit 第 15 页。
- 本功能冻结参数和 SMS；合成测量只用于数值一致性验证。详见 `STAGE_MEASUREMENT_POSTERIOR_UPDATE.md`。
