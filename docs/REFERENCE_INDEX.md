# SMS 软件升级参考资料索引

> 当前任务：实现真实工艺路线表驱动的 `topology_step` 确定性多轮装配执行器  
> 当前范围：不实现阶段实测后验更新、后续虚拟 SMS 滚动预测、Monte Carlo 和真实工程验证

## 1. 使用方式

开始修改前先阅读本索引，再按任务范围阅读对应资料。

- `reference/extracted_text/`：用于全文搜索和关键词定位；
- `reference/*.pdf`：用于最终核对公式、表格、适用边界和原文语义；
- `implementation_contracts/`：用于保存从论文冻结主线提取的软件实现合同；
- 当前仓库代码、数据包和自动测试：用于核对实际实现状态及兼容性。

机械提取文本不能替代 PDF；README、升级总结和开发记录不能替代正式说明书与实现合同。

## 2. 本轮直接实现合同

### C00. `implementation_contracts/论文结构冻结_topology_step实现约束.md`

该文件是本轮 `topology_step` 执行器的直接软件合同，主要规定：

- `topology`、`topology_step`、`assembly_cycle` 与 `stage` 的区别；
- 真实工艺路线必须按表逐步执行；
- 同一步全部活动接口必须统一组装并只调用一次全局 LCP；
- 同一步新零件形成多个接口时必须同时激活；
- 新加入零件读取装配前 SMS，已有子装配体继承父状态；
- JOIN 生成锁定历史，RELEASE 保留连接并重新平衡；
- 当前可使用预计算 `q`、`W_struct`、`C_n` 和连接算子；
- 旧四阶段数据包通过 legacy adapter 保持兼容；
- 本轮不得实现或宣称阶段实测后验更新与虚拟 SMS 滚动预测。

## 3. 基础规范

### R00. `reference/00_data_structure_v25_full.pdf`

原名：《数据结构说明书_V2.5_V2.2风格正式版》

完整数据结构基础规范，用于确定：

- 通用对象、字段、主键、外键和量纲；
- `StageInput`、`BoundaryItem`、`LoadItem`；
- `StageStateSnapshot`、`StageTransitionRecord`、`StageIncrement`；
- `ConnectionLockHistory`、`ReleaseHistoryRecord`；
- `ContactComputationTrace`；
- 缓存失效、质量门、版本治理和防重复计算。

## 4. 多零件增量覆盖

### R01. `reference/01_data_structure_v25_multipart.pdf`

原名：《数学模型数据结构说明书_V2.5_多零件串并联同步修订稿》

本文件不是完整说明书。它替换和补充 R00 中与多零件装配有关的条款，重点用于核对：

- Part、Interface、Joint、Subassembly 和 TopologyStep；
- 独立 `topology_step_id`；
- 同一步多个接口同时激活；
- 多 ContactDomain 公共 VectorLayout；
- 全局接触向量和矩阵分块；
- `W_struct` 接口交叉柔度块；
- 多接口统一 LCP；
- 聚合、逐零件和逐接口状态父链；
- JOIN 锁定、RELEASE 继承；
- 多路径贡献唯一键和合成数据真实性标签。

未被 R01 覆盖的测量、参数识别、验证和版本治理条款继续遵循 R00。

## 5. 辅助资料

### R02. `reference/02_model_theory_v12.pdf`

原名：《SMS叠层装配接触预测模型_理论说明书_V1.2_修订版》

用于核对：

- SMS 到初始间隙映射；
- `q`、`W_struct`、`C_n` 与 `W_total` 的物理意义；
- LCP/NCP、主动集和接触模式；
- 已有子装配体状态与新加入零件 SMS 的物理边界；
- 状态继承、增量和释放回弹。

### R03. `reference/03_technical_specification_v24.pdf`

原名：《技术说明书_V2.4_全量保留正式整合定稿版》

用于核对：

- 总体计算链和模块关系；
- 阶段边界、载荷和连接状态切换；
- 全局接触求解与模式暖启动；
- JOIN 锁定和 RELEASE 重新平衡；
- `RESOLVE_LCP`、`REBUILD_W`、`REBUILD_IRED` 等重算边界；
- 当前快速模型与高保真层的真实性边界。

### R04. `reference/04_input_data_requirements_v03.pdf`

原名：《数学模型输入数据获取需求说明书_沟通版_V0.3》

用于核对：

- 工艺路线、装配顺序和对象编号的输入需求；
- 定位、夹持、连接和释放数据；
- 结构算子、接触域、接口参数和追溯数据；
- 数据来源、单位、必需程度和真实性。

### R05. `reference/05_experiment_and_data_processing_v12.pdf`

原名：《SMS数学模型_实验操作与数据处理流程说明书_V1.2》

用于核对实验、数据处理、独立验证和结果追溯流程。

本轮只使用其中与阶段边界、状态继承、连接历史和质量门有关的内容；不据此实现阶段实测更新。

### R06. `reference/06_eight_standard_inputs_v12.pdf`

原名：《八类标准输入数据获取与计算方法_V1.2_修订版》

用于核对八类输入包 `I0`、`I_Gamma`、`I_stage`、`I_key`、`I_meas`、`I_red`、`I_pred` 和 `I_stat` 的来源与转换关系，尤其是：

- 工艺路线和装配顺序属于工程交付数据；
- 阶段边界、载荷和连接状态进入 `I_stage`；
- `W_struct` 等算子属于仿真/程序计算数据；
- SMS、阶段状态和验证数据不得混用。

若仓库中尚无该文件，应将当前有效 PDF 加入 `reference/`，并可同步生成 `reference/extracted_text/06_eight_standard_inputs_v12.txt`。

## 6. 当前任务的解释优先级

出现不一致时，按以下优先级处理：

1. 当前 Codex 任务提示词；
2. C00 `论文结构冻结_topology_step实现约束.md`；
3. 当前仓库代码、数据包、自动测试和实际 Git 状态；
4. R01 多零件串并联同步修订稿；
5. R00 完整数据结构说明书；
6. R03 技术说明书；
7. R02 理论说明书；
8. R04 输入数据需求说明书；
9. R05 实验操作与数据处理流程说明书；
10. R06 八类标准输入数据获取与计算方法；
11. README、升级说明和测试摘要。

其中：

- C00 规定本轮实现范围；
- R01 在其生效范围内覆盖 R00；
- R00 继续规定通用对象和治理规则；
- 当前软件通过适配层维持新旧数据兼容。

## 7. 冲突处理规则

若资料之间存在冲突：

1. 不自行猜测或拼接相互冲突的定义；
2. 优先保持当前稳定功能和旧数据包兼容；
3. 优先使用 schema adapter，不直接破坏旧字段；
4. 无法确定时使用显式 `WARN`、`FUTURE` 或 `NOT_IMPLEMENTED`；
5. 在最终报告中列出冲突位置、采用解释及影响；
6. 未经人工确认，不得扩大到阶段实测更新、滚动预测、完整摩擦 NCP 或全阶 FE 自动凝聚。

## 8. 当前实现状态分级

### 8.1 已实现基线

- 多零件串联、并联、闭环和共享零件拓扑；
- 平行接口保留；
- 同一阶段活动接口统一全局 LCP；
- 跨接口 `W_struct` 非对角块；
- 聚合、逐零件和逐接口状态对象；
- 状态父链；
- 简化 JOIN/RELEASE 历史；
- 数据包质量门、贡献账本和运行报告；
- 旧 E1、V2.5 和四零件数据包兼容。

### 8.2 本轮目标

- 真实工艺路线表读取；
- `TopologyStepSpec` 与 `TopologyStepRunner`；
- 多轮装配步骤执行；
- 动态子装配体成员；
- 动态活动接口、边界、载荷和连接；
- 每一步只调用一次全局耦合 LCP；
- JOIN/RELEASE 跨步骤历史；
- legacy 四阶段自动转换；
- 多步骤合成基准包；
- 步骤级追溯、质量门和报告。

### 8.3 允许的简化

- 每步 `q`、`U_FREE`、`W_struct` 和 `C_n` 可由数据包预计算；
- 不在线重建全阶结构算子；
- 不自动调用 Abaqus；
- 合成案例只证明流程和数值一致性；
- 必须记录 `operator_source`、`fallback_flag` 和真实性标签。

### 8.4 尚未实现

- 阶段实测后验更新；
- 后续虚拟 SMS 滚动预测；
- 未来场景 Monte Carlo；
- 全阶 FE 自动 Schur 凝聚；
- 完整法向—切向摩擦 NCP；
- 坐标链自动执行；
- JSS/J-T 自动构建；
- 独立真实 KCP 验证；
- 接触域收敛、几何/材料非线性和工艺优化。

## 9. 数据真实性规则

四零件和新增多步骤基准包只能作为最小集成或合成数值一致性案例，不能成为软件固定规模，也不能用于真实工程精度声明。

所有合成数据应声明：

```text
engineering_claim_allowed=false
```

不得混淆：

- 合成算子与真实 FE 凝聚算子；
- 预计算步骤算子与运行时自动重建算子；
- legacy 四阶段兼容路线与真实多轮工艺路线；
- 接触坐标响应与完整全阶零件位移；
- 方法接口与已完成工程验证。

## 10. Codex 执行要求

开始前执行：

```bash
git status
git branch --show-current
git diff --stat
python scripts/run_automated_tests.py
```

任务中不得：

- 修改或删除原始参考资料；
- 降低测试阈值或修改数据制造通过；
- 将阶段实测更新或滚动预测做成空壳“已实现”功能；
- 执行 `git add`、`git commit`、`git push`；
- 未经人工确认生成最终发布 ZIP。

完成后必须报告：修改文件、路线适配方式、每步算子来源、每步 LCP 调用次数、legacy 等价性、新旧测试、Python 3.11 和 Streamlit HTTP 结果、Git 状态、未解决问题和真实性边界。

## 11. 维护规则

1. 原始说明书放入 `docs/reference/`；
2. 提取文本放入 `docs/reference/extracted_text/`；
3. 软件实现合同放入 `docs/implementation_contracts/`；
4. 新增或替换资料时同步更新本索引；
5. 保留稳定编号 R00、R01……；
6. 记录原始中文文件名、标准仓库文件名和适用职责；
7. 不因文件最近修改时间而覆盖内容更新、适用范围更明确的资料。

## 12. 本轮完成后的允许表述

可以表述为：

> 软件已实现由装配工艺步骤表驱动的确定性多轮装配执行，并在每个步骤中对当前全部活动接口进行统一耦合接触求解，支持子装配体成员更新、状态父链、连接锁定与释放历史追溯。

仍不得表述为：

> 软件已实现阶段实测驱动的在线后验更新、未来虚拟 SMS 滚动预测或真实飞机装配精度验证。
