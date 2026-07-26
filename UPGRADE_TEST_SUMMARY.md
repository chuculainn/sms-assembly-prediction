# 多零件串并联升级测试摘要

测试日期：2026-07-19

## 修改前基线

- 使用仓库自带 `scripts/run_automated_tests.py` 运行。
- 测试总数 39：PASS 32、FAIL 0、SKIP 7。
- 系统默认 Python 当时缺少 pandas；修改前基线改用已安装项目依赖的工作区运行时执行。

## 修改后自动回归

- 最终测试总数 62：PASS 56、FAIL 0、SKIP 6。
- 相对于原 39 项基线共新增 23 项测试；其中多零件升级主体新增 18 项，本轮定向缺陷修复新增 5 项。
- 新增覆盖包括通用校验器、全部六类内置数据包兼容、任意 N/M 拓扑、VectorLayout、串并联/闭环识别、同一零件对平行接口保留、六对跨接口块、耦合消融、KCP贡献路径过滤、按预测/样本/KCP分组重构贡献账本、运行时阶段父链及首阶段边界/载荷、StageState响应语义、JOIN/RELEASE锁定继承、Pandas警告回归、真实性声明、报告ZIP、新页面与Streamlit测试。
- 原“阶段状态继承”SKIP 已转为真实通过，因此保留的工程化 SKIP 由 7 项降为 6 项；没有把失败测试改成 SKIP。

## Python 3.11 与 Streamlit HTTP 验收

- 使用官方 Windows CPython 3.11.9 按 `requirements.txt` 安装依赖后执行完整回归：62 tests、PASS 56、FAIL 0、SKIP 6。
- Python 3.11.9 下执行 `python -m compileall -q app.py core tests scripts`：通过。
- Python 3.11.9 下执行 `python -m streamlit run app.py --server.headless true`：服务成功监听 `localhost:8501`。
- 请求 `http://127.0.0.1:8501/_stcore/health`：HTTP 200，响应正文 `ok`；检查后已停止服务器。

## 数据包与求解检查

- 旧 E1、V2.5 标准数据包和 `01_DEFAULT_MIN_CASE_4_PART` 四零件包均保持可加载和可求解。
- 四零件包以一个 12 维公共 VectorLayout 统一求解，正式模式保留全部六对跨接口非对角块。
- 正式求解继续使用 `W_total = W_struct + Cn`，并满足全局 LCP 非负性和互补残差要求。
- 去除跨接口块仅作为诊断对照，并明确标记为非正式工程结果。

## 页面与报告检查

- 新增“装配拓扑、阶段路径与状态传递”和“接口耦合诊断与对照试算”两个独立页面。
- Streamlit AppTest 已选择四零件包并渲染全部页面，无未捕获异常。
- 报告 ZIP 已验证包含拓扑、路径、运行时阶段传递、接口阶段、耦合块、消融对照、状态父链、校验摘要及真实性声明九类新增文件。

## 仍保留的 6 项工程化验收边界

以下功能未伪装为已实现，仍作为明确的后续工程化工作保留：

1. 从全阶刚度矩阵自动 Schur 凝聚 `W_struct`；
2. 完整法向—切向耦合摩擦 NCP；
3. 坐标链和单位归一化的运行时执行；
4. 从原始模型自动构建 JSS/J-T 贡献映射；
5. 独立验证集的 MAE/RMSE/最大误差计算；
6. 接触域离散加密的收敛性验证。

当前阶段递推框架会继承运行时父状态和锁定历史；当包内数据不足以重建下一阶段时，明确使用包内预计算 `q`、`U_FREE` 或矩阵作为兼容回退，不宣称完成全阶重新凝聚。

## topology_step 执行器增量验收（2026-07-22）

- 修改前重新确认基线：62 tests、PASS 56、FAIL 0、SKIP 6，解释器为 `D:\anaconda\envs\thesis\python.exe`（Python 3.11）。
- 新增 `tests/test_topology_step_executor.py`，覆盖 36 项：步骤唯一性/排序/父链、主外键、TS301 单次 12 维全局 LCP、交叉块、活动 mask、动态成员、接口撤除、INIT、JOIN/RELEASE、多轮重复 stage、legacy 转换与数值等价、算子与 manifest 阻断、真实性、独立 oracle、八类报告、页面选择、无案例规模硬编码和新包 CLI。
- 旧 62 项测试全部保留，未删除断言、未降低旧测试阈值、未把失败改为跳过。独立 LCP oracle 使用既有求解器合同的 `atol=1e-7`，适配主动集线性方程中固定的 `1e-12` 正则项；oracle 文件未修改。
- 最终全量统计：99 tests、PASS 93、FAIL 0、SKIP 6；其中新增 topology_step 测试 37 项（含真实 Streamlit 步骤选择测试）。
- `python -m compileall -q app.py core tests scripts` 通过；六个旧数据包与 `02_TOPOLOGY_STEP_MIN_CASE` 分别执行 `scripts/cli_check.py`，七次均为 exit 0。
- 使用 `D:\anaconda\envs\thesis\python.exe -m streamlit run app.py --server.headless true` 启动后，`http://127.0.0.1:8501/_stcore/health` 返回 HTTP 200、正文 `ok`，随后已停止本次服务进程。

## topology_step 提交前定向收尾（2026-07-23）

- 收尾前基线为 99 tests、93 PASS、0 FAIL、6 SKIP；原 99 项全部保留，没有删除断言、降低阈值或将失败改为 SKIP。
- 新增 `tests/test_topology_step_closeout.py` 35 项，覆盖数据包真实自校验与失败退出、字段字典、对象映射、156-key MatrixManifest/NPZ、静态附件一致性、生成器重复构建、INIT/中间 NOT_REQUIRED、父机械状态继承、不调用 LCP、RELEASE retain/remove/显式停用优先、未知规则阻断、fallback/物理一致性、CLI 0/1/2/3 语义、预计算倍率禁用、Legacy 倍率回归及 Streamlit AppTest。
- 新增内存小夹具为 2 零件、1 接口、2 接触点/2 维 VectorLayout、6 个步骤；包含 MEASURE、JOIN、RELEASE 删除 joint 与后续 INSPECT，不依赖 TS301、G_AD、G_DB 或 12 维布局。
- topology_step 专项测试合计 72 项：原执行器模块 37 项 + 收尾模块 35 项；专项质量门为 23 项。
- 最终全量统计：134 tests、128 PASS、0 FAIL、6 SKIP。
- `02_TOPOLOGY_STEP_MIN_CASE` 自带校验器为 22/22 PASS、exit 0；任一 blocking check 失败时 exit 非 0。实际 MatrixManifest 与 NPZ 均为 156 个唯一 key，shape/dtype/row-layout/column-layout 一致。
- CLI 固定输出 `FINAL_STATUS`、`BLOCKING_FAIL_COUNT`、`PHYSICAL_FAIL_COUNT`；0=成功，1=阻断校验失败，2=运行时/求解/物理失败，3=未预期异常。

## 阶段实测后验更新增量

- 新增 03 确定性合成包、2 维状态、GAP_G/LAMBDA_N 测量、P/H/R/G_q/F/Q、独立 NumPy 后验 oracle 和独立活动集 LCP oracle。
- 03 本地验证器检查 checkpoint 外键、用途隔离、冻结目标、矩阵清单、协方差传递、oracle、对象映射与真实性边界。
- 完整报告和 CLI 已增加 measurement checkpoint、接受/回滚和失败计数；Streamlit 增加第 15 页并增强第 13 页双状态父链。
- 最终全量统计：169 tests、163 PASS、0 FAIL、6 个既有 V6 acceptance SKIP；原 134 项全部保留。
- 八个正式数据包 CLI 均为 exit 0 / `FINAL_STATUS=PASS`；无 checkpoint 包计数为 0，03 包为 checkpoint 1、attempt 1、accepted 1、rollback 0、update fail 0。
- 03 本地验证器 31/31 PASS，MatrixManifest/NPZ 均为 167 项；生成器连续两次包级 SHA-256 为 `344cb414d9a1c37326668c7d95f03a670c7df3a4d8156cb12c7d6e39368edd32`。
- AppTest 已覆盖 03 第 15 页、checkpoint 选择、prior/posterior 展示、接受状态及 02 未配置提示；Streamlit 健康端点返回 HTTP 200 / `ok`。

## 阶段实测后验更新独立审计定向修复（2026-07-26）

- 修复了原实现用线性后验残差代替 post-LCP 实际物理残差的 BLOCKER；正式接受判据现使用统一 extractor 从后验全局 LCP 状态提取的物理观测及加权残差。
- 03 包的 prior physical residual 为 `0.691859980228161`，posterior linearized residual 为约 `9.32338e-06`，post-LCP physical residual 为 `9.323377868711058e-06`；加权物理指标由 `54150.25458676836` 降至 `2.101731704278381e-05`。
- H 来自独立全局 LCP 中央有限差分，epsilon=`1e-5`，prior、正负扰动和 eta_true 主动集稳定；oracle 不调用生产 update 或 topology runner。
- 修复连续 checkpoint 重复施加 eta：当前 checkpoint 使用增量 q，后续新步骤使用原始算子 q 加完整当前 eta。双 checkpoint 零创新的 q 差异小于 `1e-12`。
- 冻结快照覆盖 SMS、材料/参数、Cn/Ct、mu/beta、连接刚度、W_struct 与 MatrixManifest，并在数值处理前后真实比较。
- 新增 35 项独立审计回归，原 169 项全部保留；最终自动化统计为 204 tests、198 PASS、0 FAIL、6 个既有 V6 acceptance SKIP。
- 四组显式 unittest 分别为 stage measurement 70/70、topology executor 37/37、closeout 35/35、multi-part round2 23/23。
- 03 本地 validator 为 38/38 PASS，MatrixManifest/NPZ 为 167/167；连续两次包级 SHA-256 均为 `c1d24323abda402b89a78ff2362fa4df37347810debb03b40b43598896511c16`。
- 八个正式数据包 CLI 均为 exit 0 / `FINAL_STATUS=PASS` / blocking 0 / physical 0。03 包为 checkpoint 1、attempt 1、accepted 1、rollback 0、update fail 0、物理残差门 PASS；其余七包为 NOT_APPLICABLE。
- 独立 AppTest 2/2 PASS；Streamlit 健康端点返回 HTTP 200 / `ok`，随后已停止测试进程。
