# 后验状态驱动虚拟 SMS 滚动预测进度

## 当前状态

- 当前里程碑：提交前独立审计定向修复已完成，等待人工独立复审。
- 分支：`feature/posterior-virtual-sms-rolling-prediction`。
- 修改前 Git 状态：工作区干净。
- 基线：204 tests，198 PASS，0 FAIL，6 SKIP。
- 03 自校验：38/38 PASS，blocking fail 0。
- posterior 物理残差门：PASS，raw residual `0.691859980228161 -> 9.323377868711058e-06`。

## 已完成事项

- 完整读取任务正文、仓库 `AGENTS.md`、参考索引、两份实现合同和相关辅助说明。
- 检索六份强制参考 PDF，并渲染核对公共 `VectorLayout`、SMS→g0/q、状态继承、KCP 聚合、防重复、验证独立性和 Monte Carlo 外层边界。
- M1：完成 rolling plan、显式样本库、SMS component、assignment、mapping、scenario、sample result、summary 和 run result 数据模型。
- M2：完成九类 rolling 表加载、显式 component order 装配、reference/Delta alpha、MatrixManifest/NPZ shape-layout 校验和 `G_SMS @ Delta alpha` 映射。
- M3：完成从 accepted posterior / predicted cutoff 双源状态恢复的分支滚动执行器；每个正式求解步复用统一 `_execute_step`，保持一个全局 LCP、`W_total = W_struct + Cn`、跨接口块与连接锁定历史。
- 建成独立的 `04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE` 生成器、5 个显式合成样本、独立 active-set LCP/KCP oracle 和数据包 validator。
- 04 当前自校验：61/61 PASS，blocking fail 0；5/5 posterior 样本成功，5/5 predicted-cutoff 对照成功，40/40 q/LCP oracle 行匹配，30/30 KCP oracle 行匹配。
- M4：完成 KCP 贡献账本、15 张 rolling 报告表、trace JSON、独立 rolling ZIP 和完整运行报告可选集成。
- CLI 04：exit 0，rolling plan/run `1/1`，sample success/failure `5/0`，KCP `15`，physical/double-count fail `0/0`，final status PASS。
- CLI 03：exit 0，rolling count `0`，final status NOT_APPLICABLE。
- M5：新增 20 项 rolling 专项测试和 3 维/1 维 SMS 的内存夹具；专项套件 20/20 PASS，生成器连续两次包级哈希一致。
- Streamlit 第 16 页已接入正式入口，支持 plan/sample 选择、q 分解、KCP/账本、对照、汇总、质量门与 ZIP 下载；04/03 AppTest 通过。
- topology_step + stage measurement 专项回归：107 tests PASS。
- `compileall` 通过；最终全量自动化为 224 tests、218 PASS、0 FAIL、6 个既有 V6 acceptance SKIP。
- 五组显式 unittest 分别为 rolling 20/20、stage measurement 70/70、topology executor 37/37、closeout 35/35、multi-part round2 23/23。
- 九个正式数据包 CLI 均为 exit 0 / `FINAL_STATUS=PASS` / blocking 0 / physical 0；无 rolling plan 的八个包为 `NOT_APPLICABLE`。
- 03 validator 为 38/38 PASS；04 validator 为 61/61 PASS；04 生成器连续两次包级 SHA-256 均为 `51f3ad4da7e9a1129c44f264e4711669808991b1d7a4eb534fb5385a96a877d2`。
- AppTest 已覆盖 04 第 16 页、03 第 16 页未配置提示和页面隔离；Streamlit 健康端点返回 HTTP 200 / `ok`，随后停止测试进程。

## 本轮新增文件

- `POSTERIOR_VIRTUAL_SMS_ROLLING_PREDICTION.md`
- `core/rolling_prediction.py`
- `data/04_POSTERIOR_VIRTUAL_SMS_ROLLING_MIN_CASE/`
- `docs/implementation_contracts/后验状态驱动虚拟SMS滚动预测_实现约束.md`
- `docs/worklogs/posterior_virtual_sms_rolling_progress.md`
- `scripts/build_posterior_virtual_sms_rolling_fixture.py`
- `scripts/posterior_virtual_sms_rolling_package_validator.py`
- `tests/rolling_prediction_fixture_factory.py`
- `tests/test_posterior_virtual_sms_rolling.py`

## 本轮修改文件

- `core/topology_step.py`
- `core/reporting.py`
- `scripts/cli_check.py`
- `app.py`
- `README.md`
- `TOPOLOGY_STEP_EXECUTOR.md`
- `STAGE_MEASUREMENT_POSTERIOR_UPDATE.md`
- `MULTI_PART_UPGRADE.md`
- `UPGRADE_TEST_SUMMARY.md`
- `tests/test_core_contracts.py`
- `tests/test_multi_part_upgrade_round2.py`
- `tests/test_stage_measurement_update.py`
- `test_reports/latest/test_report.csv`
- `test_reports/latest/test_report.json`

## 当前失败测试

- 无。

## 提交前独立审计定向修复

- BLOCKER：KCP contribution double-count 现统一传播到 sample、run、summary、trace、quality gate、报告、UI 与 CLI；duplicate 配置故障注入为 run FAIL / CLI exit 2。
- HIGH：新增 first/last、future part 首次加入、持续生效和 expected application 矩阵；正常 04 为 5×4=20 条正式 application，漏项、提前项和 count>1 均失败。
- HIGH：普通 `EFFECTIVE_MAPPING` 与 `EXPLICIT_ZERO_NO_EFFECT` 现由 mapping spec、MatrixManifest 和矩阵范数联合治理。
- HIGH：`kcp_set_id`、aggregation/baseline/failure policy、quality flag、reference SMS 和 coefficient unit 均进入执行前阻断门。
- MEDIUM：posterior 与 predicted baseline 分开执行、记录和统计；正式 sample count 不再包含 baseline failure。
- MEDIUM：不可变门覆盖完整 topology result，包括 prediction start 与全部未来原始步骤；TS301 故障注入能够检测。
- MEDIUM：CLI 按唯一 `virtual_sms_sample_id` 计数，多 future-part 的 10 行 sample×part 表仍计为 5 个样本。
- LOW：包级完整性哈希改为按相对路径、文件长度和原始 bytes SHA-256 组合，不再依赖 pandas/DataFrame 表示。
- rolling 专项当前 33/33 PASS；04 validator 当前 62/62 PASS。

## 尚未完成事项

- 无代码或自动化验证遗留项；按任务要求不执行暂存、提交或推送，等待人工独立复审。

## 独立审计定向修复最终验证（2026-07-27）

- 原 224 项测试全部保留；新增 13 个审计测试方法（含参数化故障子项）后，全量自动化为 237 tests、231 PASS、0 FAIL、6 个既有 V6 acceptance SKIP。
- 五组显式回归分别为 rolling 33/33、stage measurement 70/70、topology executor 37/37、closeout 35/35、multi-part round2 23/23；`compileall` 通过。
- duplicate KCP 配置故障实测：5 个正式样本均为 FAIL，double-count fail count=5，run/trace/report 为 FAIL，CLI exit 2 且 `ROLLING_FINAL_STATUS=FAIL`；所有 5 个样本分支均执行完成，没有把继续执行误判为 run PASS。
- application 故障实测：单样本漏掉应有 application 后，formal sample 1、formal failure 1、application fail count 1，required baseline 也为 1/0/1，最终 run FAIL；正常 04 为 5×4=20 条 application，全部 `application_count=1`。
- 显式零映射实测：role=`EXPLICIT_ZERO_NO_EFFECT`，4 条 application 均保留，mapping matrix norm 最大值 0，q correction norm 最大值 0，run PASS。普通 role+零矩阵、显式零 role+非零矩阵、未知 role 和缺失 mapping 均被阻断。
- predicted-only 故障实测：正式 posterior 仍为 5/5/0，baseline 为 5/4/1，失败记录 role=`PREDICTED`，required baseline policy 使 run FAIL；`POSTERIOR_ONLY` policy 实测不运行 baseline 且保持 PASS。
- TS301 原位修改故障实测：完整 topology hash 门返回 `immutability_status=FAIL`、changed object=`TS301`、run FAIL，对应运行时退出语义为 2。正常完整 topology hash 前后均为 `67695da89cb1ff964fd9087e9fbe179cfe10f32963f82367e314aa050e5e825d`。
- 原始字节哈希故障实测：测试文件单字节变更前为 `5050a62104b997dc6a40173523e27229fd23a1a3f6fd152c937af1da321703c6`，变更后为 `cb6481f34a3443f36af861b4795344ba330a5c4e31b39dfb9bfa30e8a2ae5a98`。
- 04 生成器连续两次结果一致：包级稳定 SHA-256 均为 `28c62d2d9278668dc6cdda3b168f77172dfbd6fef46c5cced8f7c9ccd9402ef7`，每次 validator 62/62 PASS；MatrixManifest/NPZ 为 173/173。
- 03 validator 为 38/38 PASS，04 validator 为 62/62 PASS；04 oracle、5 个样本的 15 个 KCP 预测及 reference baseline 均未回归。
- 九个正式数据包 CLI 均为 exit 0、`FINAL_STATUS=PASS`、blocking 0、physical 0。04 额外为 formal 5/5/0、baseline 5/5/0、application/double-count/immutability fail 均为 0、`ROLLING_FINAL_STATUS=PASS`。
- 第 16 页 AppTest 通过；Streamlit `http://127.0.0.1:8501/_stcore/health` 返回 HTTP 200 / `ok`，测试后 8501 和辅助检查端口 8507 均无监听进程。

## 最终验证入口

如需复核，使用 `D:\anaconda\envs\thesis\python.exe` 运行全量自动化、五组显式 unittest、03/04 validator、九包 CLI、AppTest 与 Streamlit HTTP 健康检查。

## 当前 Git 状态

- 未执行 `git add/commit/push`。
- 未生成发布 ZIP。
## 最终提交前定向修复（2026-07-27）

- direct SMS 布尔字段已成为真实执行分支：`false` 保持 `base + direct`；
  `true` 只取 base，不生成 `FUTURE_SMS_DIRECT_GEOMETRY` 账本项。
- source 链路闭合校验覆盖 plan、checkpoint、measurement update、accepted
  posterior 与实际 rolling source。TS205→TS204 错配为 blocking FAIL / CLI exit 1。
- 第 16 页真实覆盖 5/5 formal failure：`sample_results=0`、
  `sample_failures=5`、run FAIL；无异常、无伪 KCP 或伪描述性统计。
- `POSTERIOR_ONLY` baseline attempt/success/failure=`0/0/0`，
  status=`NOT_APPLICABLE`，comparison 为 0 行 14 列固定 schema 空表。

| 问题 | 测试 | 注入 | 实际/预期 |
|---|---|---|---|
| direct SMS 重复累计 | `test_final_01_direct_sms_boolean_changes_actual_kcp_path` | false/true | 差值等于 direct candidate；true 无 direct 账本，PASS |
| 布尔/聚合治理 | `test_final_02_direct_sms_invalid_and_conflicting_config_blocks` | `yes`、冲突、未知 policy | blocking FAIL |
| source 链路 | `test_final_03_source_checkpoint_and_runtime_linkage_faults_block` | topology/step/update/posterior/role/checkpoint/重复 checkpoint | 全部阻断；正常 04 PASS |
| CLI 配置退出码 | `test_final_04_checkpoint_step_fault_cli_exit_one` | TS205→TS204、非法布尔 | exit 1 |
| UI 空集合/policy | `test_final_05_streamlit_real_all_failure_and_posterior_only` | duplicate KCP、POSTERIOR_ONLY、direct=true | AppTest 无 exception、显示真实状态 |
| 重复声明 | `test_final_06_coefficient_delta_has_single_assignment` | `_coefficient_vectors` | `deltas` 仅声明一次 |

- 最终全量 243 tests：237 PASS、0 FAIL、6 SKIP；原 237 项全部保留，
  新增 6 项且无新增 SKIP。
- 专项：rolling 39/39、stage measurement 70/70、topology executor 37/37、
  closeout 35/35、multi-part 23/23。
- 03 validator 38/38；04 validator 67/67；04 MatrixManifest/NPZ 173/173。
- 04 生成器连续两次包 SHA-256 均为
  `6db1f993c031861d818902b091d578349bbfe38f6e328be26ed16d693e29db6f`；
  KCP oracle 重建前后 SHA-256 均为
  `e9c74c32c03c3e431ccc418a7b27958612e1619c99cf8c5de628a1f2db93defd`。
- 九包 CLI 均 exit 0 / `FINAL_STATUS=PASS` / blocking 0 / physical 0；
  04 formal `5/5/0`、baseline `5/5/0`、source linkage PASS、rolling PASS，
  其余八包 rolling `NOT_APPLICABLE`。
- Streamlit 健康端点 HTTP 200 / `ok`，停止后 8501 无监听。
- 未执行 git add/commit/push，未生成发布 ZIP。
