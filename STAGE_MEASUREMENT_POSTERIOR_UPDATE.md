# 阶段实测后验更新与状态回代

## 范围

本功能在确定性 `topology_step` 执行器之上增加过程测量检查点。它只更新数据包定义的低维阶段状态 `eta`，不辨识或写回 Cn、Ct、mu、beta_r、连接刚度、装配前 SMS，也不执行在线有限元重构、EKF/UKF、Monte Carlo 后验传播或虚拟 SMS 滚动预测。

合成 03 数据包仅用于数值一致性、软件联调和算法自洽验证，`engineering_claim_allowed=false`，不得据此声明工程精度。

## 数据流

每个启用的 `measurement_checkpoint_id` 唯一关联一个 `topology_step`，并显式声明更早的机械源步骤：

```text
上一有效状态
  -> checkpoint topology_step 的 PREDICTED 状态
  -> 过程测量筛选与线性高斯更新
  -> 当前机械状态的一次统一全局 LCP 重求
  -> POSTERIOR 状态（成功）或回滚 PREDICTED 状态（失败）
  -> 下一 topology_step 的 PREDICTED 状态
```

PREDICTED 与 POSTERIOR 同时保留。只有已接受的 POSTERIOR 才能成为后续步骤的父状态；失败后保留预测状态，并生成 `UpdateRollbackRecord`。

## 数学形式

低维状态不是完整有限元位移，而是由 `I_stage/state_update_basis.csv` 定义的阶段修正坐标。第一版采用一次确定性线性高斯更新：

```text
r = z_measured - z_predicted
S = H P_prior H^T + R
K = P_prior H^T S^-1
eta_post = eta_prior + K r
P_post = (I-KH) P_prior (I-KH)^T + K R K^T
```

实现使用 Cholesky 和 `numpy.linalg.solve`，不显式计算矩阵逆；后验协方差采用 Joseph 形式，并检查有限性、对称性、半正定性与 NIS。

观测结果明确分为：

- `z_predicted_prior_physical`：更新前实际物理状态；
- `z_predicted_posterior_linearized`：线性观测模型预测，仅用于诊断；
- `z_predicted_posterior_physical`：后验全局 LCP 回代后的实际物理状态。

正式接受判据使用第三者的原始、标准化和 `r^T R^-1 r` 加权残差，并结合配置阈值、单项恶化、LCP、NIS、PSD 和状态链质量门。线性残差改善不能替代实际物理改善。

状态通过显式矩阵映射进入自由间隙：

```text
当前 checkpoint:
q_post = q_source_effective + G_q (eta_post - eta_prior)

后续新 topology_step:
q_effective = q_operator_base + G_q eta_current
W_total = W_struct + Cn
0 <= lambda ⟂ q_effective + W_total lambda >= 0
```

重求仍使用公共 `VectorLayout` 下所有活动接口的单一全局 LCP，保留 `W_struct` 的跨接口非对角块。不得按接口分别求解后拼接。

后续步骤按 `eta_k^- = F_k eta_parent^+`、`P_k^- = F_k P_parent^+ F_k^T + Q_k` 传播。数据包可显式提供 F/Q；缺失时只使用并记录第一版 identity/zero 策略，不冒充全阶协方差传播。

## 测量治理

- 优先读取 `I_meas/measurement_record.csv`，兼容 `I_meas/process_record.csv`。
- `CALIBRATE` 可更新；`UPDATE` 由适配器显式规范化为 `CALIBRATE`。
- `VALIDATE` 只评价，不回流；`IDENTIFY` 不进入本轮状态更新。
- `VALIDATE` 报告 prior/post 物理残差并标记 `EVALUATION_ONLY`；`IDENTIFY` 标记 `SKIPPED_IDENTIFY`。只含评价/跳过观测时不更新、不重求 LCP、不生成 rollback。
- 缺失 `data_role` 不会被默认允许。
- 指向参数、SMS、Cn、Ct、mu、beta_r 或连接刚度的 `update_target` 被阻断。
- sample/reference state、stage/checkpoint、measurement set/ID、vector source、坐标系、单位、活动接口、object/interface/global index 和质量标志均需通过质量门。
- prior、post-LCP 和 VALIDATE 共用 `extract_observation_vector`；第一版只支持 `GAP_G`、`LAMBDA_N`、`PRESSURE_P_N`、`LOCAL_COMPRESSION_W_N` 与 `RUNTIME_STAGE_STATE`。
- 声明完整 R block 后任何读取、shape、layout、有限性、对称性或正定性错误均阻断；只有完全未声明且配置显式允许时才由正的 `standard_uncertainty^2` 构造对角矩阵。
- 运行时 CSV 只能覆盖匹配 checkpoint/measurement ID 的 value 和 standard uncertainty；不会写回数据包，也不能改写治理字段。

## 03 合成最小数据包

`data/03_STAGE_MEASUREMENT_UPDATE_MIN_CASE` 由 `scripts/build_stage_measurement_update_fixture.py` 从 02 规范构建，增加一个 MEASURE 步骤、一个 checkpoint、2 维低维状态以及 GAP_G/LAMBDA_N 两条合成测量。测量由 `eta_true -> q_true -> 独立全局 LCP -> 统一 observation extractor` 生成；H 使用 `eta_prior` 附近的独立中央有限差分，并记录正负扰动活动集稳定性。包内含 eta/P/H/R、源步骤及全部后续求解步骤的 G_q、显式 F/Q、独立 NumPy 后验 oracle 和独立活动集 LCP oracle。

本地验证器为 `validation/validate_package.py`。生成器采用固定输入和确定性 NPZ 写入，连续重建应产生相同包级哈希。

## 报告、CLI 与 UI

完整运行报告包含 checkpoint、更新摘要、创新、观测映射、决策、重求要求、预测/后验快照与对比、协方差轨迹、回滚、状态谱系和 JSON trace。

CLI 额外输出：

- `MEASUREMENT_CHECKPOINT_COUNT`
- `MEASUREMENT_UPDATE_ATTEMPT_COUNT`
- `POSTERIOR_ACCEPTED_COUNT`
- `POSTERIOR_ROLLBACK_COUNT`
- `MEASUREMENT_UPDATE_FAIL_COUNT`
- `POSTERIOR_PHYSICAL_RESIDUAL_GATE`

Streamlit 第 15 页展示 checkpoint 路线、测量、更新决策、prior/posterior 指标、物理重求、回滚和后续步骤影响；第 13 页同步展示双状态父链字段。

## 当前未实现

当前不支持状态—参数联合估计、参数辨识、在线 FE/Schur 更新、非线性观测、EKF/UKF/粒子滤波、多次高斯—牛顿、完整摩擦 NCP、后验 Monte Carlo 或生产级工程精度声明。
