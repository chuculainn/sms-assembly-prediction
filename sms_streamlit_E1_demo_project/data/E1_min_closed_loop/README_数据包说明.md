# E1_min_closed_loop 测试数据包

这是一个用于软件运行验证的合成最小闭环数据包。结构对象为“CFRP 壁板局部段 + 肋缘条局部连接区”。

## 计算链

1. `I0` 定义零件、结合面、材料、铺层、装配阶段；
2. `I_meas/sms_point_or_node.csv` 提供虚拟 SMS 点/节点偏差；
3. `I_Gamma/contact_points.csv` 和 `I_Gamma/gap_field.csv` 定义候选接触点、面积权重、g0；
4. `I_red/condensed_operator.csv` 和 `matrices/E1_matrices.npz` 提供 W_struct、Cn、q、u_free；
5. `I_stage/stage_plan.csv` 定义 LOCATE / CLAMP / JOIN / RELEASE；
6. Streamlit 程序用主动集算法求解 `0 <= lambda ⟂ g = q + W lambda >= 0`；
7. `validation/validation_kcp.csv` 是合成 VALIDATE 数据，只用于界面展示验证误差。

## 注意

- 该数据包用于验证程序，不是物理真值；
- `W_struct` 与 `Cn_local` 已按不同物理来源拆开；
- `GapField` 保留 `nominal_component` 与 `sms_component`，用于防重复贡献解释。
