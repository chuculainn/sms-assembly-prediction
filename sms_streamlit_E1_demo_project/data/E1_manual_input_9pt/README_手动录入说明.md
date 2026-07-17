# E1_manual_input_9pt 手动录入测试数据包

这是一个更小的 3×3 候选接触点数据包，用于在当前 `sms_streamlit_E1_demo_project` 中手动输入数据包路径并运行。

## 怎么用

1. 解压本 zip。
2. 把文件夹 `E1_manual_input_9pt` 放到：

```text
sms_streamlit_E1_demo_project/data/E1_manual_input_9pt
```

3. 启动 Streamlit：

```bash
cd sms_streamlit_E1_demo_project
streamlit run app.py
```

4. 在左侧 `标准输入包目录` 输入：

```text
data/E1_manual_input_9pt
```

或输入完整路径。

## 手动改哪些值

优先打开根目录下的：

```text
manual_input_table.csv
```

可以先改这些列：

- `nominal_gap_mm`：名义间隙；
- `sms_component_mm`：SMS 造成的间隙贡献；
- `u_free_S_CLAMP_02_mm`、`u_free_S_JOIN_03_mm`、`u_free_S_RELEASE_04_mm`：各阶段自由闭合量；
- `Cn_local_diag_mm_per_N`：局部界面柔度；
- `W_struct_diag_*_mm_per_N`：结构柔度对角项。

改完后运行：

```bash
python tools/rebuild_npz_from_manual_input.py
```

该脚本会同步更新 `I_Gamma/gap_field.csv`、`I_stage/stage_plan.csv` 和 `matrices/E1_matrices.npz`。

## 注意

这是合成数据，只用于检查软件能否完成：读取标准输入包 → 构造 g0/q/W → 四阶段 LCP → KCP 显示。
