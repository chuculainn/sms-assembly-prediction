`cli_check.py` 可在命令行下读取默认数据包并输出质量门、阶段求解和 KCP 验证结果。

完整数据由项目生成脚本创建；如需扩展为 15/25/49 候选点或加入 N-2-1 扩展域，可以在 `data/E1_min_closed_loop` 基础上复制一份并替换 `contact_points.csv` 与 `matrices/E1_matrices.npz`。
