# SMS V2.5 数据包模板（01_DEFAULT_MIN_CASE）

本目录依据《数据结构说明书 V2.5 状态传递、局部模型与 N-2-1 过约束配套版》整理。

- `row_mode = default`
- `CSV 中填入了最小默认占位值，可用于软件读取/界面开发测试`
- 该包不是工程真实数据包，不能直接用于论文计算结论。
- 矩阵文件位于 `matrices/default_matrices.npz`，矩阵索引见 `matrices/matrix_manifest.csv`。
- 字段索引见 `field_dictionary.csv`。
- 对象与文件对应见 `object_file_map.csv`。

建议先让 Streamlit 软件支持读取本模板，再逐步替换真实数据。
