# 先读我：Codex说明书仓库包

## 用途

本包用于给Codex提供SMS软件升级所需的长期参考资料。它只包含说明书、检索文本、项目规则和操作说明，不包含也不会替换软件源代码或数据包。

## 解压位置

把压缩包直接解压到最新SMS Streamlit软件的仓库根目录，也就是包含`app.py`、`core/`、`tests/`和`requirements.txt`的目录。

解压后应看到：

```text
项目根目录/
├─ AGENTS.md
├─ README_FIRST.md
├─ CODEX_TASK_STARTER.md
├─ GITHUB_SUBMISSION_GUIDE_WINDOWS.md
├─ app.py
├─ core/
├─ tests/
└─ docs/
   ├─ REFERENCE_INDEX.md
   └─ reference/
```

如果仓库已有`AGENTS.md`，不要直接覆盖：先保留原文件，再把本包`AGENTS.md`中的SMS规则合并进去。

## 提交前检查

1. `docs/reference/00_data_structure_v25_full.pdf`可以打开。
2. `docs/reference/01_data_structure_v25_multipart.pdf`可以打开。
3. `docs/REFERENCE_INDEX.md`可以正常显示中文。
4. `AGENTS.md`位于仓库根目录。
5. 不要把压缩包本身再次提交进仓库；提交解压后的文件即可。
6. 不要把说明书放进`data/`，它们应保留在`docs/reference/`。

## 启动Codex任务

把`CODEX_TASK_STARTER.md`中的启动提示词复制给Codex，再接上本轮完整升级提示词。
