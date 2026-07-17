# Windows本地仓库提交到GitHub

## 一、解压

1. 找到最新SMS Streamlit本地仓库。
2. 确认目录内有`app.py`、`core`和`tests`。
3. 把本压缩包直接解压到该目录。
4. 如果Windows询问是否合并`docs`目录，选择合并。
5. 如果已经存在`AGENTS.md`，先比较并合并内容，不要盲目覆盖。

## 二、用VS Code提交（推荐）

1. 用VS Code打开仓库根目录。
2. 点击左侧“源代码管理”图标。
3. 查看新增文件，应主要是`AGENTS.md`、`docs/`和本包的说明文件。
4. 确认没有把本地虚拟环境、缓存或密码文件加入提交。
5. 点击“暂存所有更改”。
6. 提交信息填写：

   `Add SMS model reference documents for Codex`

7. 点击“提交”。
8. 点击“同步更改”或“Push”。
9. 打开GitHub网页，确认目标分支已经出现`AGENTS.md`和`docs/reference/`。

## 三、用Anaconda Prompt或终端提交

先进入仓库目录。例如：

```bat
D:
cd D:\你的路径\sms_streamlit_project
```

检查仓库状态：

```bat
git status
```

只暂存本次说明书文件：

```bat
git add AGENTS.md README_FIRST.md CODEX_TASK_STARTER.md GITHUB_SUBMISSION_GUIDE_WINDOWS.md docs
```

再次检查：

```bat
git status
```

提交：

```bat
git commit -m "Add SMS model reference documents for Codex"
```

推送：

```bat
git push
```

如果这是新分支第一次推送，Git提示没有上游分支时，按提示使用：

```bat
git push -u origin 当前分支名
```

## 四、检查是否成功

在GitHub仓库网页确认存在：

```text
AGENTS.md
docs/REFERENCE_INDEX.md
docs/reference/00_data_structure_v25_full.pdf
docs/reference/01_data_structure_v25_multipart.pdf
```

然后再启动Codex云端任务。云端Codex只能读取已经提交并推送到目标分支的文件，不能读取电脑D盘中尚未上传的说明书。

## 五、不要提交的内容

- `.venv/`或`env/`；
- `__pycache__/`；
- `.streamlit/secrets.toml`；
- API密钥和账号密码；
- 软件运行生成的大量临时输出；
- 本说明书压缩包本身。
