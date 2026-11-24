# lerobot-check

面向 LeRobot v3.0 数据集的 subtask 标注、校验、重标与数据分布分析工具。

## 功能概览

CLI 入口为 `lerobot-check`，提供以下子命令：

- `info`：打印数据集的 `repo_id`、`revision`、`fps`、相机列表、episode 数、每个 episode 帧数以及默认 `.xense-viewer/` 路径。
- `propose`：基于相对幅度迟滞检测各臂夹爪开合运动、基于 `action - state` 差值检测夹紧区间（`squeeze`），并计算 TCP 低速段，输出候选事件 CSV（可选辅助提示）。
- `render`：低内存解码并输出带帧号与时间戳的 PNG 拼图。支持按 `--subtask` 使用 task spec 配置的相机、`--frames a:b:s` 半开区间 `[a, b)`、单相机网格粗扫（`--columns N`）以及自定义多路拼接（`--cameras`）。
- `validate`：对照数据集元数据中的 episode 长度、数据集 revision 与 task spec 校验标注表，逐条输出违规并以非零状态码退出。
- `relabel` / `verify-relabel`：按标注表与 task spec 逐帧重标 N 个 subtask 的 `task_index`（视频硬链接不重编码），并校验除 task 相关字段外与源数据零差异。
- `report`：生成离线中文 HTML 数据分布报告（`index.html`）与机读摘要（`summary.json`），支持页首标注来源统计与未复核 `agent` 警告、单数据集按 episode 分组着色（`--group`），以及嵌入 Agent 补采建议（`--advice`）。
- `mix`：计算多份数据集按重复次数加权混合后的各 subtask 与精细窗口帧占比（不写数据集）。
- `suggest`：从覆盖网格的稀疏格子中为每个物体独立抽样，输出下一批 N 条的摆放建议（未检查物体间碰撞）。

面向标注 agent 的操作规范见 [docs/annotate.md](docs/annotate.md)，面向补采建议 agent 的规范及 `summary.json` 字段说明见 [docs/advise.md](docs/advise.md)。

## 默认路径约定（与 xense-lerobot-viewer 共用）

设 `<lerobot_root>` 为数据集根目录的上两级目录（例如 `~/.cache/huggingface/lerobot`），`<org>__<name>` 为 `repo_id` 将 `/` 替换为 `__` 后的名称：

- task spec：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- 标注表：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- 分析报告与摘要：`<lerobot_root>/.xense-viewer/reports/<org>__<name>@<revision>/`（含 `index.html`、`summary.json`）

不传 `--task-spec`、`--annotations` 或 `report -o` 时自动读写上述默认路径；显式传参仍会覆盖默认路径。

## 安装与环境

要求 Python >= 3.12 与 [uv](https://docs.astral.sh/uv/)，依赖固定版本的 PyPI `lerobot==0.5.1`（不依赖 `lerobot-xense` 或任何硬件 SDK）：

```bash
uv sync --all-groups
```

## 快速使用

```bash
# 1. 查看数据集元信息、revision 与每条 episode 长度
uv run lerobot-check info --dataset-root /path/to/dataset

# 2. 按 subtask 配置的相机渲染粗扫网格（半开区间 [a, b)）
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --subtask open_carton \
  --frames ::30 \
  --columns 6 \
  -o /tmp/lerobot-check/ep0000_coarse.png

# 3. 校验标注表（默认读写 .xense-viewer/ 路径）
uv run lerobot-check validate --dataset-root /path/to/dataset

# 4. 生成中文离线报告与 summary.json（支持按 episode 分组着色与嵌入建议）
uv run lerobot-check report \
  --dataset-root /path/to/dataset \
  --advice /tmp/lerobot-check/advice.md
```

## 鸡蛋盒示例

[examples/egg_carton/](examples/egg_carton/) 包含鸡蛋盒两阶段任务的 task spec（含按 subtask 配置的相机列表）与旧 `cuts.csv` 转换脚本：

```bash
uv run python examples/egg_carton/convert_cuts.py \
  --dataset-root ~/.cache/huggingface/lerobot/Xense/bi_elite_cs66_rt_egg_carton_to_tray_1016_fixed \
  --cuts /path/to/cuts.csv \
  -o /tmp/lerobot-check/egg_carton_annotations.csv
```

## 开发与测试

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
```

## 数据边界说明

本仓库为公开仓库，数据集、标注 CSV、渲染 PNG 与分析报告均属于数据产物，不得提交到仓库中。请通过默认 `.xense-viewer/` 路径、仓库外路径或 `.gitignore` 忽略的目录输出产物。
