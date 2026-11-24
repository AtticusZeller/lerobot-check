# lerobot-check

面向 LeRobot v3.0 数据集的 subtask 标注与分析工具。

## 功能概览

当前提供以下 CLI 子命令（入口 `lerobot-check`）：

- `propose`：基于相对幅度迟滞检测各臂夹爪的闭合 / 张开运动，结合 `action` 与 `observation.state` 夹爪差值检测夹紧区间（`squeeze`），并基于 TCP xyz 差分计算线速度与低速段，输出候选事件 CSV。
- `render`：低内存解码并输出带帧号与时间戳的 PNG 拼图，支持单相机网格粗扫（`--columns N`，如全 episode 按 1 s 步长）以及默认三路相机（`head`、`left_wrist`、`right_wrist`）精细逐帧查看。
- `validate`：对照数据集元数据中的 episode 长度、数据集 revision 与 task spec 校验标注表，逐条输出违规并以非零状态码退出。

面向标注 agent 的操作规范见 [docs/annotate.md](docs/annotate.md)。

## 安装与环境

要求 Python >= 3.12 与 [uv](https://docs.astral.sh/uv/)，依赖固定版本的 PyPI `lerobot==0.5.1`（不依赖 `lerobot-xense` 或任何硬件 SDK）：

```bash
uv sync --all-groups
```

## 快速使用

```bash
# 1. 提取候选事件
uv run lerobot-check propose \
  --dataset-root /path/to/dataset \
  --episodes 0,1 \
  -o /tmp/lerobot-check/proposals.csv

# 2. 渲染候选事件附近的三路相机拼图
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --proposals /tmp/lerobot-check/proposals.csv \
  --window-s 1.5 \
  --stride 15 \
  -o /tmp/lerobot-check/ep0000.png

# 3. 校验标注表
uv run lerobot-check validate \
  --dataset-root /path/to/dataset \
  --task-spec examples/egg_carton/task_spec.yaml \
  --annotations /tmp/lerobot-check/annotations.csv
```

## 鸡蛋盒示例

[examples/egg_carton/](examples/egg_carton/) 包含鸡蛋盒两阶段任务的 task spec 与旧 `cuts.csv` 转换脚本：

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

本仓库为公开仓库，数据集、标注 CSV、渲染 PNG 与分析报告均属于数据产物，不得提交到仓库中。请通过 CLI 参数将产物输出到仓库外路径或 `.gitignore` 忽略的默认目录（`outputs/`、`annotations/`、`reports/`、`renders/`、`datasets/`）。
