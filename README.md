# lerobot-check

面向 LeRobot v3.0 数据集的 subtask 标注、校验、重标、分布计算与头部相机标定 CLI 工具。分析结果以结构化 JSON 输出，由 `xense-lerobot-viewer` 原生渲染。

## 功能概览

CLI 入口为 `lerobot-check`，提供以下子命令：

- `info`：打印数据集的 `repo_id`、`revision`、`fps`、相机列表、episode 数、每个 episode 帧数以及默认 `.xense-viewer/` 路径。
- `propose`：基于相对幅度迟滞检测各臂夹爪开合运动、基于 `action - state` 差值检测夹紧区间（`squeeze`），并计算 TCP 低速段，输出候选事件 CSV（可选辅助提示）。
- `render`：低内存解码并输出带帧号与时间戳的 PNG 拼图。支持按 `--subtask` 使用 task spec 配置的相机、`--frames a:b:s` 半开区间 `[a, b)`、单相机网格粗扫（`--columns N`）以及自定义多路拼接（`--cameras`）。
- `validate`：对照数据集元数据中的 episode 长度、数据集 revision 与 task spec 校验标注表，逐条输出违规并以非零状态码退出。标注表中 `contact` 不再要求标注（留空合法），`grasp` 仍用于提取 `event: grasp` 物体的初始位姿。
- `relabel` / `verify-relabel`：按标注表与 task spec 逐帧重标 N 个 subtask 的 `task_index`（视频硬链接不重编码），并校验除 task 相关字段外与源数据零差异。
- `distribution`：输出机读 `distribution.json`（含标注来源统计、各 subtask 帧占比、物体初始位姿与覆盖网格、各 subtask 参与臂的起终点与 0→1 进度重采样轨迹、起终点覆盖与轨迹离散度指标），支持 `--episodes` 过滤；存在 `calib/*.head.json`（或显式传 `--calib`）时自动将各臂位置换算到标定后的工具点 `p_tcp + R_tcp @ tool_offset`，并在 `provenance.position_point` 记录（`tool_point` 或 `tcp`）。
- `suggest`：读取 `distribution.json`、头部图像像素空间目标多边形（`targets/*.json`）与头部相机外参（`calib/*.head.json`），将多边形反投到各物体所在平面（`z` 取事件帧工具点 `z` 中位数），在多边形内按空白优先、稀疏其次、yaw 分箱均衡的原则抽样 N 个补采摆放，并生成 `dense` / `edge` / `empty` 三类测试点（附带头部像素坐标与朝向箭头端点）；写出 `suggestions/*.json`，可选导出 CSV。未提供目标多边形时默认使用数据包围盒外扩 5 cm；若已有标定但 `distribution.json` 仍基于原始 `tcp` 计算，则拒绝混用并提示重跑 `distribution`。
- `calib-frames`：跨全部 episode 对每条臂的 TCP 位置与姿态执行最远点采样（默认每臂 15 帧），写出 `calib/*.frames.json`；已有文件时默认拒绝覆盖（防止重置 ID 导致已点选数据失效），支持 `--append K` 保留已有条目与 ID 追加采样，或显式传 `--overwrite` 重新生成。
- `calibrate`：读取 `calib/*.points.json` 与 `calib/*.intrinsics.json`，按臂独立执行 `solvePnP` 初始化与外参 + 工具偏移 `tool_offset` 联合最小二乘求解，输出 `calib/*.head.json`（含每点残差、RMS 与留一误差）。

面向标注 agent 的操作规范见 [docs/annotate.md](docs/annotate.md)，`distribution.json`、`suggestions.json` 与标定文件的完整字段与指标公式见 [docs/distribution.md](docs/distribution.md)。

## 默认路径约定（与 xense-lerobot-viewer 共用）

设 `<lerobot_root>` 为数据集根目录的上两级目录（例如 `~/.cache/huggingface/lerobot`），`<org>__<name>` 为 `repo_id` 将 `/` 替换为 `__` 后的名称：

- task spec：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- 标注表：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- 分布分析 JSON：`<lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json`
- 目标多边形 JSON：`<lerobot_root>/.xense-viewer/targets/<org>__<name>.json`
- 补采与测试点 JSON：`<lerobot_root>/.xense-viewer/suggestions/<org>__<name>@<revision>.json`
- 头部相机标定文件：`<lerobot_root>/.xense-viewer/calib/<org>__<name>.{intrinsics,frames,points,head}.json`

不传显式路径参数时自动读写上述默认路径；显式传参仍会覆盖默认路径。

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
  --subtask place_insert \
  --frames ::30 \
  --columns 6 \
  -o /tmp/lerobot-check/ep0000_coarse.png

# 3. 校验标注表（默认读写 .xense-viewer/ 路径）
uv run lerobot-check validate --dataset-root /path/to/dataset

# 4. 生成 distribution.json（可加 --episodes 0,30,60 过滤）
uv run lerobot-check distribution --dataset-root /path/to/dataset

# 5. 选取头部相机标定帧并求解外参
uv run lerobot-check calib-frames --dataset-root /path/to/dataset
# （在 viewer 中点选指尖中点保存 points.json 后）
uv run lerobot-check calibrate --dataset-root /path/to/dataset

# 6. 生成补采摆放与三类测试点
uv run lerobot-check suggest --dataset-root /path/to/dataset -n 10
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

本仓库为公开仓库，数据集、标注 CSV、渲染 PNG、标定文件与分布/补采 JSON 均属于数据产物，不得提交到仓库中。请通过默认 `.xense-viewer/` 路径、仓库外路径或 `.gitignore` 忽略的目录输出产物。
