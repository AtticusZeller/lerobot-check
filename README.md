# lerobot-check

面向 LeRobot v3.0 数据集的 subtask 标注、校验、重标、分布计算与头部相机标定 CLI 工具。分析结果以结构化 JSON 输出，由 `xense-lerobot-viewer` 原生渲染。

## 功能概览

CLI 入口为 `lerobot-check`，提供以下子命令：

- `info`：打印数据集的 `repo_id`、`revision`、`fps`、相机列表、episode 数、每个 episode 帧数以及默认 `.xense-viewer/` 路径。
- `propose`：基于相对幅度迟滞检测各臂夹爪开合运动、基于 `action - state` 差值检测夹紧区间（`squeeze`），并计算 TCP 低速段，输出候选事件 CSV（可选辅助提示）。
- `render`：低内存解码并输出带帧号与时间戳的 PNG 拼图。支持按 `--subtask` 使用 task spec 配置的相机、`--frames a:b:s` 半开区间 `[a, b)`、单相机网格粗扫（`--columns N`）以及自定义多路拼接（`--cameras`）。
- `exemplars`：从标注 CSV 中筛选人工复核（`annotator=human`）的边界（`<prev>:<next>`）或抓取帧（`<subtask>:grasp`）示范，按 `[b - window, b + window)` 与给定步长渲染带红框及 `BOUNDARY`/`GRASP` 标记的单 episode 条带 PNG，供标注 agent 在精标阶段对齐人工标准。
- `window-sheet`：按统一时间窗口（`--window-s LO:HI`）或每条 episode 独立的帧窗口 CSV（`--windows FILE`）在窗口内均匀采样帧（默认 8 帧、每页 8 条 episode），输出每行一条 episode、多路相机纵向堆叠的 JPEG 联系表，用于按边界由粗到精定位阶段切换帧。
- `boundary-sheet`：按非末尾 subtask 汇总全部（或 `--episodes` 指定的）episode 的边界检查帧（默认 `end - 1`，可用 `--offset` 调整），将多路相机（默认 `head left_wrist right_wrist`，支持 `--crop` 裁剪）并排拼接为带 `ep<idx> f<frame>` 与人工 `H` 标记的 JPEG 联系表，用于 agent 自检循环与最终人工复核。
- `validate`：对照数据集元数据中的 episode 长度、数据集 revision 与 task spec 校验标注表，逐条输出违规并以非零状态码退出。标注表中 `contact` 不再要求标注（留空合法），`grasp` 仍用于提取 `event: grasp` 物体的初始位姿。
- `relabel` / `verify-relabel`：按标注表与 task spec 逐帧重标 N 个 subtask 的 `task_index`（视频硬链接不重编码），并校验除 task 相关字段外与源数据零差异。
- `distribution`：输出机读 `distribution.json`（含标注来源统计、各 subtask 帧占比、物体初始位姿与覆盖网格、各 subtask 参与臂的起终点与 0→1 进度重采样轨迹、起终点覆盖与轨迹离散度指标），支持 `--episodes` 过滤；存在 `calib/*.head.json`（或显式传 `--calib`）时自动将各臂位置换算到标定后的工具点 `p_tcp + R_tcp @ tool_offset`，并在 `provenance.position_point` 记录（`tool_point` 或 `tcp`）；存在 `object-poses/*.json`（或显式传 `--object-poses`）时为对应物体附加物体层 `object_layer` 并在 `provenance.object_poses_sha256` 记录校验和。
- `object-poses`：按 task spec 中物体的 `layout_box`（第 0 帧头图原图像素框），使用 SAM 2.1 对每条 episode 的第 0 帧做单次图像编码分割，反投到对应臂基座平面 `z = plane_z_m` 计算物体质心与最小外接矩形长轴朝向，写出 `object-poses/*.json` 侧车文件；传 `--sheet DIR` 时同步输出按页拼接的 JPEG 复核联系表。
- `suggest`：读取 `distribution.json`、头部图像像素空间目标多边形（`targets/*.json`）与头部相机外参（`calib/*.head.json`），支持 `--layer object|tool`（默认 `object`）。`object` 层使用各物体的 `object_layer`（头图第 0 帧物体 mask 质心、`plane_z_m` 平面与 `yaw_period_deg` 折叠周期，矩形为 `180°`）；若存在物体缺少 `object_layer`，直接报错列出缺失物体并提示先运行 `lerobot-check object-poses` 再重跑 `distribution`（不静默回退）。`tool` 层沿用事件帧工具点位姿、`median_z_m` 与 `yaw_symmetry_deg`。将多边形反投到对应平面后，按空白优先、稀疏其次、yaw 分箱均衡的原则抽样 N 个补采摆放，并生成 `dense` / `edge` / `empty` 三类测试点（附带头部像素坐标与朝向箭头端点）；写出 `suggestions/*.json`（在 `provenance.layer` 记录所用层），可选导出 CSV。未提供目标多边形时默认使用数据包围盒外扩 5 cm；若已有标定但 `distribution.json` 仍基于原始 `tcp` 计算，则拒绝混用并提示重跑 `distribution`。
- `calib-frames`：跨全部 episode 筛选夹爪闭合（默认 `gripper <= 0.05`，可通过 `--max-gripper` 调整）且低速运动（默认 TCP 速度 `<= 0.03 m/s`，同 episode 中心差分，可通过 `--max-speed` 调整或 `--no-speed-filter` 关闭）的帧，对每条臂的 TCP 位置与姿态执行最远点采样（默认每臂 15 帧），写出含 `speed_mps` 的 `calib/*.frames.json`；已有文件时默认拒绝覆盖（防止重置 ID 导致已点选数据失效），支持 `--append K`（可配合 `--arms left[,right]` 按臂追加）保留已有条目与 ID 并以全部已有帧作为最远点种子继续采样，或显式传 `--overwrite` 重新生成。
- `calibrate`：读取 `calib/*.points.json` 与 `calib/*.intrinsics.json`，按臂独立执行 `solvePnP` / `solvePnPRansac` 初始化与 `soft_l1` 鲁棒外参 + 工具偏移 `tool_offset` 联合最小二乘求解，输出 `calib/*.head.json`（含每点残差与离群点标记 `outlier`、全点与内点 RMS/留一误差）。

面向标注 agent 的操作规范见 [docs/annotate.md](docs/annotate.md)，`distribution.json`、`suggestions.json` 与标定文件的完整字段与指标公式见 [docs/distribution.md](docs/distribution.md)。

## 默认路径约定（与 xense-lerobot-viewer 共用）

设 `<lerobot_root>` 为数据集根目录的上两级目录（例如 `~/.cache/huggingface/lerobot`），`<org>__<name>` 为 `repo_id` 将 `/` 替换为 `__` 后的名称：

- task spec：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- 标注表：`<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- 分布分析 JSON：`<lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json`
- 物体层位姿侧车 JSON：`<lerobot_root>/.xense-viewer/object-poses/<org>__<name>@<revision>.json`
- 目标多边形 JSON：`<lerobot_root>/.xense-viewer/targets/<org>__<name>.json`
- 补采与测试点 JSON：`<lerobot_root>/.xense-viewer/suggestions/<org>__<name>@<revision>.json`
- 头部相机标定文件：`<lerobot_root>/.xense-viewer/calib/<org>__<name>.{intrinsics,frames,points,head}.json`

不传显式路径参数时自动读写上述默认路径；显式传参仍会覆盖默认路径。

## 物体层（Object Layer）

`distribution.json` 中每个物体的既有 `poses`、`skipped`、`coverage` 与 `median_z_m` 为**末端执行器层**（抓取或释放事件帧处的工具点位姿）。对于从未被抓取、或抓取点相对物体几何中心存在偏移的物体（如 `box_base`），可补充**物体层**（`object_layer`），直接从每条 episode 第 0 帧的头部相机图像测量物体自身初始位姿。

在 `task_spec.yaml` 的 `objects[]` 条目中为需要物体层的物体添加 `layout_box: [x0, y0, x1, y1]`（第 0 帧头图原图像素坐标系下覆盖该物体初始摆放范围的矩形框，要求 `0 <= x0 < x1`、`0 <= y0 < y1`），可选添加 `object_plane_z_m` 覆盖反投平面高度（默认取该物体末端执行器层的 `median_z_m`）。未配置 `layout_box` 的物体不会生成物体层。

```bash
# 1. 对第 0 帧头图分割物体轮廓、反投计算初始位姿并生成复核联系表
uv run --extra mask-template lerobot-check object-poses \
  --dataset-root /path/to/dataset \
  --device cuda \
  --sheet /tmp/object-poses-sheets

# 1b. 个别 episode 的初始摆放偏离固定布局时，用 --prompt-box 只替换该 episode 中该物体的框（可重复）
uv run --extra mask-template lerobot-check object-poses \
  --dataset-root /path/to/dataset \
  --device cuda \
  --prompt-box 33:box_lid=415,62,505,158 \
  --prompt-box 34:paper_insert=318,105,387,163 \
  --sheet /tmp/object-poses-sheets

# 2. 重新生成 distribution.json（自动读取默认路径的 object-poses 侧车，也可用 --object-poses 指定）
uv run lerobot-check distribution --dataset-root /path/to/dataset
```

`object-poses` 每帧只编码一次图像，利用每个 `layout_box` 内部的规则正点网格与多掩码筛选提取不触及框边界的单物体掩码（必要时回退框提示以保留跳过诊断），并剔除空掩码、触框掩码、缺失臂标定以及矩形尺寸偏离跨 episode 中位数过大的异常项。物体层角度取反投轮廓最小外接矩形的长轴方向，折叠周期为 `yaw_period_deg = min(yaw_symmetry_deg, 180)`。`--prompt-box EP:OBJECT=x0,y0,x1,y1` 仅对第 `EP` 条 episode 的 `OBJECT` 用给定框（第 0 帧头图原图像素）代替其 `layout_box` 作为分割提示和触框检查范围，其余 episode 仍使用固定布局；该框写入侧车中对应 episode 记录的 `prompt_box` 字段以便复现。物体必须在 task spec 中配置了 `layout_box`，episode 必须在本次处理的 episode 范围内，框必须满足 `0 <= x0 < x1 <= W`、`0 <= y0 < y1 <= H`，同一 episode 与物体重复指定会报错。若存在与数据集匹配的 calibrated `distribution.json`（`--distribution PATH`，或默认路径），`object-poses` 会在写出侧车后逐物体打印末端工具点相对物体的偏移：对每条同时有物体层位姿与末端执行器层位姿的 episode，把 `tool_xy - object_xy` 旋转 `-yaw` 变换到物体坐标系，输出均值与总体标准差（cm），并附基座坐标系下的标准差作对比。物体朝向只在 `yaw_period_deg`（矩形为 180°）内有定义，因此先求各 episode 朝向在该周期上的圆均值作参考，再把每条 episode 的朝向展开到参考值 `±period/2` 内后旋转，避免 0°/180° 附近的同一摆放被当成反向。物体在 episode 间旋转时，基座系标准差混入了旋转，物体系标准差才反映抓取点相对物体的一致性。复核联系表按多条 episode 拼页保存为 `object_poses_p01.jpg` 等 JPEG 文件，每格叠加标注各物体轮廓与名称，并在页底标注跳过物体的原因。

## 补采目标轮廓

沿用已有的 calibrated `distribution.json`、`suggestions.json` 和 `head.json`。这些命令只读数据集。把生成物留在本地，不提交视频帧、标定或模板。

```bash
# 默认 object 层以第 0 帧的物体位姿为锚，取该 episode 第 0 帧头图；--layer tool 时在事件帧附近找一个无遮挡、物体没有移动的头图。
uv run lerobot-check mask-template --distribution "$DIST" --calib "$HEAD" \
  --object iphone --episode 0 --frame 0 --preview /tmp/iphone-source.png --preview-only

# 在原图像素坐标内框选物体；需要时可重复添加 --positive X Y / --negative X Y。
uv run --extra mask-template lerobot-check mask-template \
  --distribution "$DIST" --calib "$HEAD" --object iphone --episode 0 --frame 0 \
  --box 351 184 397 252 --preview /tmp/iphone-mask.png --output /tmp/iphone.json

# 检查预览：轮廓应覆盖物体且不包含夹爪或邻近物体，然后才把模板加入生成命令。
# 不可用的模板也能显式写入原因，例如 box_base 的事件记录的是释放纸垫。
uv run lerobot-check mask-template --distribution "$DIST" --calib "$HEAD" \
  --object box_base --reject 'Release is not the box base pose' --output /tmp/box_base.json

mkdir -p "$LOCAL_DATASET_ROOT/.xense-viewer/masks"
uv run lerobot-check mask-overlays --suggestions "$SUGGESTIONS" --calib "$HEAD" \
  --template /tmp/iphone.json --template /tmp/box_base.json \
  --output "$LOCAL_DATASET_ROOT/.xense-viewer/masks/${SLUG}@${REVISION}.json"
```

`DIST`、`HEAD` 和 `SUGGESTIONS` 是上述默认目录里的已有文件路径；`SLUG` 为 `org__name`，`REVISION` 与建议文件同名。若数据集不在默认本地缓存，给 `mask-template` 加 `--dataset-root /path/to/dataset`。框示例针对 `Xense/pack_iphone_box_1119` 的参考帧，其他图必须重新框选。`mask-template --preview-only` 不需要模型。只有分割步骤使用可选的 Transformers 依赖和固定版本 SAM 2.1 Hiera Tiny（约 156 MB 权重，首次运行下载；缓存后可设 `HF_HUB_OFFLINE=1`）。默认 CPU，也可用 `--device cuda`。

`mask-template` 支持 `--layer object|tool`（默认 `object`），并在模板 JSON 中写入 `"anchor_layer"`（`--reject` 模板同样记录该字段；不含 `anchor_layer` 的旧模板视为 `"tool"`）：

- `object` 层（默认）：使用源 episode 的 `object_layer` 位姿（`x_m`、`y_m`、`folded_yaw_deg`）与 `plane_z_m` 做锚，投影到目标位姿时按 `yaw_period_deg`（矩形为 `180°`，不区分首尾）选取最接近参考角的等价朝向。
- `tool` 层：沿用事件帧记录的标定工具点（`x_m`、`y_m`、`raw_yaw_deg`）与 `median_z_m` 做锚，按 `yaw_symmetry_deg` 折叠，不测量真实物体中心。

`mask-overlays` 与 `collection-plan` 会检查每个模板的 `anchor_layer` 与当前使用的位姿层一致，层不匹配时点名拒绝。若已知可见面的高度不同，用 `--height-offset-m` 给一个相对当前平面的有符号偏移。更换标定、平面或切换位姿层后需重建模板，建议点变化后重新投影。`mask-overlays` 不装模型，不修改采样，也不改变 `suggestions.json`。

## 补采轮次与进度

`collection-plan` 将已审核模板和覆盖采样结果组合为按轮采集的便携 JSON，支持拒绝相交轮次、单轮审核、避开已采位置与回传进度。[补采计划 v1 契约](docs/collection-plan.md) 列出生成命令、字段以及采集端和自动对齐接入方式。

采集端可以把补采计划与头部相机帧交给 [实时轮廓重合判定 API](docs/collection-alignment.md)，在 reset 阶段显示实际轮廓和重合程度。

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
