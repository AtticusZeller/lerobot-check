# 补采计划 v1

COLLECTION 计划是供离线审核、现场摆放和进度回传的单个 UTF-8 JSON。格式由 `collection_progress.py` 校验；[合成样例](../tests/fixtures/collection-plan.example.json) 同时供 Python 和 viewer 测试。这里的 v1 是文件版本，属于补采工作流 v2。

## 生成、审核、回传

准备工具点版 `distribution.json`、同一数据集 revision 的 `head.json`，以及 [已审核的物体模板](../README.md#补采目标轮廓)。将模板放在 `<lerobot_root>/.xense-viewer/mask-templates/<org>__<name>@<revision>/`，每个物体一个 JSON。无有效外形的对象可在模板中显式写 `reason`；不要把诊断模板当作审核结果。默认计划路径为 `.xense-viewer/plans/<org>__<name>@<revision>.json`。CLI 的 `--viewer-root` 接受显式的 `.xense-viewer` 目录；`--template` 可重复，提供时替代目录扫描。Viewer 使用 `LOCAL_DATASET_ROOT` 下的同一套路径。

```sh
uv run lerobot-check collection-plan generate \
  --dataset-root "$DATASET" --rounds 20 --objects-per-round 2
```

- `--rounds` 是这次要生成的待采轮数；`--objects-per-round` 是每轮恰好几个不同物体。默认种子为 `0`，可传 `--seed`。轮内同时检查共同基座 XY 和相机像素上的凸包，不允许交叠或接触。已审核模板只近似一个平面；检查不涵盖物体高度、实物误差或未规划物体。采集时把其他物体放到不干扰的位置。本数据集的 `box_base` 只能参加单物体人工轮。多物体候选不足或搜索预算内找不到足够的轮次时，生成失败，已有计划不变。
- 整批重生成保留已采和跳过的轮次，仅替换待采。已采目标补入覆盖统计，按 `object` 与 `episode_index` 已有真实观测时使用真实观测，不重复计数；若计划锚点与真实观测都存在，新点同时避开两者。默认同物体锚点至少相隔半个 XY 分箱宽度，与 yaw 无关。跳过不计数，也不排除位置。改标定、数据集或基准 revision 时归档旧计划，创建新计划。
- 在 COLLECTION 选择一轮可同时预览目标，并可重抽或删除该待采轮。重抽保持物体数、在原位置换入新轮 ID，且避开被换轮、已采及其他待采目标。删除只移除该轮。两种操作都保留其他轮的顺序、目标和进度。CLI 可用 `generate --replace-round ID` 或 `delete-round --plan PATH --round-id ID`；网页请求以 `--expect-plan-id`、`--expect-generation` 防止应用到未审核版本。
- 审核完毕下载 JSON，再将同一文件用 `scp` 传到采集机；由 [viewer #5](https://github.com/AtticusZeller/xense-lerobot-viewer/issues/5) 的显式录制参数加载。外部副本在录制端使用期间不编辑或重新生成笔记本上的计划。采集结束后，取回写有进度的 JSON，合并后在 viewer 点刷新。不要拿返回文件直接覆盖已有计划。

```sh
uv run lerobot-check collection-plan import-progress \
  --plan "$LOCAL_PLAN" --returned "$RETURNED_PLAN"
```

合并要求两份文件的不可变内容完全一致，包括 `plan_id`、`generation`、轮次顺序和目标。允许本地待采转终态、相同终态重复合并；拒绝冲突终态和重复 episode 编号。不支持跨代合并、两台采集机同时写不同副本。CLI `mark-round --plan PATH --round-id ID --episode-index N` 仅供已独立确认保存关系后修复进度，`--skip` 记录跳过。

## JSON 字段

| 字段 | 约定 |
| --- | --- |
| `format`, `version` | 固定为 `"lerobot-collection-plan"`、`1`。消费者拒绝未知格式或版本。 |
| `plan_id`, `generation`, `next_round_id` | 小写标准 UUID、从 1 开始的代次、下一可用轮 ID。任意待采几何编辑推进 `generation`；进度写入不推进。ID 不复用，列表顺序不由 ID 决定。 |
| `repo_id`, `revision`, `source_episode_count` | 数据集标识、生成时基准 revision（单一本地缓存树 commit 或 `"local"`）、首次生成时的 episode 总数。同一计划只追加到这一个本地数据集。 |
| `camera`, `image_size`, `camera_serial` | 头部图像 feature key、`[width,height]` 原生像素、相机序列号字符串（保留前导零）。序列号来自 `intrinsics.serial_number` 或旧版标定的 RealSense 设备说明。`null` 表示可离线审核、不可实时引导。 |
| `calibration_sha256`, `mask_bundle_sha256` | 小写 64 位十六进制，分别是读入的 `head.json` 文件字节与最近一次生成的候选 v1 mask bundle 字节的 SHA-256。每个轮次另记自己使用的 `mask_bundle_sha256`；重生成不改历史轮次的哈希。 |
| `settings` | 最近一次整批生成的 `round_count`、`objects_per_round`、`seed`、`min_revisit_distance_m`、`region_source`。逐轮编辑不改变整批设置；实际剩余数量看 `rounds`。 |
| `excluded_objects` | 最近一次生成中不能参加当前批次的物体及原因，以物体名为键。 |
| `rounds` | 按执行顺序排列的轮次，允许为空。每轮为 `{id, mask_bundle_sha256, targets, status, episode_index}`。 |

每个 `targets` 元素含 `kind: "placement"`、`object`、`index`、`arm`、`x_m`、`y_m`、`z_m`、`yaw_deg`、`yaw_symmetry_deg`、`pixel`、`arrow_end_pixel`、`outline`、`outline_plane_z_m` 和 `reason`。

- `object` 是分布的物体名，同轮唯一。`index` 是该轮对应 bundle 中该物体的候选序号，不等于轮 ID、目标显示编号或 episode 编号。展示编号可按该轮 `targets` 数组从 1 计。
- `arm` 是 `head.json.arms` 的键。`x_m/y_m/z_m` 是该臂基座坐标系下的标定工具锚点（米），不是物体几何中心。`yaw_deg` 是围绕基座正 Z 的折叠角（度，`0 <= yaw_deg < yaw_symmetry_deg`）。`outline_plane_z_m` 是投影轮廓所在基座平面高度（含模板的可见面偏移），与锚点 `z_m` 可以不同。
- `pixel` 与 `arrow_end_pixel` 是原生头图 `[u,v]`，从左上角开始，右/下为正；箭头端点可能超出画幅，绘制时裁切。`outline` 是至少三个点的隐式闭合外环，所有点在原生画幅内，`reason` 为 `null`。无可靠轮廓时，`outline` 和 `outline_plane_z_m` 都为 `null`，`reason` 为非空文本；仅单物体轮允许此形态。
- 计划内嵌所有选中目标，不需要用 `<slug>@<revision>.json` 去找单点 mask。候选 bundle 的确切字节保存在计划同目录的 `bundles/<sha256>.json` 供离线追溯，使用原有 mask bundle v1 结构；旧版 placements/test points、CSV 与 mask bundle 不覆写。导出无需传输候选 bundle。

## #5：录制端读写

在 `lerobot-xense` 复用原样的 [`collection_progress.py`](../src/lerobot_check/collection_progress.py) 单文件（只依赖 Python 标准库；Linux 写入用 `flock`），不必安装完整离线分析栈。直接依赖 lerobot-check 的环境也可从 `lerobot_check.collection_progress` 导入。持锁范围从读取计划开始覆盖录制会话，包含选轮、reset、保存、写回。同一文件的其他写者尝试立即失败；读取不需要锁。

```python
from pathlib import Path

from collection_progress import next_pending, plan_session, validate_capture

with plan_session(Path(explicit_plan_path)) as session:
    plan = session.read()
    validate_capture(
        plan,
        repo_id=local_repo_id,
        revision=local_baseline_revision,
        episode_count=local_episode_count,
        camera=configured_head_feature,
        serial=actual_head_serial,
        image_size=(native_width, native_height),
    )
    while (round_ := next_pending(session.read())) is not None:
        # Recorder owns the camera and obtains frames. The reset-only guide uses id,
        # targets, status counts, outline, pixel, arrow_end_pixel and reason.
        confirmed = reset_and_confirm(round_)
        if not confirmed:
            break
        saved = record_and_save_episode()
        if saved is not None:
            session.mark_collected(round_["id"], saved.episode_index)
```

`next_pending` 按数组顺序取首个 `pending`。`collected` 必须配真正保存成功的零基 `episode_index`（不可小于初始 `source_episode_count`，也不能已属于别的轮）；`pending` 和 `skipped` 的编号必须为 `null`。录制端只调用 `mark_collected` 或显式 `mark_skipped`，不能改目标、代次或顺序。相同终态重复写幂等，已采和跳过不可互换。发生丢弃或重录时继续同轮，不写进度。导出文件中的相机身份要核对实际帧，不能用缩放或裁剪来通过检查。

本地计划锁为同目录 `<plan>.lock`，不会替换或删除。所有写者锁内重读，写临时文件、同步文件内容、同目录原子替换、同步目录；中断后读取应是旧或新完整文件。录制的数据文件与进度 JSON 不是同一个事务：若进程在 episode 落盘后、更新进度前中断，需要记录或独立核对 `(plan_id, round_id, episode_index)` 后才能补记，不能用“下一个编号”自动推断。`source_episode_count`、已知进度和本地 `episode_count` 只用于校验，不用来替代录制器返回的保存编号。

## #2：自动对齐读取的字段

[自动对齐](https://github.com/AtticusZeller/lerobot-check/issues/2) 输入是 reset 中的原生头部帧与当前 `targets`。用 `outline` 和可选 `pixel` 做分割提示并与实际轮廓比较交并比；需要由像素比较基座位置/朝向时，使用与 `calibration_sha256` 匹配的 `head.json`，按 `arm` 选外参，投回 `outline_plane_z_m`；使用 `x_m/y_m/z_m` 作为目标锚点，`yaw_deg` 与 `yaw_symmetry_deg` 定义目标朝向及周期。不要将实际轮廓质心直接与工具锚点相减；两者偏移不一定为零。轮廓为 `null` 的目标（包括当前的 `box_base`）仅供人工确认，不计算自动分数，不得当作自动对齐通过。计划不包含算法、阈值或现场标定；#2 独立提供这些，#5 只需在 reset 调用，不改变录制循环。
