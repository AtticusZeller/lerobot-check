# 补采：用轮廓重合提示摆放到位

`lerobot_check.alignment.AlignmentEvaluator` 比较一帧头部相机 RGB 图和补采计划里本轮的目标轮廓。它不打开相机，不读取或写入录制进度，也不依赖 viewer。它只能判断**当前画面里的二维轮廓重合**：不判断 iPhone 正反方向、平面位姿或物体身份，也不能在遮挡下保证正确。“到位”是提示，仍由操作员确认。

## #5 接入

安装已有的可选模型依赖，在采集环境预缓存固定版本权重：

```bash
uv sync --extra mask-template
uv run hf download facebook/sam2.1-hiera-tiny config.json preprocessor_config.json \
  processor_config.json model.safetensors \
  --revision de431c4043854a71d8101e17995dfe596bf101a5
```

以下代码放在 reset 阶段的推理线程，不放在相机读取或窗体线程。录制端先验证计划与录制数据集、实际相机序列号及原生画幅匹配；判定只用像素，无需另外传 `head.json`。

```python
from pathlib import Path

from lerobot_check.alignment import AlignmentEvaluator
from lerobot_check.collection_progress import load_plan, next_pending

plan = load_plan(Path(explicit_plan_path))
pending_round = next_pending(plan)
if pending_round is None:
    raise ValueError("No pending collection round")

# 阈值用默认值时省略路径。
evaluator = AlignmentEvaluator.from_files(Path("thresholds.json"), device="cuda")
prepared = evaluator.prepare_round(plan, pending_round["targets"])
# rgb 来自录制进程：np.ndarray，uint8，[height, width, 3]，RGB。
results = evaluator.evaluate(rgb, prepared)
all_aligned = bool(results) and all(result.passed for result in results)
```

`load_plan` 校验 viewer 导出的完整 v1 JSON，返回 `CollectionPlan`。录制会话已持锁时，直接用 `session.read()` 返回的同类型对象。`prepare_round` 接受该计划和所选轮次的 `list[PlanTarget]`，使用计划原生画幅、目标的 `object`、`index`、`outline` 和 `reason`。轮廓是像素坐标的隐式闭合外环。不使用 `arm`、`yaw_deg` 或 `outline_plane_z_m`。无效轮廓或不匹配的帧尺寸报 `ValueError`，不自动缩放。

每个目标按输入顺序返回 `AlignmentResult`：

| 字段 | 含义 |
| --- | --- |
| `object`, `index` | 与计划中的目标对应。 |
| `observed_outline` | SAM 预测 mask 的最大外轮廓，可直接画到原生画面；不可用时为 `None`。 |
| `iou` | 完整实际 mask 与栅格化目标 mask 的交集像素数除以并集像素数；不可用时为 `None`。 |
| `passed` | 有可用轮廓且 `iou >= min_iou` 才为 `True`。 |
| `status`, `reason` | `aligned` 表示达到阈值，`misaligned` 表示低于阈值，`undetermined` 表示未分出可用轮廓，`manual` 表示不判定。界面可直接显示 `reason`。 |

`outline: null` 的目标不做分割，保留计划的 `reason` 或用人工确认提示。全是人工目标时，`evaluate` 不调用模型。每帧对目标边界框扩边，一次图像编码后批量解码，取各提示下模型分数最高的 mask；不沿用上一帧的 mask。附近的相似物体可能被选中，不能把 IoU 当作身份校验。

## 阈值与调度

默认 `min_iou` 为 `0.60`。本地 `Xense/pack_iphone_box_1119` 的遮挡前帧中，使用抓取事件的工具位姿投影目标得到如下正样本代理，源模板 episode 不计入：

| 物体 | 样本 | IoU 第 10 百分位 | IoU 中位数 | IoU 第 90 百分位 | 达到 0.60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `paper_insert` | 28 | 0.613 | 0.734 | 0.805 | 26/28 |
| `iphone` | 27 | 0.527 | 0.729 | 0.860 | 22/27 |
| `box_lid` | 29 | 0.512 | 0.613 | 0.759 | 18/29 |

初版阈值取在较难的盒盖中位数附近，允许操作员处理低重合样本。这些分数也含事件代理位姿和模板投影误差，不用于估计误通过率。操作员可用以下文件替换阈值，范围为 `(0, 1]`：

```json
{"version": 1, "min_iou": 0.60}
```

- 一个后台线程独占 evaluator，在进入引导前加载缓存模型并用一帧预热。每轮准备一次目标，推理串行；线程间只传最新帧的副本或独占缓冲，不无限排队。
- 界面按轮次、帧序号和时间戳丢弃过时结果；换轮立即清空旧提示。退出 reset 后停止提交并让在途任务结束，再开始录制。模型不在录制循环里运行，相机始终由录制进程读取。
- 模型未安装、缓存缺失、推理异常时由 #5 显示原因并允许显式人工确认。默认使用 FP32；`precision="bf16"` 只在支持 BF16 的 CUDA 设备上可选，上线前比较同一批帧的判定差异。
