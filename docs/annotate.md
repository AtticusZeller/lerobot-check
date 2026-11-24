# Agent Annotation Guide

This guide describes the **coarse-to-fine visual workflow** for an annotating agent producing and validating subtask annotations on a LeRobot v3.0 dataset with `lerobot-check`.

Gripper signals alone are not reliable enough to determine subtask boundaries or grasps (for example, a gripper may flip a lid with open jaws or hold a tool that blocks finger closure). Annotation is therefore **visual first**: inspect rendered frames at a coarse stride using the subtask's configured camera(s), stop scanning as soon as every required boundary/event is localized, then zoom in around each candidate event.

## Shared Default Paths

When `--task-spec` or `--annotations` is omitted, `lerobot-check` reads and writes the shared `xense-lerobot-viewer` directory under `<lerobot_root>` (the grandparent of `--dataset-root`, e.g. `~/.cache/huggingface/lerobot`):

- Task spec: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- Annotation CSV: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- Reports: `<lerobot_root>/.xense-viewer/reports/<org>__<name>@<revision>/`

Explicit `--task-spec`, `--annotations`, and `-o` arguments remain supported and override defaults. Never write rendered PNGs, proposals, or annotation CSVs into the git repository.

## Workflow

Run commands from the repository root with `uv run`.

### 0. Inspect Dataset Metadata (`info`)

Start by checking `repo_id`, `revision`, `fps`, available `cameras`, episode lengths, and default `.xense-viewer/` paths:

```bash
uv run lerobot-check info --dataset-root /path/to/dataset
```

Use the exact `revision` printed by `info` in the annotation CSV.

### 1. Coarse Pass (Per-Subtask Camera Grid, Stop Once Localized)

Use `--subtask <name>` so `render` selects the camera(s) configured for that subtask in the task spec (for example, `observation.images.right_wrist` for `open_carton`). Render at ~1 s spacing (`::30` at 30 fps) in a compact `--columns N` grid when the subtask uses a single camera:

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --subtask open_carton \
  --frames ::30 \
  --columns 6 \
  --tile-width 240 \
  -o /tmp/lerobot-check/ep0000_coarse.png
```

Inspect generated pages **in order** and **stop as soon as you have located every boundary, `contact`, and `grasp` you need**. Do not read trailing coarse-scan pages after the last target event is already localized (for example, when finding only the `open_carton` -> `place_eggs` boundary in a 2-subtask episode, stop on the page where the lid lies flat).

If a subtask configures multiple cameras in the task spec, omit `--columns` to render them side by side, or pass `--cameras <cam>` explicitly for a single-camera coarse grid.

### 2. Fine Pass (Per-Subtask Cameras, ±1 s at 1–3 Frame Stride)

For each approximate frame found in the coarse pass, render a ±1 s window (`--window-s 1.0`) at 1–3 frame stride (`--stride 2` or `3`, or `--stride 1` when exact precision is needed) using `--subtask <name>`:

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --subtask open_carton \
  --events 411 \
  --window-s 1.0 \
  --stride 3 \
  --tile-width 320 \
  -o /tmp/lerobot-check/ep0000_fine_411.png
```

You can also pass an explicit **half-open** range `[start, end)` with `--frames 381:441:2` (frame `441` is excluded, matching the annotation table's `[start, end)` convention).

Multi-camera stitching is **optional**: pass `--cameras head left_wrist right_wrist` only when the subtask's default camera(s) are occluded or ambiguous. Adjust `--tile-width` and `--max-frames-per-image` to stay within vision token budgets.

### 3. Optional Hints (`propose`)

If helpful for orienting on long episodes, you may run `lerobot-check propose` as an **optional hint** only:

```bash
uv run lerobot-check propose \
  --dataset-root /path/to/dataset \
  --episodes 0 \
  -o /tmp/lerobot-check/proposals.csv
```

Proposed rows (`gripper_close`, `gripper_open`, `squeeze`, `low_speed`) are heuristic candidates, not ground truth. Always confirm every boundary, `contact`, and `grasp` visually on the rendered frames.

### 4. Write the Annotation Table

Write one CSV per `(repo_id, revision)` (by default at `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`) with the exact header:

```text
repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note
```

- Set `revision` to the dataset's exact revision hash from `lerobot-check info`.
- Set `annotator` to `agent` (human reviewers later review and adjust rows in `xense-lerobot-viewer`, which flips edited rows to `human`).
- Set `success` to lowercase `true` or `false`.
- Leave `contact`, `grasp`, or `note` empty when not applicable.

### 5. Validate Against Dataset Metadata

Validate the CSV before finishing (omitting `--task-spec` / `--annotations` uses the shared default paths):

```bash
uv run lerobot-check validate --dataset-root /path/to/dataset
```

Exit code `0` means zero violations; fix all reported issues and re-run.

## Judgement Rules

### Subtask Boundary (`start`, `end`)

- **Split at object release and settle**: pick and place of one object belong to the **same** subtask. Place the boundary between subtask $k$ and subtask $k+1$ (`end` of subtask $k$, `start` of subtask $k+1$) at the **exact frame where the subtask's target object is released and rests stably in its completed state** (for `open_carton`, the moment the lid lies flat on the table).
- **Retraction belongs to the next subtask**: arm retraction or repositioning after the object is released and settled belongs to subtask $k+1$, **not** subtask $k$.
- **Bimanual rule**: judge the boundary by the active arm executing the current subtask's object release (for example, the right arm flipping the lid in `open_carton`), not an idle or preparing support arm.
- **Half-open intervals**: each row covers `[start, end)`. Sorted by `start`, rows within an episode must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal when they cover `[0, episode_length)`.
- **Unannotated / bad episodes**: corrupted or unannotated episodes may be omitted from the table rather than fabricated.

### Contact and Grasp (`contact`, `grasp`)

- **`contact`**: the first frame within `[start, end)` where a gripper finger (or held tool) visibly touches the target object.
- **`grasp`**: the first frame within `[start, end)` where the object is securely held and begins moving with the gripper.
- **Ordering invariant**: whenever present, `start <= contact <= grasp < end`.
- **Nullable fields**: if a subtask completes without grasping (for example, pushing or flipping a lid with open jaws), leave `grasp` empty (and `contact` empty if contact cannot be judged reliably). Never invent frame numbers.

### Per-Subtask Outcome (`success`)

- Evaluate `success` independently for **each subtask** against that subtask's physical goal before `end`:
  - `true`: the subtask completes its goal cleanly.
  - `false`: the robot fails to complete the subtask, drops/displaces the object without recovering, or leaves the stage incomplete. Record a short reason in `note`.
