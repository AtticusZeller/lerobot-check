# Agent Annotation Guide

This guide describes the **coarse-to-fine visual workflow** for an annotating agent producing and validating subtask annotations on a LeRobot v3.0 dataset with `lerobot-check`.

Gripper signals alone are not reliable enough to determine subtask boundaries or grasps (for example, a gripper may lift a lid with open jaws or hold a tool that blocks finger closure). Annotation is therefore **visual first**: inspect rendered frames at a coarse stride over the entire episode, then zoom in with all three cameras around each candidate event.

## Workflow

Run commands from the repository root with `uv run`. Never write rendered PNGs, proposals, or annotation CSVs into the git working tree; write outputs outside the repository (for example `/tmp/lerobot-check/`).

### 1. Coarse Pass (Whole Episode, Single Camera Grid)

Render the entire episode from one camera (`head` by default) at ~1 s spacing (`::30` at 30 fps) in a compact `--columns N` grid. Every tile carries its frame index and timestamp (`f=00900 (30.00s)`), so a 2–3 minute episode fits in 1–3 images:

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --frames ::30 \
  --cameras head \
  --columns 6 \
  --tile-width 240 \
  -o /tmp/lerobot-check/ep0000_coarse.png
```

Inspect the coarse grid(s) from start to finish and note the approximate frame of:
- every subtask boundary (object release moment),
- every `contact` (fingers/tool first touching the target object), and
- every `grasp` (object held and beginning to move with the gripper).

### 2. Fine Pass (All Three Cameras, ±1 s at 1–3 Frame Stride)

For each approximate frame found in the coarse pass, render a ±1 s window (`--window-s 1.0`) at 1–3 frame stride (`--stride 2` or `3`, or `--stride 1` when tight precision is needed) using the default multi-camera strip layout (`head | left_wrist | right_wrist`):

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --events 900 \
  --window-s 1.0 \
  --stride 3 \
  --tile-width 320 \
  -o /tmp/lerobot-check/ep0000_fine_900.png
```

You can also pass an explicit slice with `--frames 870:930:2`. Use `left_wrist` and `right_wrist` together with `head` to determine exact frame numbers — wrist views show finger-object contact and release much more reliably than `head` alone. Adjust `--tile-width` and `--max-frames-per-image` to balance visual detail against vision token budget.

### 3. Optional Hints (`propose`)

If helpful for orienting on long episodes, you may run `lerobot-check propose` as an **optional hint** only:

```bash
uv run lerobot-check propose \
  --dataset-root /path/to/dataset \
  --episodes 0 \
  -o /tmp/lerobot-check/proposals.csv
```

Proposed rows (`gripper_close`, `gripper_open`, `squeeze`, `low_speed`) are heuristic candidates, not ground truth:
- `gripper_close` / `gripper_open`: relative-amplitude hysteresis motions on `observation.state` gripper position (`frame` = start, `end_frame` = settle).
- `squeeze`: intervals `[frame, end_frame)` where `action_gripper - state_gripper <= -gap`.
- `low_speed`: intervals `[frame, end_frame)` where TCP linear speed drops below threshold.

Always confirm every boundary, contact, and grasp visually on the rendered frames.

### 4. Write the Annotation Table

Write one CSV per `(repo_id, revision)` with the exact header:

```text
repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note
```

- Set `revision` to the dataset's exact revision hash (never use a placeholder when the dataset has a known revision; `validate` rejects mismatched revisions).
- Set `annotator` to `agent`.
- Set `success` to lowercase `true` or `false`.
- Leave `contact`, `grasp`, or `note` empty when not applicable.

### 5. Validate Against Dataset Metadata

Validate the CSV before finishing:

```bash
uv run lerobot-check validate \
  --dataset-root /path/to/dataset \
  --task-spec examples/egg_carton/task_spec.yaml \
  --annotations /tmp/lerobot-check/annotations.csv
```

`validate` checks frame coverage (`[0, episode_length)` with no gaps or overlaps), subtask order, `start <= contact <= grasp < end`, and every row's `revision` against the detected dataset revision (warning without failing only if the local dataset has no detectable revision). Exit code `0` means zero violations; fix all reported issues and re-run.

## Judgement Rules

### Subtask Boundary (`start`, `end`)

- **Split at object release**: pick and place of one object belong to the **same** subtask. The boundary between subtask $k$ and subtask $k+1$ (`end` of subtask $k$, `start` of subtask $k+1$) is the frame where the gripper releases the object of the finished subtask (or clears the articulated part such as an opened lid) and begins transitioning to the next subtask.
- **Half-open intervals**: each row covers `[start, end)`. Sorted by `start`, rows within an episode must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal when they cover `[0, episode_length)`.
- **Unannotated / bad episodes**: corrupted or unannotated episodes may be omitted from the table rather than fabricated.

### Contact and Grasp (`contact`, `grasp`)

- **`contact`**: the first frame within `[start, end)` where a gripper finger (or held tool) visibly touches the target object. Confirm on the wrist camera (`left_wrist` / `right_wrist`) as well as `head`.
- **`grasp`**: the first frame within `[start, end)` where the object is securely held and begins moving with the gripper.
- **Ordering invariant**: whenever present, `start <= contact <= grasp < end`.
- **Nullable fields**: if a subtask completes without grasping (for example, pushing or flipping a lid with open jaws), leave `grasp` empty (and `contact` empty if contact cannot be judged reliably). Never invent frame numbers.

### Per-Subtask Outcome (`success`)

- Evaluate `success` independently for **each subtask** against that subtask's physical goal before `end`:
  - `true`: the subtask completes its goal cleanly (for example, in `open_carton` the lid is opened wide enough to access the eggs; in `place_eggs` the eggs are placed in the tray).
  - `false`: the robot fails to complete the subtask, drops/displaces the object without recovering, or leaves the stage incomplete. Record a short reason in `note`.
