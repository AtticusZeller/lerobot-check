# Agent Annotation Guide

This guide explains how an annotating agent produces and verifies a subtask annotation table for a LeRobot v3.0 dataset using `lerobot-check`.

## Workflow

Run all commands from the repository root with `uv run`. Never write annotation CSVs, rendered PNGs, or datasets inside the git working tree; write outputs to a directory outside the repository (for example `/tmp/lerobot-check/` or a local data directory).

### 1. Propose Candidate Events

Extract per-arm gripper close/open transitions and low-speed TCP segments from `observation.state`:

```bash
uv run lerobot-check propose \
  --dataset-root /path/to/dataset \
  --episodes 0,1,2 \
  -o /tmp/lerobot-check/proposals.csv
```

The output CSV has columns `episode_index,arm,event_type,frame,end_frame,time_s,value`:

- `gripper_close` / `gripper_open`: frame where the gripper crosses the close or open threshold (with hysteresis). On Xense/Elite rigs, `1.0` is fully open and values toward `0.0` are closed.
- `low_speed`: contiguous interval `[frame, end_frame)` where TCP linear speed stays below the threshold.

### 2. Render Multi-Camera Contact Sheets

Render tiled `head | left_wrist | right_wrist` PNG sheets with frame numbers and timestamps drawn on each row.

Start with a coarse pass centered on candidate gripper events:

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --proposals /tmp/lerobot-check/proposals.csv \
  --window-s 2.0 \
  --stride 15 \
  --tile-width 320 \
  -o /tmp/lerobot-check/ep0000_coarse.png
```

If a subtask does not close the gripper (for example pushing or flipping a lid with open fingers), render a uniform coarse sweep across the candidate region:

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --frames 0:1200:60 \
  --tile-width 320 \
  -o /tmp/lerobot-check/ep0000_sweep.png
```

Then zoom in around each candidate boundary, contact, and grasp with a finer stride (3 to 5 frames):

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --frames 860:940:5 \
  --tile-width 384 \
  -o /tmp/lerobot-check/ep0000_zoom.png
```

Adjust `--tile-width` and `--max-frames-per-image` to balance visual clarity against vision token budget.

### 3. Write the Annotation Table

Write one CSV per `(repo_id, revision)` with the exact header:

```text
repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note
```

- Set `annotator` to `agent`.
- Use lowercase `true` or `false` for `success`.
- Leave `contact`, `grasp`, or `note` empty when not applicable.

### 4. Validate Against Dataset Metadata

Validate every annotation CSV before reporting completion:

```bash
uv run lerobot-check validate \
  --dataset-root /path/to/dataset \
  --task-spec examples/egg_carton/task_spec.yaml \
  --annotations /tmp/lerobot-check/annotations.csv
```

The command exits with code `0` only when there are zero violations. Fix every reported violation and re-run `validate`.

## Judgement Rules

### Subtask Splitting (`start`, `end`)

- **Split at object release**: pick and place of the same object belong to one subtask. The boundary between subtask $k$ and subtask $k+1$ is the frame where the robot releases the object (or finishes releasing/clearing the articulated part such as a lid) and transitions to the next stage.
- **Half-open coverage**: each row covers `[start, end)`. Within an episode, rows sorted by `start` must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal as long as they cover `[0, episode_length)`.
- **Unannotated episodes**: bad or unannotated episodes may be omitted from the annotation CSV (for example, static/corrupted episodes). Do not invent labels for corrupted episodes; omit them and mention them in your notes.

### Fine-Contact Events (`contact`, `grasp`)

- `contact`: the earliest frame in `[start, end)` where the gripper fingers visibly touch the target object (or lid/handle) before grasping or manipulating it. Use the wrist camera (`left_wrist` / `right_wrist`) together with `head` to confirm contact rather than depth projection from `head` alone.
- `grasp`: the earliest frame in `[start, end)` where the gripper finishes closing around the object and securely holds it (typically near the first `gripper_close` event from `propose`, confirmed visually when fingers stop closing on the object and motion starts).
- **Ordering invariant**: whenever present, `start <= contact <= grasp < end`.
- **Nullable fields**: if a subtask manipulates an object without closing the gripper (e.g., lifting a carton lid with open jaws), leave `grasp` empty (and `contact` empty if contact cannot be determined reliably). Never fabricate frame numbers.

### Outcome (`success`)

- Set `success` to `true` if the subtask reaches its physical goal before `end` (for example, the lid is open enough for egg access, or the target objects are placed in the tray).
- Set `success` to `false` if the robot fails, drops the object without recovering, or leaves the stage incomplete. Explain failures briefly in `note`.
