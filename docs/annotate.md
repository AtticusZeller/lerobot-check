# Agent Annotation Guide

This guide explains how an annotating agent produces and verifies a subtask annotation table for a LeRobot v3.0 dataset using `lerobot-check`.

## Workflow

Run all commands from the repository root with `uv run`. Never write annotation CSVs, rendered PNGs, or datasets inside the git working tree; write outputs to a directory outside the repository (for example `/tmp/lerobot-check/` or a local data directory).

### 1. Propose Candidate Events

Extract per-arm relative gripper motions, active squeeze intervals, and low-speed TCP segments from `observation.state` and `action`:

```bash
uv run lerobot-check propose \
  --dataset-root /path/to/dataset \
  --episodes 0,1,2 \
  -o /tmp/lerobot-check/proposals.csv
```

The output CSV has columns `episode_index,arm,event_type,frame,end_frame,time_s,value`. All proposed events are **candidates** to narrow visual inspection — always confirm contact, grasp, and subtask boundaries visually on rendered frames rather than copying proposal frames blindly:

- `gripper_close` / `gripper_open`: detected by **relative-amplitude hysteresis** on `observation.state` gripper position (no absolute thresholds). A `gripper_close` fires when gripper position drops by at least `--gripper-delta` (default `0.06`) from its running local maximum; `gripper_open` fires when it rises by at least `--gripper-delta` from its running local minimum. `frame` is where the motion begins leaving its prior plateau, `end_frame` is where it settles, and `value` is the signed amplitude change (`end_pos - start_pos`). This catches both full cycles (`1.0 -> 0.25 -> 1.0`) and small partial cycles while holding a tool (e.g. `0.34 -> 0.24 -> 0.34`).
- `squeeze`: contiguous interval `[frame, end_frame)` where the commanded gripper closes significantly past the observed state (`action_gripper - state_gripper <= -gap`, default `--squeeze-gap 0.05` for at least `--min-squeeze-s 0.2` s); `value` is the median `action - state` gap. A squeeze reliably highlights active object/tool grasps even when a rigid object or tongs physically blocks the jaws from moving far.
- `low_speed`: contiguous interval `[frame, end_frame)` where TCP linear speed stays below `--low-speed-threshold` (default `0.02` m/s); `value` is the median TCP speed.

### 2. Render Multi-Camera Contact Sheets

Render tiled `head | left_wrist | right_wrist` PNG sheets with frame numbers and timestamps drawn on each row.

Start with a coarse pass centered on candidate events (use `--event-types squeeze` to focus specifically on active squeezes, or omit it to include `gripper_close,gripper_open,squeeze`):

```bash
uv run lerobot-check render \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --proposals /tmp/lerobot-check/proposals.csv \
  --event-types squeeze \
  --window-s 1.5 \
  --stride 15 \
  --tile-width 320 \
  -o /tmp/lerobot-check/ep0000_coarse.png
```

If a subtask does not close or squeeze the gripper (for example pushing or flipping a lid with open fingers), render a uniform coarse sweep across the candidate region:

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

- Set `revision` to the exact dataset revision hash (never use a placeholder when the dataset has a known revision; `validate` rejects mismatched revisions).
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

`validate` checks frame coverage, subtask ordering, `contact`/`grasp` bounds, and every row's `revision` against the dataset's detected revision (printing a warning without failing if the local copy has no detectable revision). The command exits with code `0` only when there are zero violations. Fix every reported violation and re-run `validate`.

## Judgement Rules

### Subtask Splitting (`start`, `end`)

- **Split at object release**: pick and place of the same object belong to one subtask. The boundary between subtask $k$ and subtask $k+1$ is the frame where the robot releases the object (or finishes releasing/clearing the articulated part such as a lid) and transitions to the next stage.
- **Half-open coverage**: each row covers `[start, end)`. Within an episode, rows sorted by `start` must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal as long as they cover `[0, episode_length)`.
- **Unannotated episodes**: bad or unannotated episodes may be omitted from the annotation CSV (for example, static/corrupted episodes). Do not invent labels for corrupted episodes; omit them and mention them in your notes.

### Fine-Contact Events (`contact`, `grasp`)

- `contact`: the earliest frame in `[start, end)` where the gripper fingers (or held tool) visibly touch the target object before grasping or manipulating it. Use the wrist camera (`left_wrist` / `right_wrist`) together with `head` to confirm contact rather than depth projection from `head` alone.
- `grasp`: the earliest frame in `[start, end)` where the gripper finishes closing around the object and securely holds it (often aligned with a `gripper_close` or `squeeze` start candidate, confirmed visually when fingers stop closing on the object and lift/motion starts).
- **Ordering invariant**: whenever present, `start <= contact <= grasp < end`.
- **Nullable fields**: if a subtask manipulates an object without closing the gripper (e.g., lifting a carton lid with open jaws), leave `grasp` empty (and `contact` empty if contact cannot be determined reliably). Never fabricate frame numbers.

### Outcome (`success`)

- Set `success` to `true` if the subtask reaches its physical goal before `end` (for example, the lid is open enough for egg access, or the target objects are placed in the tray).
- Set `success` to `false` if the robot fails, drops the object without recovering, or leaves the stage incomplete. Explain failures briefly in `note`.
