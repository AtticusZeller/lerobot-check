# Agent Annotation Guide

This guide describes the **coarse-to-fine visual workflow** for an annotating agent producing and validating subtask annotations on a LeRobot v3.0 dataset with `lerobot-check`.

Gripper signals alone are not reliable enough to determine subtask boundaries or grasps (for example, a gripper may flip a lid with open jaws or hold a tool that blocks finger closure). Annotation is therefore **visual first**: inspect rendered frames at a coarse stride using the subtask's configured camera(s), stop scanning as soon as every required boundary and `grasp` frame is localized, then zoom in around each candidate event.

## Shared Default Paths

When `--task-spec` or `--annotations` is omitted, `lerobot-check` reads and writes the shared `xense-lerobot-viewer` directory under `<lerobot_root>` (the grandparent of `--dataset-root`, e.g. `~/.cache/huggingface/lerobot`):

- Task spec: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- Annotation CSV: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- Distribution JSON: `<lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json`
- Suggestions JSON: `<lerobot_root>/.xense-viewer/suggestions/<org>__<name>@<revision>.json`

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

Inspect generated pages **in order** and **stop as soon as you have located every boundary and `grasp` you need**. Do not read trailing coarse-scan pages after the last target event is already localized (for example, when finding only the `open_carton` -> `place_eggs` boundary in a 2-subtask episode, stop on the page where the lid lies flat).

If a subtask configures multiple cameras in the task spec, omit `--columns` to render them side by side, or pass `--cameras <cam>` explicitly for a single-camera coarse grid.

### 2. Fine Pass (Compare Against Human Exemplars, ±1 s at 1–3 Frame Stride)

Before placing a boundary (or `grasp`) in a new episode, render the user's own human-reviewed exemplars for that boundary so you match the **exact same physical event**:

```bash
uv run lerobot-check exemplars \
  --repo-id Xense/pack_iphone_box_1119 \
  --boundary place_phone:move_lid \
  --max-episodes 3 \
  --window-s 1.0 \
  --stride 3 \
  -o /tmp/lerobot-check/exemplars
```

For grasp frames, pass `--boundary <subtask>:grasp` (for example `--boundary place_phone:grasp`). Each exemplar PNG shows the subtask's configured cameras (the union of both subtasks' `cameras` for a transition boundary) over `[b - window, b + window)` with the human-reviewed frame highlighted by a red border and `BOUNDARY` / `GRASP` badge. If no human exemplars exist yet for that boundary, `exemplars` prints a message and exits `0` so you can proceed from the written rules.

The task spec comments state the written rule; the exemplars show the user's standard. **If the exemplars and the written rule disagree, follow the exemplars and record the discrepancy in the row's `note`.**

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

Proposed rows (`gripper_close`, `gripper_open`, `squeeze`, `low_speed`) are heuristic candidates, not ground truth. Always confirm every boundary and `grasp` visually on the rendered frames.

### 4. Write the Annotation Table

Write one CSV per `(repo_id, revision)` (by default at `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`) with the exact header:

```text
repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note
```

- Set `revision` to the dataset's exact revision hash from `lerobot-check info`.
- Set `annotator` to `agent` (human reviewers later review and adjust rows in `xense-lerobot-viewer`, which flips edited rows to `human`).
- Set `success` to lowercase `true` or `false`.
- **Leave `contact` empty** (`contact` is no longer annotated; the column stays in the CSV schema for compatibility and empty is legal).
- Annotate `grasp` for every subtask that grasps or holds something, whether or not the task spec lists an `objects` entry for it (for example `flip_lid`); leave `grasp` empty when the subtask does not grasp. Leave `note` empty when not applicable.

### 5. Validate Against Dataset Metadata

Validate the CSV before finishing (omitting `--task-spec` / `--annotations` uses the shared default paths):

```bash
uv run lerobot-check validate --dataset-root /path/to/dataset
```

Exit code `0` means zero violations; fix all reported issues and re-run.

## Judgement Rules

### Subtask Boundary (`start`, `end`)

- **Split at object release and settle (last arm in contact)**: pick and place of one object belong to the **same** subtask. A subtask ends at the **exact frame where its target object is placed and stable and the last arm in contact with that object lets go** (for `open_carton`, the moment the lid lies flat on the table and the flipping arm lets go; in bimanual tasks where a second arm presses or steadies the object flat after the primary arm releases, the boundary is when that last arm lets go).
- **Subtask-specific end-rule overrides**: a subtask may define its own end rule in the task spec (written as a YAML comment directly above that subtask), which overrides the default release rule above — for example, `flip_lid` ends at the frame the flipped lid first lands on the box even if a hand still holds it. Always read the task spec comments before annotating.
- **Retraction and next-stage prep**: arm retraction after the last arm lets go belongs to subtask $k+1$, **not** subtask $k$. Conversely, if the non-contacting arm begins moving toward the next subtask's object *before* the last arm lets go of the current object, those frames still belong to subtask $k$.
- **Half-open intervals**: each row covers `[start, end)`. Sorted by `start`, rows within an episode must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal when they cover `[0, episode_length)`.
- **Unannotated / bad episodes**: corrupted or unannotated episodes may be omitted from the table rather than fabricated.

### Grasp (`grasp`) and Legacy `contact` Column

- **`contact`**: no longer annotated; leave empty. (Existing rows containing a `contact` value remain valid as long as `start <= contact <= grasp < end`.)
- **`grasp`**: record the first frame within `[start, end)` where the object is securely held and begins moving with the gripper. It is required for any subtask that grasps something, not only those with an `event: grasp` object: `distribution` reads the TCP pose at this frame both as the object's initial pose (for `objects` entries) and as the subtask's trajectory start (`start_source: grasp`; without it the start falls back to the subtask's first frame). A hold with open jaws that cradles or straddles the object counts when the object then moves with the gripper (human exemplars for `flip_lid` use this).
- **Nullable fields**: if a subtask completes without grasping (for example, only pushing or pressing), leave `grasp` empty. Never invent frame numbers.

### Per-Subtask Outcome (`success`)

- Evaluate `success` independently for **each subtask** against that subtask's physical goal before `end`:
  - `true`: the subtask completes its goal cleanly.
  - `false`: the robot fails to complete the subtask, drops/displaces the object without recovering, or leaves the stage incomplete. Record a short reason in `note`.
