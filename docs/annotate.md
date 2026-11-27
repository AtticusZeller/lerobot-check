# Agent Annotation Guide

This guide describes the visual workflow for an annotating agent producing and validating subtask annotations on a LeRobot v3.0 dataset with `lerobot-check`.

Gripper signals alone are not reliable enough to determine subtask boundaries or grasps (for example, a gripper may flip a lid with open jaws or hold a tool that blocks finger closure). Annotation is therefore **visual first**: locate each boundary coarse-to-fine across episodes with `window-sheet`, then refine every boundary across all episodes with `boundary-sheet`.

## Shared Default Paths

When `--task-spec` or `--annotations` is omitted, `lerobot-check` reads and writes the shared `xense-lerobot-viewer` directory under `<lerobot_root>` (the grandparent of `--dataset-root`, e.g. `~/.cache/huggingface/lerobot`):

- Task spec: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>.task_spec.yaml`
- Annotation CSV: `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`
- Distribution JSON: `<lerobot_root>/.xense-viewer/distribution/<org>__<name>@<revision>.json`
- Suggestions JSON: `<lerobot_root>/.xense-viewer/suggestions/<org>__<name>@<revision>.json`

Explicit `--task-spec`, `--annotations`, and `-o` arguments remain supported and override defaults. Never write rendered images, proposals, or annotation CSVs into the git repository.

## Workflow

Run commands from the repository root with `uv run`.

### 1. Inspect Dataset Metadata and Task Spec (`info`)

Check `repo_id`, `revision`, `fps`, available `cameras`, episode lengths, and default `.xense-viewer/` paths:

```bash
uv run lerobot-check info --dataset-root /path/to/dataset
```

Read the task spec YAML comments for any subtask-specific boundary overrides, and note which subtasks have an `objects` entry with `event: grasp`. Use the exact `revision` printed by `info` in the annotation CSV.

### 2. Coarse Pass and Initial Table (`window-sheet`)

Search each non-final subtask boundary separately, coarse to fine, using `lerobot-check window-sheet`:

1. **One agent per non-final boundary**: assign one annotating agent to each non-final boundary `k` (`1 .. N-1`). Each agent judges only its own boundary across all episodes.
2. **Choose the search window**: if an earlier annotation of the same task already established a time window for boundary `k`, reuse it. Otherwise, first run `window-sheet` over the full episode length at ~4 s spacing on ~8 episodes: set `HI` to at least the longest `duration_s` printed by `info` (each window is clamped to its episode's last frame) and `--samples` to about `HI / 4 + 1` (for example, with a longest episode of 106.5 s, `--episodes 0,15,30,45,60,75,90,100 --window-s 0:108 --samples 28`). Choose a window covering the observed boundary times with margin.
3. **Round 1 (`--window-s`, ~4 s spacing)**: render 8 evenly spaced frames inside the window with 8 episodes per sheet (`--per-page 8`, default). Crop `head` to the active workspace when helpful:

```bash
uv run lerobot-check window-sheet \
  --dataset-root /path/to/dataset \
  --window-s 10:42 \
  --samples 8 \
  --crop head=90,40,550,360 \
  --name b1_r1 \
  -o /tmp/lerobot-check/window-r1
```

For every cell in each episode row, answer one monotone yes/no question: **"Has subtask `k` ended at this frame?"** Judge from the **object's state** (whether the object has been placed, seated, or flipped into its target state), not gripper opening—gripper motion is not monotone due to regrasps, closing in the air, or open-jaw pushes. Record the last `"no"` frame and the first `"yes"` frame for each episode.

4. **Round 2 (`--windows`, ~0.5 s spacing)**: write a CSV with header `episode_index,lo,hi` containing each episode's `(last_no_frame, first_yes_frame)` from Round 1. If an episode row was all `"no"` (boundary occurs after the window) or all `"yes"` (boundary occurs before the window), widen its window on that side and render again before narrowing. Render Round 2 with `--windows` (adding wrist cameras if helpful):

```bash
uv run lerobot-check window-sheet \
  --dataset-root /path/to/dataset \
  --windows /tmp/lerobot-check/b1_windows.csv \
  --samples 8 \
  --cameras head left_wrist right_wrist \
  --crop head=90,40,550,360 \
  --name b1_r2 \
  -o /tmp/lerobot-check/window-r2
```

From Round 2, each boundary agent outputs a CSV with header `episode_index,end` for its boundary, using each episode's first `"yes"` frame as `end`; Step 3 refines it to the exact frame.

5. **Assemble the initial annotation table**: the coordinator combines the per-boundary `episode_index,end` CSVs into one annotation CSV per `(repo_id, revision)` (default `<lerobot_root>/.xense-viewer/subtasks/<org>__<name>@<revision>.csv`), setting `start = 0` for the first subtask, each subsequent row's `start` to the previous row's `end`, and the final subtask's `end` to `episode_length`, with the exact header:

```text
repo_id,revision,episode_index,subtask,start,end,contact,grasp,success,annotator,note
```

- Set `revision` to the exact hash from `lerobot-check info`, `annotator` to `agent`, and `success` to lowercase `true` or `false`.
- **Leave `contact` empty** (legacy column kept for schema compatibility).
- Fill `grasp` only for subtasks with an `objects` entry of `event: grasp` in the task spec; leave `grasp` empty for other subtasks. Leave `note` empty when not applicable.
- Once assembled, run Step 3 (`boundary-sheet`) to fix exact boundary frames across all episodes.

### 3. Boundary Self-Check Loop (`boundary-sheet`)

Precision comes from inspecting every non-final subtask's end frame across all episodes side by side. Run the [Boundary self-check loop](#boundary-self-check-loop) with `lerobot-check boundary-sheet` to refine every `agent` row's `end` until all tiles pass, then render the final full sheets (`--per-page 0 --columns 4`) for human review.

### 4. Validate (`validate`)

Validate the finished CSV against dataset metadata and the task spec:

```bash
uv run lerobot-check validate --dataset-root /path/to/dataset
```

Exit code `0` means zero violations; fix all reported issues and re-run.

## Boundary self-check loop

1. **Render sheets for all non-final subtasks** with `lerobot-check boundary-sheet` (the last subtask runs to the episode end and is excluded automatically). For `pack_iphone_box`-style tabletop scenes, cropping the head camera (for example `--crop head=90,40,550,360`) keeps the active workspace legible alongside the wrist cameras:

```bash
uv run lerobot-check boundary-sheet \
  --repo-id Xense/pack_iphone_box_1119 \
  --crop head=90,40,550,360 \
  -o /tmp/lerobot-check/boundary-sheets
```

2. **Judge every tile** on each `<k>_<subtask>_end_p<page>.jpg` sheet as `ok`, `early`, or `late` using the Subtask Boundary rule below and any subtask-specific override in the task spec. Wrist cameras often show jaw clearance more clearly than the head camera. Each tile is labeled `ep<idx> f<frame>`, with a coloured `H` marker when the row's `annotator` is `human`.
3. **Fix bad `agent` rows only**: for each bad tile whose row has `annotator == agent`, update that row's `end` (delay `end` if `early`, advance `end` if `late`) and the next row's `start` in the same edit so intervals stay contiguous. **Never edit `human` rows**—if a `human`-marked tile looks off, leave the CSV untouched and report it to the human reviewer instead.
4. **Re-render only the changed episodes** with `--episodes` (inclusive ranges supported) and re-check those tiles:

```bash
uv run lerobot-check boundary-sheet \
  --repo-id Xense/pack_iphone_box_1119 \
  --subtasks place_insert \
  --episodes 5,9,20-28 \
  --crop head=90,40,550,360 \
  -o /tmp/lerobot-check/boundary-sheets-recheck
```

5. **Repeat steps 2–4** until every `agent` tile passes.
6. **Validate** the updated CSV:

```bash
uv run lerobot-check validate --repo-id Xense/pack_iphone_box_1119
```

7. **Render final full sheets for human review** with all episodes per subtask on a single image (`--per-page 0 --columns 4`) so the human only has to scan the finished sheets:

```bash
uv run lerobot-check boundary-sheet \
  --repo-id Xense/pack_iphone_box_1119 \
  --crop head=90,40,550,360 \
  --per-page 0 \
  --columns 4 \
  -o /tmp/lerobot-check/boundary-sheets-final
```

## Optional Tools

Use these only when a specific tile or event is unclear (for example, an occluded tile during the self-check loop or an `event: grasp` frame):

- **Single-episode coarse grid (`render --frames ::30`)**: render a 1 s-spaced single-camera grid (`--frames ::30 --columns 6 --tile-width 240`) for an episode that does not fit the window search.
- **Fine strip around one frame (`render`)**: render a ±1 s multi-camera strip around a candidate frame (`--events 411 --window-s 1.0 --stride 3`, or an explicit half-open `--frames 381:441:2`) using `--subtask <name>` or `--cameras head left_wrist right_wrist`.
- **Human exemplars (`exemplars`)**: render human-reviewed reference strips for a boundary (`--boundary place_phone:move_lid`) or grasp (`--boundary place_phone:grasp`). Exits `0` if none exist yet. If human exemplars and task-spec comments disagree, follow the exemplars and record the discrepancy in `note`.
- **Proprioceptive hints (`propose`)**: emit heuristic `gripper_close`, `gripper_open`, `squeeze`, and `low_speed` candidates for orienting on long episodes; always confirm visually.

## Judgement Rules

### Subtask Boundary (`start`, `end`)

- **Default pick-and-place end rule (`end - 1`)**: pick and place of one object belong to the **same** subtask. A pick-and-place subtask's last frame (`end - 1`) is the first frame where the last gripper holding (or pressing/steadying) its object has opened and lifted clear of it, with its jaws in the air just above the settled object. Too early: the gripper still touches or presses the object. Too late: the arm has already retracted toward the next object.
- **Subtask-specific end-rule overrides**: a subtask may define its own end rule in the task spec (written as a YAML comment directly above that subtask), which overrides the default release rule above—for example, `flip_lid` ends at the frame the flipped lid first lands on the box even if a hand still holds it. Always read the task spec comments before annotating.
- **Retraction and next-stage prep**: further arm retraction after the gripper has lifted clear and the approach toward the next object belong to subtask $k+1$, **not** subtask $k$. Conversely, if the non-contacting arm begins moving toward the next subtask's object *before* the last gripper lifts clear of the current object, those frames still belong to subtask $k$.
- **Half-open intervals**: each row covers `[start, end)`. Sorted by `start`, rows within an episode must satisfy `start_0 = 0`, `start_{k+1} = end_k`, and `end_last = episode_length`. Prompts switch at `end_k`.
- **Order and supplement episodes**: subtask names must come from the task spec and appear in task spec order as a contiguous subsequence. Single-stage or partial-stage supplement episodes are legal when they cover `[0, episode_length)`.
- **Unannotated / bad episodes**: corrupted or unannotated episodes may be omitted from the table rather than fabricated.

### Grasp (`grasp`) and Legacy `contact` Column

- **`contact`**: no longer annotated; leave empty. (Existing rows containing a `contact` value remain valid as long as `start <= contact <= grasp < end`.)
- **`grasp`**: required only for subtasks that have an `objects` entry with `event: grasp` in the task spec (needed for object initial poses in `distribution`). For other subtasks, leave `grasp` empty (the trajectory start then falls back to the subtask's first frame). Where required, record the first frame within `[start, end)` where the gripper jaws have closed on the object, even when the lift starts some frames later; after a failed attempt, use the closure of the attempt that carries the object. When the jaws never close on the object, such as an open-jaw hold that cradles or straddles it, record the frame where the hold is formed and the object then moves with the gripper. Never invent frame numbers.

### Per-Subtask Outcome (`success`)

- Evaluate `success` independently for **each subtask** against that subtask's physical goal before `end`:
  - `true`: the subtask completes its goal cleanly.
  - `false`: the robot fails to complete the subtask, drops/displaces the object without recovering, or leaves the stage incomplete. Record a short reason in `note`.
