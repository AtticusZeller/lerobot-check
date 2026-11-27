# AGENTS.md

Repository guidelines for coding and annotating agents working in `lerobot-check`.

## Environment & Tooling

- Python >= 3.12 managed by `uv` (`.python-version` is pinned to `3.12`).
- Runtime depends on a pinned PyPI release of `lerobot` (`lerobot==0.5.1`) that reads LeRobot v3.0 datasets. Do **not** depend on `lerobot-xense` or any robot hardware SDK.
- Install and sync all dependencies (including dev tools):

```bash
uv sync --all-groups
```

- Run tests, linting, formatting checks, and type checks via `uv run`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
```

Enable the pre-commit hook for new comments and docstrings with `uvx pre-commit install`.

## Public Repository Data Policy

This is a **public** repository. Never commit, stage, or add:

- LeRobot datasets (parquet files, MP4 videos, metadata caches)
- Subtask annotation CSV tables
- Candidate event proposal CSVs
- Rendered contact-sheet PNGs
- Generated analysis HTML reports

CLI commands take explicit input/output paths. Write all generated artifacts outside the repository (e.g. `/tmp/...`) or under the gitignored default output directories (`outputs/`, `annotations/`, `reports/`, `renders/`, `datasets/`).

## Code Conventions

- Keep module boundaries clean:
  - `schema.py`: dataclasses and CSV/YAML I/O for annotation tables and task specs
  - `validate.py`: annotation contract validation
  - `dataset.py`: LeRobot v3.0 metadata/state loading, shared `.xense-viewer/` default paths, and 6D rotation helpers
  - `propose.py`: proprioceptive gripper event and TCP low-speed proposal
  - `render.py`: low-memory PyAV frame decoding, multi-camera PNG tiling, and `boundary-sheet` / `window-sheet` JPEG contact sheets
  - `relabel.py`: N-stage `task_index` relabeling and verification
  - `tcp_retarget.py`: rigid-body SE(3) TCP retargeting, verification, and calibration payload retargeting across tool snapshots
  - `analysis.py`: `distribution.json` computation (frame shares, object poses/coverage, resampled trajectories, subtask-arm metrics)
  - `calib.py`: head-camera calibration frame selection (`calib-frames` with `--max-gripper`, `--max-speed`, `--no-speed-filter`, `--append`, `--arms`), joint extrinsic+tool-offset solver (`calibrate`), and projection/back-projection geometry
  - `suggest.py`: target polygon back-projection, collection placement sampling, and three-tier test point generation
  - `masks.py`: object mask template geometry and projected overlay bundles
  - `mask_cli.py`: mask template extraction, rejection, and overlay commands
  - `collection_planner.py`: placement candidates, overlap checks, and collection round selection
  - `collection_plan_cli.py`: collection plan generation, editing, and progress commands
  - `collection_progress.py`: collection plan schema, locking, persistence, and progress merging
  - `segmentation.py`: SAM model loading, mask prediction, and exterior contours
  - `alignment.py`: live mask overlap scoring and target-specific thresholds
  - `json_io.py`: shared JSON object reading
  - `cli.py`: `lerobot-check` CLI entry point
- Express intent through names, explicit types, and small, single-responsibility functions instead of new comments or docstrings.
- Use `KEEP:` comments only for unavoidable external constraints that code, types, and tests cannot express.
- Commit with Conventional Commits (`type(scope): subject`).
