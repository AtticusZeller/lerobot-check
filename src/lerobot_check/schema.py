"""Data models and CSV/YAML I/O for task specs and subtask annotation tables."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Annotator = Literal["agent", "human", "converted"]
ObjectEvent = Literal["grasp", "release"]
ArmSide = Literal["left", "right"]

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "repo_id",
    "revision",
    "episode_index",
    "subtask",
    "start",
    "end",
    "contact",
    "grasp",
    "success",
    "annotator",
    "note",
)

VALID_ANNOTATORS: frozenset[str] = frozenset({"agent", "human", "converted"})
VALID_EVENTS: frozenset[str] = frozenset({"grasp", "release"})
VALID_ARMS: frozenset[str] = frozenset({"left", "right"})


@dataclass(frozen=True)
class SubtaskSpec:
    """Definition of one ordered subtask in a task specification.

    Attributes:
        name: Short identifier referenced by annotation tables and object specs.
        prompt: Verbatim language instruction used by training and robot clients.
        cameras: Optional ordered camera feature keys (or short names) preferred when
            visually inspecting or annotating this subtask. Empty when unspecified.
    """

    name: str
    prompt: str
    cameras: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectSpec:
    """Object pose sampling specification used by downstream distribution analysis.

    Attributes:
        name: Unique object name within the task spec.
        subtask: Subtask name during which the object pose is sampled.
        event: Event frame selector: ``"grasp"`` uses the ``grasp`` frame and
            ``"release"`` uses ``end - 1``.
        arm: Robot arm whose TCP pose is sampled (``"left"`` or ``"right"``).
        yaw_symmetry_deg: Rotational symmetry period in degrees (e.g. 360, 180, 90)
            used to fold yaw angles.
    """

    name: str
    subtask: str
    event: ObjectEvent
    arm: ArmSide
    yaw_symmetry_deg: float


@dataclass(frozen=True)
class CoverageSpec:
    """Grid binning parameters for workspace coverage analysis.

    Attributes:
        xy_bin_m: Side length of each XY grid cell in meters (default 5 cm).
        yaw_bin_deg: Width of each yaw bin in degrees (default 30 deg).
        sparse_threshold: Minimum sample count below which a bin is considered sparse.
    """

    xy_bin_m: float = 0.05
    yaw_bin_deg: float = 30.0
    sparse_threshold: int = 3


@dataclass(frozen=True)
class TaskSpec:
    """Task specification defining ordered subtasks, objects, and analysis parameters.

    Attributes:
        name: Human-readable task name.
        subtasks: Ordered tuple of subtask definitions.
        objects: Tuple of object pose definitions referencing ``subtasks``.
        fine_window_s: Half-window duration delta (seconds) for fine-contact analysis.
        coverage: Workspace coverage grid parameters.
    """

    name: str
    subtasks: tuple[SubtaskSpec, ...]
    objects: tuple[ObjectSpec, ...] = ()
    fine_window_s: float = 1.0
    coverage: CoverageSpec = field(default_factory=CoverageSpec)

    @property
    def subtask_names(self) -> tuple[str, ...]:
        """Return ordered subtask names."""
        return tuple(item.name for item in self.subtasks)

    @property
    def subtask_index(self) -> dict[str, int]:
        """Map each subtask name to its 0-based order index."""
        return {item.name: idx for idx, item in enumerate(self.subtasks)}

    def prompt_for(self, subtask: str) -> str:
        """Return the verbatim prompt string for ``subtask``.

        Args:
            subtask: Name of the subtask to look up.

        Returns:
            The prompt string configured for ``subtask``.

        Raises:
            KeyError: If ``subtask`` is not present in this task spec.
        """
        for item in self.subtasks:
            if item.name == subtask:
                return item.prompt
        raise KeyError(f"Unknown subtask: {subtask!r}")

    def cameras_for(self, subtask: str) -> tuple[str, ...]:
        """Return configured camera keys for ``subtask`` (empty tuple if unspecified).

        Args:
            subtask: Name of the subtask to look up.

        Returns:
            Ordered tuple of camera feature keys for ``subtask``.

        Raises:
            KeyError: If ``subtask`` is not present in this task spec.
        """
        for item in self.subtasks:
            if item.name == subtask:
                return item.cameras
        raise KeyError(f"Unknown subtask: {subtask!r}")


@dataclass(frozen=True)
class AnnotationRow:
    """Single subtask segment annotation within one episode.

    Frame intervals are half-open ``[start, end)``. Optional ``contact`` and
    ``grasp`` frames, when present, must satisfy ``start <= contact <= grasp < end``.

    Attributes:
        repo_id: Hugging Face dataset repository ID.
        revision: Dataset revision (commit hash or branch tag).
        episode_index: Zero-based episode index in the dataset.
        subtask: Subtask name from the task spec.
        start: Inclusive start frame index within the episode.
        end: Exclusive end frame index within the episode.
        contact: Optional frame index where the gripper contacts the object.
        grasp: Optional frame index where the grasp completes.
        success: Whether the subtask succeeded.
        annotator: Annotation source (``"agent"``, ``"human"``, or ``"converted"``).
        note: Optional free-form note.
    """

    repo_id: str
    revision: str
    episode_index: int
    subtask: str
    start: int
    end: int
    contact: int | None
    grasp: int | None
    success: bool
    annotator: Annotator
    note: str = ""


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _require_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"{field_name} must be a positive number, got {value!r}")
    return float(value)


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def parse_task_spec(raw: Any) -> TaskSpec:
    """Validate and convert a raw YAML mapping into a ``TaskSpec``.

    Args:
        raw: Parsed YAML object (expected to be a dict).

    Returns:
        Validated ``TaskSpec`` instance.

    Raises:
        ValueError: If any required field is missing or violates the contract.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Task spec must be a YAML mapping, got {type(raw).__name__}")

    name = str(raw.get("name", "")).strip()

    raw_subtasks = raw.get("subtasks")
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ValueError("Task spec 'subtasks' must be a non-empty list")

    subtasks: list[SubtaskSpec] = []
    seen_subtasks: set[str] = set()
    for idx, entry in enumerate(raw_subtasks):
        if not isinstance(entry, dict):
            raise ValueError(f"subtasks[{idx}] must be a mapping")
        subtask_name = _require_non_empty_str(entry.get("name"), f"subtasks[{idx}].name")
        prompt = _require_non_empty_str(entry.get("prompt"), f"subtasks[{idx}].prompt")
        raw_cameras = entry.get("cameras")
        cameras: list[str] = []
        if raw_cameras is not None:
            if not isinstance(raw_cameras, list):
                raise ValueError(
                    f"subtasks[{idx}].cameras must be a list of camera keys, "
                    f"got {type(raw_cameras).__name__}"
                )
            seen_cams: set[str] = set()
            for cam_idx, cam_val in enumerate(raw_cameras):
                cam_key = _require_non_empty_str(
                    cam_val, f"subtasks[{idx}].cameras[{cam_idx}]"
                ).strip()
                if cam_key in seen_cams:
                    raise ValueError(
                        f"subtasks[{idx}].cameras contains duplicate camera {cam_key!r}"
                    )
                seen_cams.add(cam_key)
                cameras.append(cam_key)
        if subtask_name in seen_subtasks:
            raise ValueError(f"Duplicate subtask name: {subtask_name!r}")
        seen_subtasks.add(subtask_name)
        subtasks.append(SubtaskSpec(name=subtask_name, prompt=prompt, cameras=tuple(cameras)))

    raw_objects = raw.get("objects", [])
    if raw_objects is None:
        raw_objects = []
    if not isinstance(raw_objects, list):
        raise ValueError("Task spec 'objects' must be a list")

    objects: list[ObjectSpec] = []
    seen_objects: set[str] = set()
    for idx, entry in enumerate(raw_objects):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{idx}] must be a mapping")
        obj_name = _require_non_empty_str(entry.get("name"), f"objects[{idx}].name")
        if obj_name in seen_objects:
            raise ValueError(f"Duplicate object name: {obj_name!r}")
        seen_objects.add(obj_name)

        obj_subtask = _require_non_empty_str(entry.get("subtask"), f"objects[{idx}].subtask")
        if obj_subtask not in seen_subtasks:
            raise ValueError(f"objects[{idx}].subtask references unknown subtask {obj_subtask!r}")

        event = entry.get("event")
        if event not in VALID_EVENTS:
            raise ValueError(
                f"objects[{idx}].event must be one of {sorted(VALID_EVENTS)}, got {event!r}"
            )

        arm = entry.get("arm")
        if arm not in VALID_ARMS:
            raise ValueError(f"objects[{idx}].arm must be one of {sorted(VALID_ARMS)}, got {arm!r}")

        yaw_symmetry_deg = _require_positive_float(
            entry.get("yaw_symmetry_deg"), f"objects[{idx}].yaw_symmetry_deg"
        )
        if yaw_symmetry_deg > 360.0:
            raise ValueError(
                f"objects[{idx}].yaw_symmetry_deg must be <= 360, got {yaw_symmetry_deg}"
            )

        objects.append(
            ObjectSpec(
                name=obj_name,
                subtask=obj_subtask,
                event=event,
                arm=arm,
                yaw_symmetry_deg=yaw_symmetry_deg,
            )
        )

    fine_window_s = _require_positive_float(raw.get("fine_window_s", 1.0), "fine_window_s")

    raw_coverage = raw.get("coverage") or {}
    if not isinstance(raw_coverage, dict):
        raise ValueError("Task spec 'coverage' must be a mapping")

    coverage = CoverageSpec(
        xy_bin_m=_require_positive_float(raw_coverage.get("xy_bin_m", 0.05), "coverage.xy_bin_m"),
        yaw_bin_deg=_require_positive_float(
            raw_coverage.get("yaw_bin_deg", 30.0), "coverage.yaw_bin_deg"
        ),
        sparse_threshold=_require_positive_int(
            raw_coverage.get("sparse_threshold", 3), "coverage.sparse_threshold"
        ),
    )

    return TaskSpec(
        name=name,
        subtasks=tuple(subtasks),
        objects=tuple(objects),
        fine_window_s=fine_window_s,
        coverage=coverage,
    )


def load_task_spec(path: Path) -> TaskSpec:
    """Load and validate a YAML task spec file.

    Args:
        path: Path to the YAML task spec file.

    Returns:
        Parsed ``TaskSpec`` instance.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return parse_task_spec(raw)


def save_task_spec(spec: TaskSpec, path: Path) -> None:
    """Write a ``TaskSpec`` to YAML format.

    Args:
        spec: Task specification to serialize.
        path: Destination YAML path.
    """
    subtask_payloads: list[dict[str, Any]] = []
    for s in spec.subtasks:
        entry: dict[str, Any] = {"name": s.name, "prompt": s.prompt}
        if s.cameras:
            entry["cameras"] = list(s.cameras)
        subtask_payloads.append(entry)

    payload: dict[str, Any] = {
        "name": spec.name,
        "subtasks": subtask_payloads,
        "objects": [
            {
                "name": o.name,
                "subtask": o.subtask,
                "event": o.event,
                "arm": o.arm,
                "yaw_symmetry_deg": o.yaw_symmetry_deg,
            }
            for o in spec.objects
        ],
        "fine_window_s": spec.fine_window_s,
        "coverage": {
            "xy_bin_m": spec.coverage.xy_bin_m,
            "yaw_bin_deg": spec.coverage.yaw_bin_deg,
            "sparse_threshold": spec.coverage.sparse_threshold,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _parse_int(value: str | None, field_name: str, line_no: int) -> int:
    text = "" if value is None else value.strip()
    if not text:
        raise ValueError(f"Line {line_no}: missing integer value for '{field_name}'")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: invalid integer for '{field_name}': {value!r}") from exc


def _parse_optional_int(value: str | None, field_name: str, line_no: int) -> int | None:
    text = "" if value is None else value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: invalid integer for '{field_name}': {value!r}") from exc


def _parse_bool(value: str | None, field_name: str, line_no: int) -> bool:
    text = "" if value is None else value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Line {line_no}: '{field_name}' must be 'true' or 'false', got {value!r}")


def _parse_annotator(value: str | None, line_no: int) -> Annotator:
    text = "" if value is None else value.strip()
    if text == "agent":
        return "agent"
    if text == "human":
        return "human"
    if text == "converted":
        return "converted"
    raise ValueError(
        f"Line {line_no}: 'annotator' must be one of {sorted(VALID_ANNOTATORS)}, got {value!r}"
    )


def load_annotations(path: Path) -> list[AnnotationRow]:
    """Read an annotation table CSV file into ``AnnotationRow`` objects.

    Args:
        path: Path to the annotation CSV file.

    Returns:
        List of parsed ``AnnotationRow`` records in file order.

    Raises:
        ValueError: If required columns are missing or any row cannot be parsed.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Annotation CSV {path} is empty (no header row)")

        missing = [col for col in ANNOTATION_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Annotation CSV {path} is missing required columns: {missing}")

        rows: list[AnnotationRow] = []
        for line_no, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"Line {line_no}: unexpected extra CSV columns: {raw[None]}")
            rows.append(
                AnnotationRow(
                    repo_id=(raw["repo_id"] or "").strip(),
                    revision=(raw["revision"] or "").strip(),
                    episode_index=_parse_int(raw["episode_index"], "episode_index", line_no),
                    subtask=(raw["subtask"] or "").strip(),
                    start=_parse_int(raw["start"], "start", line_no),
                    end=_parse_int(raw["end"], "end", line_no),
                    contact=_parse_optional_int(raw["contact"], "contact", line_no),
                    grasp=_parse_optional_int(raw["grasp"], "grasp", line_no),
                    success=_parse_bool(raw["success"], "success", line_no),
                    annotator=_parse_annotator(raw["annotator"], line_no),
                    note=(raw["note"] or "").strip(),
                )
            )
    return rows


def save_annotations(rows: Sequence[AnnotationRow], path: Path) -> None:
    """Write ``AnnotationRow`` records to a CSV file following the contract schema.

    Args:
        rows: Ordered annotation rows to write.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "repo_id": row.repo_id,
                    "revision": row.revision,
                    "episode_index": row.episode_index,
                    "subtask": row.subtask,
                    "start": row.start,
                    "end": row.end,
                    "contact": "" if row.contact is None else row.contact,
                    "grasp": "" if row.grasp is None else row.grasp,
                    "success": "true" if row.success else "false",
                    "annotator": row.annotator,
                    "note": row.note,
                }
            )
