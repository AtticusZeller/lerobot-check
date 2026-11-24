"""Analysis and annotation tools for LeRobot v3.0 datasets."""

from lerobot_check.schema import (
    ANNOTATION_COLUMNS,
    AnnotationRow,
    CoverageSpec,
    ObjectSpec,
    SubtaskSpec,
    TaskSpec,
    load_annotations,
    load_task_spec,
    save_annotations,
    save_task_spec,
)
from lerobot_check.validate import ValidationViolation, validate_annotations

__all__ = [
    "ANNOTATION_COLUMNS",
    "AnnotationRow",
    "CoverageSpec",
    "ObjectSpec",
    "SubtaskSpec",
    "TaskSpec",
    "ValidationViolation",
    "load_annotations",
    "load_task_spec",
    "save_annotations",
    "save_task_spec",
    "validate_annotations",
]

__version__ = "0.1.0"
