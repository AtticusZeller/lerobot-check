"""Analysis and annotation tools for LeRobot v3.0 datasets."""

from lerobot_check.relabel import (
    RelabelResult,
    RelabelValidationError,
    VerifyResult,
    relabel_dataset,
    verify_relabel,
)
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
    "RelabelResult",
    "RelabelValidationError",
    "SubtaskSpec",
    "TaskSpec",
    "ValidationViolation",
    "VerifyResult",
    "load_annotations",
    "load_task_spec",
    "relabel_dataset",
    "save_annotations",
    "save_task_spec",
    "validate_annotations",
    "verify_relabel",
]

__version__ = "0.1.0"
