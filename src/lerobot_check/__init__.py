"""Analysis and annotation tools for LeRobot v3.0 datasets."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from lerobot_check.relabel import (
        RelabelResult,
        RelabelValidationError,
        VerifyResult,
        relabel_dataset,
        verify_relabel,
    )

_RELABEL_EXPORTS = {
    "RelabelResult",
    "RelabelValidationError",
    "VerifyResult",
    "relabel_dataset",
    "verify_relabel",
}


def __getattr__(name: str) -> Any:
    if name not in _RELABEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".relabel", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _RELABEL_EXPORTS)


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
