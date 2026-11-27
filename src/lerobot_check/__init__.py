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
    from lerobot_check.tcp_retarget import (
        RetargetTcpResult,
        VerifyTcpResult,
        retarget_calib_payload,
        retarget_tcp_dataset,
        verify_tcp_retarget,
    )

_LAZY_EXPORTS = {
    "RelabelResult": ".relabel",
    "RelabelValidationError": ".relabel",
    "VerifyResult": ".relabel",
    "relabel_dataset": ".relabel",
    "verify_relabel": ".relabel",
    "RetargetTcpResult": ".tcp_retarget",
    "VerifyTcpResult": ".tcp_retarget",
    "retarget_calib_payload": ".tcp_retarget",
    "retarget_tcp_dataset": ".tcp_retarget",
    "verify_tcp_retarget": ".tcp_retarget",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(submodule, __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ANNOTATION_COLUMNS",
    "AnnotationRow",
    "CoverageSpec",
    "ObjectSpec",
    "RelabelResult",
    "RelabelValidationError",
    "RetargetTcpResult",
    "SubtaskSpec",
    "TaskSpec",
    "ValidationViolation",
    "VerifyResult",
    "VerifyTcpResult",
    "load_annotations",
    "load_task_spec",
    "relabel_dataset",
    "retarget_calib_payload",
    "retarget_tcp_dataset",
    "save_annotations",
    "save_task_spec",
    "validate_annotations",
    "verify_relabel",
    "verify_tcp_retarget",
]

__version__ = "0.1.0"
