import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, content: bytes) -> dict[str, Any]:
    try:
        result = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
