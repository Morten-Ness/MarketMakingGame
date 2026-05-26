from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import resolve_repo_path


class JsonlLogger:
    def __init__(self, path: str | Path | None) -> None:
        self._path = resolve_repo_path(path) if path else None

    def write(self, payload: dict[str, Any]) -> None:
        if not self._path:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

