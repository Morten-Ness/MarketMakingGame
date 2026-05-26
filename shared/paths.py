from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate

