from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.paths import resolve_repo_path


DIRECTIONS_ROOT = "games/research_papers/directions"
DEFAULT_RESEARCH_SUBJECT = "embeddings"
_SUBJECT_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class ResearchDirectionError(ValueError):
    pass


@dataclass(frozen=True)
class ResearchDirection:
    subject: str
    name: str
    root: str
    seed_paper_id: str
    seed_query: str

    @property
    def corpus_path(self) -> str:
        return f"{self.root}/corpus.json"

    @property
    def pdf_dir(self) -> str:
        return f"{self.root}/pdfs"

    @property
    def raw_text_dir(self) -> str:
        return f"{self.root}/raw_text"

    @property
    def game_log_path(self) -> str:
        return f"{self.root}/logs/prediction_games.jsonl"

    @classmethod
    def load(
        cls,
        subject: str = DEFAULT_RESEARCH_SUBJECT,
        *,
        directions_root: str = DIRECTIONS_ROOT,
    ) -> "ResearchDirection":
        safe_subject = validate_research_subject(subject)
        root = _join_path_text(directions_root, safe_subject)
        root_path = resolve_repo_path(root)
        if not root_path.exists():
            raise ResearchDirectionError(
                f"Research direction {safe_subject!r} does not exist at {root_path}."
            )

        config_path = root_path / "direction.json"
        if not config_path.exists():
            raise ResearchDirectionError(
                f"Research direction {safe_subject!r} is missing {config_path}."
            )

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchDirectionError(
                f"Could not read research direction config at {config_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ResearchDirectionError(
                f"Research direction config at {config_path} must be a JSON object."
            )

        name = _required_text(payload, "name", config_path)
        if name != safe_subject:
            raise ResearchDirectionError(
                f"Research direction config name {name!r} does not match "
                f"subject {safe_subject!r}."
            )

        return cls(
            subject=safe_subject,
            name=name,
            root=root,
            seed_paper_id=_required_text(payload, "seedPaperId", config_path),
            seed_query=_required_text(payload, "seedQuery", config_path),
        )


def validate_research_subject(subject: str) -> str:
    safe_subject = subject.strip()
    if not _SUBJECT_PATTERN.fullmatch(safe_subject):
        raise ResearchDirectionError(
            "Research subject names must contain only lowercase letters, numbers, "
            "underscores, and hyphens."
        )
    return safe_subject


def _join_path_text(root: str, child: str) -> str:
    return str(Path(root) / child)


def _required_text(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResearchDirectionError(
            f"Research direction config at {path} must define {key!r}."
        )
    return value.strip()
