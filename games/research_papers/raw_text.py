from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from shared.paths import resolve_repo_path

from .models import Corpus


class RawTextExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawTextExtractionSummary:
    raw_text_dir: Path
    extracted_count: int
    skipped_count: int


class RawTextExtractor:
    def __init__(
        self,
        raw_text_dir: str,
        *,
        extract_text_func: Callable[[Path], str] | None = None,
    ) -> None:
        self._raw_text_dir = resolve_repo_path(raw_text_dir)
        self._extract_text_func = extract_text_func or extract_pdf_text

    @property
    def raw_text_dir(self) -> Path:
        return self._raw_text_dir

    def extract_for_corpus(self, corpus: Corpus) -> RawTextExtractionSummary:
        extracted_count = 0
        skipped_count = 0
        self._raw_text_dir.mkdir(parents=True, exist_ok=True)

        for row in corpus.papers:
            if not row.paper.pdf_path:
                continue

            pdf_path = resolve_repo_path(row.paper.pdf_path)
            if not pdf_path.exists():
                raise RawTextExtractionError(f"PDF does not exist: {row.paper.pdf_path}")

            text_path = raw_text_path_for_pdf(pdf_path, self._raw_text_dir)
            if _raw_text_is_current(text_path, pdf_path):
                skipped_count += 1
                continue

            raw_text = self._extract_text_func(pdf_path)
            text_path.write_text(_normalize_raw_text(raw_text), encoding="utf-8")
            extracted_count += 1

        return RawTextExtractionSummary(
            raw_text_dir=self._raw_text_dir,
            extracted_count=extracted_count,
            skipped_count=skipped_count,
        )


def raw_text_path_for_pdf(pdf_path: Path, raw_text_dir: Path) -> Path:
    return raw_text_dir / f"{pdf_path.stem}.txt"


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RawTextExtractionError(
            "pypdf is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
        page_text = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise RawTextExtractionError(f"Could not extract text from {pdf_path}: {exc}") from exc

    return "\n\n".join(text.strip() for text in page_text if text.strip())


def _raw_text_is_current(text_path: Path, pdf_path: Path) -> bool:
    return text_path.exists() and text_path.stat().st_mtime >= pdf_path.stat().st_mtime


def _normalize_raw_text(raw_text: str) -> str:
    normalized = raw_text.strip()
    if not normalized:
        return ""
    return f"{normalized}\n"
