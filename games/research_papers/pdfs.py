from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from shared.paths import resolve_repo_path

from .models import Paper


ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"
PDF_SOURCE_ARXIV = "arxiv"
PDF_SOURCE_OPEN_ACCESS = "semantic_scholar_open_access"


class PdfDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PdfCandidate:
    url: str
    source: str


@dataclass(frozen=True)
class PdfDownloadResult:
    url: str
    source: str
    path: Path
    downloaded_at_utc: str
    byte_size: int
    sha256: str


class PdfDownloader:
    def __init__(
        self,
        pdf_dir: str,
        *,
        timeout_seconds: int = 45,
        prefer_arxiv: bool = True,
    ) -> None:
        self._pdf_dir = resolve_repo_path(pdf_dir)
        self._timeout_seconds = max(1, timeout_seconds)
        self._prefer_arxiv = prefer_arxiv

    @property
    def pdf_dir(self) -> Path:
        return self._pdf_dir

    def download_for_paper(self, paper: Paper) -> Paper | None:
        for candidate in pdf_candidates_for_paper(
            paper,
            prefer_arxiv=self._prefer_arxiv,
        ):
            try:
                result = self.download(candidate, title=paper.title)
            except PdfDownloadError:
                continue
            return paper.with_pdf_metadata(
                pdf_url=result.url,
                pdf_source=result.source,
                pdf_path=_display_path(result.path),
                pdf_downloaded_at_utc=result.downloaded_at_utc,
                pdf_byte_size=result.byte_size,
                pdf_sha256=result.sha256,
            )
        return None

    def download(self, candidate: PdfCandidate, *, title: str) -> PdfDownloadResult:
        data, content_type = self._fetch_pdf(candidate.url)
        if not is_pdf_response(data, content_type):
            raise PdfDownloadError(f"Downloaded content was not a PDF: {candidate.url}")

        self._pdf_dir.mkdir(parents=True, exist_ok=True)
        path = unique_pdf_path(self._pdf_dir, title)
        path.write_bytes(data)
        return PdfDownloadResult(
            url=candidate.url,
            source=candidate.source,
            path=path,
            downloaded_at_utc=_utc_now(),
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def _fetch_pdf(self, url: str) -> tuple[bytes, str]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/pdf,*/*",
                "User-Agent": "MarketMakingGame/0.1",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                return response.read(), content_type
        except urllib.error.URLError as exc:
            raise PdfDownloadError(f"Could not download PDF: {url}: {exc}") from exc


def pdf_candidates_for_paper(
    paper: Paper,
    *,
    prefer_arxiv: bool = True,
) -> list[PdfCandidate]:
    candidates: list[PdfCandidate] = []
    arxiv_url = arxiv_pdf_url(paper)
    if arxiv_url:
        candidates.append(PdfCandidate(arxiv_url, PDF_SOURCE_ARXIV))
    if paper.open_access_pdf_url:
        candidates.append(
            PdfCandidate(paper.open_access_pdf_url, PDF_SOURCE_OPEN_ACCESS)
        )
    if prefer_arxiv:
        return candidates
    return [
        candidate for candidate in candidates if candidate.source != PDF_SOURCE_ARXIV
    ] + [candidate for candidate in candidates if candidate.source == PDF_SOURCE_ARXIV]


def arxiv_pdf_url(paper: Paper) -> str | None:
    arxiv_id = arxiv_id_from_paper(paper)
    if not arxiv_id:
        return None
    return f"{ARXIV_PDF_BASE_URL}/{urllib.parse.quote(arxiv_id)}"


def arxiv_id_from_paper(paper: Paper) -> str | None:
    for key, value in paper.external_ids.items():
        if key.lower() != "arxiv" or not isinstance(value, str):
            continue
        return normalize_arxiv_id(value)
    return None


def normalize_arxiv_id(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None

    prefixes = (
        "arxiv:",
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    )
    lower = cleaned.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.removesuffix(".pdf").strip()
    return cleaned or None


def unique_pdf_path(pdf_dir: Path, title: str, *, max_stem_length: int = 120) -> Path:
    stem = sanitize_filename(title, max_length=max_stem_length)
    path = pdf_dir / f"{stem}.pdf"
    if not path.exists():
        return path

    for suffix in range(2, 10_000):
        suffix_text = f"-{suffix}"
        trimmed_stem = stem[: max_stem_length - len(suffix_text)].rstrip(" .-_")
        candidate = pdf_dir / f"{trimmed_stem}{suffix_text}.pdf"
        if not candidate.exists():
            return candidate
    raise PdfDownloadError(f"Could not find an unused PDF filename for {title!r}.")


def sanitize_filename(title: str, *, max_length: int = 120) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = "paper"
    cleaned = cleaned[:max_length].rstrip(" .-_")
    return cleaned or "paper"


def is_pdf_response(data: bytes, content_type: str | None = None) -> bool:
    del content_type
    return data[:1024].lstrip().startswith(b"%PDF")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(resolve_repo_path(".")))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
