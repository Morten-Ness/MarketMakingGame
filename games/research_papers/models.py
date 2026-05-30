from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    corpus_id: int | None = None
    external_ids: dict[str, Any] = field(default_factory=dict)
    url: str | None = None
    abstract: str | None = None
    venue: str | None = None
    year: int | None = None
    reference_count: int | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    is_open_access: bool | None = None
    open_access_pdf_url: str | None = None
    fields_of_study: list[str] = field(default_factory=list)
    publication_types: list[str] = field(default_factory=list)
    publication_date: str | None = None
    authors: list[str] = field(default_factory=list)
    tldr: str | None = None

    @classmethod
    def from_api_payload(cls, payload: dict[str, Any]) -> "Paper | None":
        paper_id = _as_non_empty_string(payload.get("paperId"))
        title = _as_non_empty_string(payload.get("title"))
        if not paper_id or not title:
            return None

        open_access_pdf = payload.get("openAccessPdf")
        tldr = payload.get("tldr")
        return cls(
            paper_id=paper_id,
            title=title,
            corpus_id=_as_int(payload.get("corpusId")),
            external_ids=_as_dict(payload.get("externalIds")),
            url=_as_non_empty_string(payload.get("url")),
            abstract=_as_non_empty_string(payload.get("abstract")),
            venue=_as_non_empty_string(payload.get("venue")),
            year=_as_int(payload.get("year")),
            reference_count=_as_int(payload.get("referenceCount")),
            citation_count=_as_int(payload.get("citationCount")),
            influential_citation_count=_as_int(payload.get("influentialCitationCount")),
            is_open_access=_as_bool_or_none(payload.get("isOpenAccess")),
            open_access_pdf_url=(
                _as_non_empty_string(open_access_pdf.get("url"))
                if isinstance(open_access_pdf, dict)
                else None
            ),
            fields_of_study=_string_list(payload.get("fieldsOfStudy")),
            publication_types=_string_list(payload.get("publicationTypes")),
            publication_date=_as_non_empty_string(payload.get("publicationDate")),
            authors=_authors_from_payload(payload.get("authors")),
            tldr=(
                _as_non_empty_string(tldr.get("text"))
                if isinstance(tldr, dict)
                else None
            ),
        )

    @classmethod
    def from_corpus_row(cls, row: dict[str, Any]) -> "Paper | None":
        paper_id = _as_non_empty_string(row.get("paperId"))
        title = _as_non_empty_string(row.get("title"))
        if not paper_id or not title:
            return None

        return cls(
            paper_id=paper_id,
            title=title,
            corpus_id=_as_int(row.get("corpusId")),
            external_ids=_as_dict(row.get("externalIds")),
            url=_as_non_empty_string(row.get("url")),
            abstract=_as_non_empty_string(row.get("abstract")),
            venue=_as_non_empty_string(row.get("venue")),
            year=_as_int(row.get("year")),
            reference_count=_as_int(row.get("referenceCount")),
            citation_count=_as_int(row.get("citationCount")),
            influential_citation_count=_as_int(row.get("influentialCitationCount")),
            is_open_access=_as_bool_or_none(row.get("isOpenAccess")),
            open_access_pdf_url=_as_non_empty_string(row.get("openAccessPdfUrl")),
            fields_of_study=_string_list(row.get("fieldsOfStudy")),
            publication_types=_string_list(row.get("publicationTypes")),
            publication_date=_as_non_empty_string(row.get("publicationDate")),
            authors=_string_list(row.get("authors")),
            tldr=_as_non_empty_string(row.get("tldr")),
        )

    def as_corpus_dict(self) -> dict[str, Any]:
        return {
            "paperId": self.paper_id,
            "corpusId": self.corpus_id,
            "externalIds": self.external_ids,
            "url": self.url,
            "title": self.title,
            "abstract": self.abstract,
            "venue": self.venue,
            "year": self.year,
            "referenceCount": self.reference_count,
            "citationCount": self.citation_count,
            "influentialCitationCount": self.influential_citation_count,
            "isOpenAccess": self.is_open_access,
            "openAccessPdfUrl": self.open_access_pdf_url,
            "fieldsOfStudy": self.fields_of_study,
            "publicationTypes": self.publication_types,
            "publicationDate": self.publication_date,
            "authors": self.authors,
            "tldr": self.tldr,
        }


@dataclass(frozen=True)
class CorpusPaper:
    rank: int
    added_at_utc: str
    added_reason: str
    paper: Paper
    recommendation_source_paper_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CorpusPaper | None":
        paper = Paper.from_corpus_row(row)
        if paper is None:
            return None
        rank = _as_int(row.get("rank"))
        if rank is None:
            return None
        return cls(
            rank=rank,
            added_at_utc=_as_non_empty_string(row.get("addedAtUtc")) or "",
            added_reason=_as_non_empty_string(row.get("addedReason")) or "unknown",
            paper=paper,
            recommendation_source_paper_ids=_string_list(
                row.get("recommendationSourcePaperIds")
            ),
        )

    def as_corpus_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "addedAtUtc": self.added_at_utc,
            "addedReason": self.added_reason,
            "recommendationSourcePaperIds": self.recommendation_source_paper_ids,
            **self.paper.as_corpus_dict(),
        }


@dataclass(frozen=True)
class Corpus:
    created_at_utc: str
    updated_at_utc: str
    seed_paper_id: str
    seed_query: str
    papers: list[CorpusPaper]

    @property
    def paper_ids(self) -> list[str]:
        return [row.paper.paper_id for row in self.papers]

    @property
    def corpus_ids(self) -> set[int]:
        return {
            row.paper.corpus_id
            for row in self.papers
            if row.paper.corpus_id is not None
        }

    @property
    def paper_id_set(self) -> set[str]:
        return set(self.paper_ids)

    @classmethod
    def empty(cls, *, now_utc: str, seed_paper_id: str, seed_query: str) -> "Corpus":
        return cls(
            created_at_utc=now_utc,
            updated_at_utc=now_utc,
            seed_paper_id=seed_paper_id,
            seed_query=seed_query,
            papers=[],
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        seed_paper_id: str,
        seed_query: str,
        now_utc: str,
    ) -> "Corpus":
        papers_payload = payload.get("papers")
        papers: list[CorpusPaper] = []
        if isinstance(papers_payload, list):
            for row in papers_payload:
                if isinstance(row, dict):
                    corpus_paper = CorpusPaper.from_row(row)
                    if corpus_paper is not None:
                        papers.append(corpus_paper)

        return cls(
            created_at_utc=(
                _as_non_empty_string(payload.get("createdAtUtc")) or now_utc
            ),
            updated_at_utc=(
                _as_non_empty_string(payload.get("updatedAtUtc")) or now_utc
            ),
            seed_paper_id=(
                _as_non_empty_string(payload.get("seedPaperId")) or seed_paper_id
            ),
            seed_query=_as_non_empty_string(payload.get("seedQuery")) or seed_query,
            papers=sorted(papers, key=lambda row: row.rank),
        )

    def with_added_paper(
        self,
        paper: Paper,
        *,
        now_utc: str,
        reason: str,
        source_paper_ids: list[str] | None = None,
    ) -> "Corpus":
        return Corpus(
            created_at_utc=self.created_at_utc,
            updated_at_utc=now_utc,
            seed_paper_id=self.seed_paper_id,
            seed_query=self.seed_query,
            papers=[
                *self.papers,
                CorpusPaper(
                    rank=len(self.papers) + 1,
                    added_at_utc=now_utc,
                    added_reason=reason,
                    paper=paper,
                    recommendation_source_paper_ids=source_paper_ids or [],
                ),
            ],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "description": (
                "Incrementally grown Semantic Scholar paper corpus. "
                "Each run adds the highest-ranked recommendation not already present."
            ),
            "createdAtUtc": self.created_at_utc,
            "updatedAtUtc": self.updated_at_utc,
            "seedPaperId": self.seed_paper_id,
            "seedQuery": self.seed_query,
            "paperCount": len(self.papers),
            "papers": [row.as_corpus_dict() for row in self.papers],
        }


def _as_non_empty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _authors_from_payload(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    authors: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = _as_non_empty_string(item.get("name"))
            if name:
                authors.append(name)
    return authors

