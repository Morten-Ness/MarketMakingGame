from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from shared.paths import resolve_repo_path

from .models import Corpus, Paper
from .pdfs import PdfDownloader
from .semantic_scholar import SemanticScholarClient, SemanticScholarError


class CorpusGrowthError(RuntimeError):
    def __init__(self, message: str, *, partial_corpus: Corpus | None = None) -> None:
        super().__init__(message)
        self.partial_corpus = partial_corpus


@dataclass(frozen=True)
class CorpusGrowthResult:
    corpus: Corpus
    added_paper: Paper
    reason: str
    backfilled_pdf_count: int
    pruned_paper_count: int
    recommendation_limit_used: int | None = None


class CorpusStore:
    def __init__(self, path: str) -> None:
        self._path = resolve_repo_path(path)

    @property
    def path(self):
        return self._path

    def load(self, *, seed_paper_id: str, seed_query: str) -> Corpus:
        now = _utc_now()
        if not self._path.exists():
            return Corpus.empty(
                now_utc=now,
                seed_paper_id=seed_paper_id,
                seed_query=seed_query,
            )

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusGrowthError(f"Could not read corpus JSON at {self._path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise CorpusGrowthError(f"Corpus JSON at {self._path} must contain an object.")
        return Corpus.from_payload(
            payload,
            seed_paper_id=seed_paper_id,
            seed_query=seed_query,
            now_utc=now,
        )

    def save(self, corpus: Corpus) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(corpus.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


class CorpusGrower:
    def __init__(
        self,
        *,
        client: SemanticScholarClient,
        pdf_downloader: PdfDownloader,
        paper_fields: str,
        recommendation_fields: str,
        recommendation_initial_limit: int,
        recommendation_max_limit: int,
        require_pdf: bool = True,
    ) -> None:
        self._client = client
        self._pdf_downloader = pdf_downloader
        self._paper_fields = paper_fields
        self._recommendation_fields = recommendation_fields
        self._recommendation_initial_limit = recommendation_initial_limit
        self._recommendation_max_limit = recommendation_max_limit
        self._require_pdf = require_pdf

    def grow(self, corpus: Corpus) -> CorpusGrowthResult:
        prepared_corpus, backfilled_count, pruned_count = self._prepare_existing_corpus(
            corpus
        )

        if not prepared_corpus.papers:
            seed = self._resolve_seed(corpus.seed_paper_id, corpus.seed_query)
            seed = self._ensure_downloaded_pdf(seed)
            if seed is None:
                raise CorpusGrowthError(
                    "The seed paper did not have a downloadable PDF.",
                    partial_corpus=prepared_corpus,
                )
            updated_corpus = prepared_corpus.with_added_paper(
                seed,
                now_utc=_utc_now(),
                reason="seed",
            )
            return CorpusGrowthResult(
                corpus=updated_corpus,
                added_paper=seed,
                reason="seed",
                backfilled_pdf_count=backfilled_count,
                pruned_paper_count=pruned_count,
            )

        next_paper, limit_used = self._find_downloadable_recommendation(prepared_corpus)
        if next_paper is None:
            raise CorpusGrowthError(
                "Semantic Scholar returned no untracked recommendations with a "
                "downloadable PDF by the configured max recommendation limit.",
                partial_corpus=prepared_corpus,
            )

        updated_corpus = prepared_corpus.with_added_paper(
            next_paper,
            now_utc=_utc_now(),
            reason="semantic_scholar_recommendation",
            source_paper_ids=prepared_corpus.paper_ids,
        )
        return CorpusGrowthResult(
            corpus=updated_corpus,
            added_paper=next_paper,
            reason="recommendation",
            backfilled_pdf_count=backfilled_count,
            pruned_paper_count=pruned_count,
            recommendation_limit_used=limit_used,
        )

    def _prepare_existing_corpus(self, corpus: Corpus) -> tuple[Corpus, int, int]:
        if not self._require_pdf:
            return corpus, 0, 0

        kept_rows = []
        backfilled_count = 0
        pruned_count = 0
        for row in corpus.papers:
            if row.paper.has_pdf and _local_pdf_exists(row.paper):
                kept_rows.append(row)
                continue

            detailed_paper = self._fetch_detailed_paper(row.paper) or row.paper
            paper_with_pdf = self._ensure_downloaded_pdf(detailed_paper)
            if paper_with_pdf is None:
                pruned_count += 1
                continue

            kept_rows.append(row.with_paper(paper_with_pdf))
            backfilled_count += 1

        if backfilled_count == 0 and pruned_count == 0:
            return corpus, 0, 0
        return (
            corpus.with_papers(kept_rows, now_utc=_utc_now()),
            backfilled_count,
            pruned_count,
        )

    def _find_downloadable_recommendation(
        self,
        corpus: Corpus,
    ) -> tuple[Paper | None, int | None]:
        failed_paper_ids: set[str] = set()
        for limit in recommendation_limits(
            self._recommendation_initial_limit,
            self._recommendation_max_limit,
        ):
            recommendations = self._client.recommend_papers(
                positive_paper_ids=corpus.paper_ids,
                negative_paper_ids=[],
                fields=self._recommendation_fields,
                limit=limit,
            )
            for recommendation in recommendations:
                if recommendation.paper_id in failed_paper_ids:
                    continue
                if _is_tracked(recommendation, corpus):
                    continue
                detailed_paper = (
                    self._fetch_detailed_paper(recommendation) or recommendation
                )
                paper_with_pdf = self._ensure_downloaded_pdf(detailed_paper)
                if paper_with_pdf is not None:
                    return paper_with_pdf, limit
                failed_paper_ids.add(recommendation.paper_id)

        return None, None

    def _ensure_downloaded_pdf(self, paper: Paper) -> Paper | None:
        if not self._require_pdf:
            return paper
        if paper.has_pdf:
            return paper
        return self._pdf_downloader.download_for_paper(paper)

    def _resolve_seed(self, seed_paper_id: str, seed_query: str) -> Paper:
        try:
            papers = self._client.batch_papers([seed_paper_id], fields=self._paper_fields)
        except SemanticScholarError:
            papers = []

        if papers:
            return papers[0]

        search_results = self._client.search_papers(
            seed_query,
            fields=self._paper_fields,
            limit=10,
        )
        if not search_results:
            raise CorpusGrowthError(
                "Could not find the seed paper through Semantic Scholar. "
                "Set RESEARCH_PAPERS_SEED_PAPER_ID or RESEARCH_PAPERS_SEED_QUERY."
            )
        return max(
            search_results,
            key=lambda paper: _title_similarity(seed_query, paper.title),
        )

    def _fetch_detailed_paper(self, paper: Paper) -> Paper | None:
        try:
            papers = self._client.batch_papers([paper.paper_id], fields=self._paper_fields)
        except SemanticScholarError:
            return None
        return papers[0] if papers else None


def first_untracked_paper(papers: list[Paper], corpus: Corpus) -> Paper | None:
    existing_paper_ids = corpus.paper_id_set
    existing_corpus_ids = corpus.corpus_ids
    for paper in papers:
        if paper.paper_id in existing_paper_ids:
            continue
        if paper.corpus_id is not None and paper.corpus_id in existing_corpus_ids:
            continue
        return paper
    return None


def recommendation_limits(initial_limit: int, max_limit: int) -> list[int]:
    initial = max(1, initial_limit)
    maximum = max(1, max_limit)
    if initial >= maximum:
        return [initial]

    limits = [initial]
    current = initial
    while current < maximum:
        current = min(current * 2, maximum)
        if current != limits[-1]:
            limits.append(current)
    return limits


def _is_tracked(paper: Paper, corpus: Corpus) -> bool:
    if paper.paper_id in corpus.paper_id_set:
        return True
    return paper.corpus_id is not None and paper.corpus_id in corpus.corpus_ids


def _local_pdf_exists(paper: Paper) -> bool:
    return bool(paper.pdf_path and resolve_repo_path(paper.pdf_path).exists())


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
