from __future__ import annotations

import json
from datetime import datetime, timezone
from difflib import SequenceMatcher

from shared.paths import resolve_repo_path

from .models import Corpus, Paper
from .semantic_scholar import SemanticScholarClient, SemanticScholarError


class CorpusGrowthError(RuntimeError):
    pass


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
        paper_fields: str,
        recommendation_fields: str,
        recommendation_limit: int,
    ) -> None:
        self._client = client
        self._paper_fields = paper_fields
        self._recommendation_fields = recommendation_fields
        self._recommendation_limit = recommendation_limit

    def grow(self, corpus: Corpus) -> tuple[Corpus, Paper, str]:
        if not corpus.papers:
            seed = self._resolve_seed(corpus.seed_paper_id, corpus.seed_query)
            return (
                corpus.with_added_paper(seed, now_utc=_utc_now(), reason="seed"),
                seed,
                "seed",
            )

        recommendations = self._client.recommend_papers(
            positive_paper_ids=corpus.paper_ids,
            negative_paper_ids=[],
            fields=self._recommendation_fields,
            limit=max(1, self._recommendation_limit),
        )
        next_paper = first_untracked_paper(recommendations, corpus)
        if next_paper is None:
            raise CorpusGrowthError(
                "Semantic Scholar returned no untracked recommendations for this corpus."
            )
        detailed_paper = self._fetch_detailed_paper(next_paper) or next_paper
        return (
            corpus.with_added_paper(
                detailed_paper,
                now_utc=_utc_now(),
                reason="semantic_scholar_recommendation",
                source_paper_ids=corpus.paper_ids,
            ),
            detailed_paper,
            "recommendation",
        )

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


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
