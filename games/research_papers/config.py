from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import env_int, load_repo_env


DEFAULT_SEED_QUERY = "Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini"
DEFAULT_SEED_PAPER_ID = "ArXiv:2605.27295"

PAPER_FIELDS = ",".join(
    (
        "paperId",
        "corpusId",
        "externalIds",
        "url",
        "title",
        "abstract",
        "venue",
        "year",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "fieldsOfStudy",
        "publicationTypes",
        "publicationDate",
        "authors",
        "tldr",
    )
)

RECOMMENDATION_FIELDS = ",".join(
    field for field in PAPER_FIELDS.split(",") if field != "tldr"
)


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    api_key: str | None
    corpus_path: str
    seed_paper_id: str
    seed_query: str
    recommendation_limit: int
    paper_fields: str
    recommendation_fields: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_repo_env()
        return cls(
            api_base_url=os.getenv(
                "SEMANTIC_SCHOLAR_API_BASE_URL",
                "https://api.semanticscholar.org",
            ).rstrip("/"),
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
            corpus_path=os.getenv(
                "RESEARCH_PAPERS_CORPUS_PATH",
                "games/research_papers/data/corpus.json",
            ),
            seed_paper_id=os.getenv(
                "RESEARCH_PAPERS_SEED_PAPER_ID",
                DEFAULT_SEED_PAPER_ID,
            ).strip(),
            seed_query=os.getenv(
                "RESEARCH_PAPERS_SEED_QUERY",
                DEFAULT_SEED_QUERY,
            ).strip(),
            recommendation_limit=env_int("RESEARCH_PAPERS_RECOMMENDATION_LIMIT", 100),
            paper_fields=os.getenv("RESEARCH_PAPERS_PAPER_FIELDS", PAPER_FIELDS).strip(),
            recommendation_fields=os.getenv(
                "RESEARCH_PAPERS_RECOMMENDATION_FIELDS",
                RECOMMENDATION_FIELDS,
            ).strip(),
        )
