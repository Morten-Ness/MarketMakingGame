from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import env_bool, env_int, load_repo_env


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
    recommendation_initial_limit: int
    recommendation_max_limit: int
    require_pdf: bool
    prefer_arxiv: bool
    pdf_dir: str
    raw_text_dir: str
    pdf_timeout_seconds: int
    enable_prediction_game: bool
    openai_api_key: str | None
    strong_model: str
    game_log_path: str
    game_max_text_chars: int
    prompt_path: str
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
            recommendation_initial_limit=env_int(
                "RESEARCH_PAPERS_RECOMMENDATION_INITIAL_LIMIT",
                25,
            ),
            recommendation_max_limit=env_int(
                "RESEARCH_PAPERS_RECOMMENDATION_MAX_LIMIT",
                env_int("RESEARCH_PAPERS_RECOMMENDATION_LIMIT", 200),
            ),
            require_pdf=env_bool("RESEARCH_PAPERS_REQUIRE_PDF", True),
            prefer_arxiv=env_bool("RESEARCH_PAPERS_PREFER_ARXIV", True),
            pdf_dir=os.getenv(
                "RESEARCH_PAPERS_PDF_DIR",
                "games/research_papers/pdfs",
            ),
            raw_text_dir=os.getenv(
                "RESEARCH_PAPERS_RAW_TEXT_DIR",
                "games/research_papers/raw_text",
            ),
            pdf_timeout_seconds=env_int("RESEARCH_PAPERS_PDF_TIMEOUT_SECONDS", 45),
            enable_prediction_game=env_bool(
                "RESEARCH_PAPERS_ENABLE_PREDICTION_GAME",
                True,
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
            strong_model=os.getenv("RESEARCH_PAPERS_STRONG_MODEL", "gpt-5").strip(),
            game_log_path=os.getenv(
                "RESEARCH_PAPERS_GAME_LOG_PATH",
                "games/research_papers/logs/prediction_games.jsonl",
            ),
            game_max_text_chars=env_int("RESEARCH_PAPERS_GAME_MAX_TEXT_CHARS", 160_000),
            prompt_path=os.getenv(
                "RESEARCH_PAPERS_PREDICTION_PROMPT_PATH",
                "games/research_papers/prompt.txt",
            ),
            paper_fields=os.getenv("RESEARCH_PAPERS_PAPER_FIELDS", PAPER_FIELDS).strip(),
            recommendation_fields=os.getenv(
                "RESEARCH_PAPERS_RECOMMENDATION_FIELDS",
                RECOMMENDATION_FIELDS,
            ).strip(),
        )
