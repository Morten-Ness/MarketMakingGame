from __future__ import annotations

from shared.llm import OpenAIResponsesClient

from .config import Settings
from .corpus import CorpusGrower, CorpusGrowthError, CorpusStore
from .pdfs import PdfDownloader
from .prediction_game import (
    generate_prediction_exercise,
    read_prompt,
    read_raw_text_for_paper,
    run_prediction_game,
)
from .raw_text import RawTextExtractionError, RawTextExtractor
from .semantic_scholar import SemanticScholarClient, SemanticScholarError


def main() -> int:
    settings = Settings.from_env()
    if settings.enable_prediction_game and not settings.openai_api_key:
        print(
            "Research paper corpus error: OPENAI_API_KEY is required when "
            "RESEARCH_PAPERS_ENABLE_PREDICTION_GAME=true."
        )
        return 2

    store = CorpusStore(settings.corpus_path)
    client = SemanticScholarClient(settings.api_base_url, settings.api_key)
    pdf_downloader = PdfDownloader(
        settings.pdf_dir,
        timeout_seconds=settings.pdf_timeout_seconds,
        prefer_arxiv=settings.prefer_arxiv,
    )
    raw_text_extractor = RawTextExtractor(settings.raw_text_dir)
    grower = CorpusGrower(
        client=client,
        pdf_downloader=pdf_downloader,
        paper_fields=settings.paper_fields,
        recommendation_fields=settings.recommendation_fields,
        recommendation_initial_limit=settings.recommendation_initial_limit,
        recommendation_max_limit=settings.recommendation_max_limit,
        require_pdf=settings.require_pdf,
    )

    try:
        corpus = store.load(
            seed_paper_id=settings.seed_paper_id,
            seed_query=settings.seed_query,
        )
        result = grower.grow(corpus)
        store.save(result.corpus)
        raw_text_summary = raw_text_extractor.extract_for_corpus(result.corpus)
    except (CorpusGrowthError, SemanticScholarError, RawTextExtractionError) as exc:
        if isinstance(exc, CorpusGrowthError) and exc.partial_corpus is not None:
            store.save(exc.partial_corpus)
        print(f"Research paper corpus error: {exc}")
        return 2

    added_paper = result.added_paper
    print("Research Paper Corpus")
    print(f"Corpus path: {store.path}")
    print(f"PDF dir: {pdf_downloader.pdf_dir}")
    print(f"Raw text dir: {raw_text_summary.raw_text_dir}")
    print(f"Previous size: {len(corpus.papers)}")
    print(f"New size: {len(result.corpus.papers)}")
    print(f"Extracted raw text files: {raw_text_summary.extracted_count}")
    print(f"Skipped current raw text files: {raw_text_summary.skipped_count}")
    if result.backfilled_pdf_count:
        print(f"Backfilled PDFs: {result.backfilled_pdf_count}")
    if result.pruned_paper_count:
        print(f"Pruned non-PDF papers: {result.pruned_paper_count}")
    if result.recommendation_limit_used:
        print(f"Recommendation limit used: {result.recommendation_limit_used}")
    print(f"Added via: {result.reason}")
    print(f"Title: {added_paper.title}")
    print(f"Paper ID: {added_paper.paper_id}")
    if added_paper.corpus_id is not None:
        print(f"Corpus ID: {added_paper.corpus_id}")
    if added_paper.year is not None:
        print(f"Year: {added_paper.year}")
    if added_paper.authors:
        print(f"Authors: {', '.join(added_paper.authors[:8])}")
    if added_paper.url:
        print(f"URL: {added_paper.url}")
    if added_paper.pdf_path:
        print(f"PDF: {added_paper.pdf_path}")
    if added_paper.pdf_source:
        print(f"PDF source: {added_paper.pdf_source}")
    if added_paper.tldr:
        print(f"TLDR: {added_paper.tldr}")

    if settings.enable_prediction_game:
        try:
            assert settings.openai_api_key is not None
            llm_client = OpenAIResponsesClient(
                api_key=settings.openai_api_key,
                model=settings.strong_model,
            )
            print(llm_client.status)
            prompt_text = read_prompt(settings.prompt_path)
            raw_text = read_raw_text_for_paper(added_paper, settings.raw_text_dir)
            exercise = generate_prediction_exercise(
                llm_client=llm_client,
                prompt_text=prompt_text,
                paper=added_paper,
                raw_text=raw_text,
                max_text_chars=settings.game_max_text_chars,
            )
        except Exception as exc:
            print(f"Research paper prediction game error: {exc}")
            return 2

        run_prediction_game(
            exercise=exercise,
            paper=added_paper,
            model_name=settings.strong_model,
            log_path=settings.game_log_path,
            llm_client=llm_client,
        )
    return 0
