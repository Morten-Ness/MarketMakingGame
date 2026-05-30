from __future__ import annotations

from .config import Settings
from .corpus import CorpusGrower, CorpusGrowthError, CorpusStore
from .semantic_scholar import SemanticScholarClient, SemanticScholarError


def main() -> int:
    settings = Settings.from_env()
    store = CorpusStore(settings.corpus_path)
    client = SemanticScholarClient(settings.api_base_url, settings.api_key)
    grower = CorpusGrower(
        client=client,
        paper_fields=settings.paper_fields,
        recommendation_fields=settings.recommendation_fields,
        recommendation_limit=settings.recommendation_limit,
    )

    try:
        corpus = store.load(
            seed_paper_id=settings.seed_paper_id,
            seed_query=settings.seed_query,
        )
        updated_corpus, added_paper, reason = grower.grow(corpus)
        store.save(updated_corpus)
    except (CorpusGrowthError, SemanticScholarError) as exc:
        print(f"Research paper corpus error: {exc}")
        return 2

    print("Research Paper Corpus")
    print(f"Corpus path: {store.path}")
    print(f"Previous size: {len(corpus.papers)}")
    print(f"New size: {len(updated_corpus.papers)}")
    print(f"Added via: {reason}")
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
    if added_paper.tldr:
        print(f"TLDR: {added_paper.tldr}")
    return 0
