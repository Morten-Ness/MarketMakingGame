from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from games.research_papers.config import (
    DEFAULT_SEED_PAPER_ID,
    RECOMMENDATION_FIELDS,
    Settings,
)
from games.research_papers.corpus import CorpusGrower, CorpusStore, first_untracked_paper
from games.research_papers.models import Corpus, Paper


def _paper(
    paper_id: str,
    title: str,
    *,
    corpus_id: int | None = None,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        corpus_id=corpus_id,
        url=f"https://www.semanticscholar.org/paper/{paper_id}",
    )


class FakeSemanticScholarClient:
    def __init__(self) -> None:
        self.batch_papers_response: list[Paper] = []
        self.search_papers_response: list[Paper] = []
        self.recommend_papers_response: list[Paper] = []
        self.batch_papers_by_id: dict[str, Paper] = {}
        self.recommend_positive_paper_ids: list[str] | None = None
        self.recommend_fields: str | None = None

    def batch_papers(self, paper_ids: list[str], *, fields: str) -> list[Paper]:
        if self.batch_papers_by_id:
            return [
                self.batch_papers_by_id[paper_id]
                for paper_id in paper_ids
                if paper_id in self.batch_papers_by_id
            ]
        return self.batch_papers_response

    def search_papers(self, query: str, *, fields: str, limit: int = 10) -> list[Paper]:
        return self.search_papers_response

    def recommend_papers(
        self,
        *,
        positive_paper_ids: list[str],
        negative_paper_ids: list[str],
        fields: str,
        limit: int,
    ) -> list[Paper]:
        self.recommend_positive_paper_ids = positive_paper_ids
        self.recommend_fields = fields
        return self.recommend_papers_response


class ResearchPaperCorpusTests(unittest.TestCase):
    def test_empty_corpus_grows_from_seed_paper_id(self) -> None:
        client = FakeSemanticScholarClient()
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1)
        client.batch_papers_response = [seed]
        grower = CorpusGrower(
            client=client,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_limit=100,
        )
        corpus = Corpus.empty(
            now_utc="2026-05-30T00:00:00+00:00",
            seed_paper_id=DEFAULT_SEED_PAPER_ID,
            seed_query="Seed Paper",
        )

        updated, added, reason = grower.grow(corpus)

        self.assertEqual(reason, "seed")
        self.assertEqual(added.paper_id, "seed-paper")
        self.assertEqual(updated.paper_ids, ["seed-paper"])
        self.assertEqual(updated.papers[0].rank, 1)

    def test_existing_corpus_grows_from_recommendations(self) -> None:
        client = FakeSemanticScholarClient()
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1)
        next_paper = _paper("next-paper", "Next Paper", corpus_id=2)
        client.recommend_papers_response = [seed, next_paper]
        detailed_next_paper = _paper("next-paper", "Detailed Next Paper", corpus_id=2)
        client.batch_papers_by_id = {"next-paper": detailed_next_paper}
        grower = CorpusGrower(
            client=client,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_limit=100,
        )
        corpus = Corpus.empty(
            now_utc="2026-05-30T00:00:00+00:00",
            seed_paper_id="seed-paper",
            seed_query="Seed Paper",
        ).with_added_paper(
            seed,
            now_utc="2026-05-30T00:00:01+00:00",
            reason="seed",
        )

        updated, added, reason = grower.grow(corpus)

        self.assertEqual(reason, "recommendation")
        self.assertEqual(added.paper_id, "next-paper")
        self.assertEqual(added.title, "Detailed Next Paper")
        self.assertEqual(updated.paper_ids, ["seed-paper", "next-paper"])
        self.assertEqual(updated.papers[1].rank, 2)
        self.assertEqual(client.recommend_positive_paper_ids, ["seed-paper"])
        self.assertEqual(client.recommend_fields, "title")
        self.assertEqual(
            updated.papers[1].recommendation_source_paper_ids,
            ["seed-paper"],
        )

    def test_first_untracked_paper_skips_existing_ids_and_corpus_ids(self) -> None:
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1)
        duplicate_by_id = _paper("seed-paper", "Duplicate by ID", corpus_id=10)
        duplicate_by_corpus_id = _paper("other-id", "Duplicate by Corpus", corpus_id=1)
        fresh = _paper("fresh-paper", "Fresh Paper", corpus_id=3)
        corpus = Corpus.empty(
            now_utc="2026-05-30T00:00:00+00:00",
            seed_paper_id="seed-paper",
            seed_query="Seed Paper",
        ).with_added_paper(
            seed,
            now_utc="2026-05-30T00:00:01+00:00",
            reason="seed",
        )

        selected = first_untracked_paper(
            [duplicate_by_id, duplicate_by_corpus_id, fresh],
            corpus,
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.paper_id, "fresh-paper")

    def test_corpus_store_writes_human_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corpus.json")
            store = CorpusStore(path)
            corpus = Corpus.empty(
                now_utc="2026-05-30T00:00:00+00:00",
                seed_paper_id="seed-paper",
                seed_query="Seed Paper",
            ).with_added_paper(
                _paper("seed-paper", "Seed Paper", corpus_id=1),
                now_utc="2026-05-30T00:00:01+00:00",
                reason="seed",
            )

            store.save(corpus)
            raw_text = Path(path).read_text(encoding="utf-8")
            payload = json.loads(raw_text)

        self.assertIn('\n  "schemaVersion"', raw_text)
        self.assertEqual(payload["paperCount"], 1)
        self.assertEqual(payload["papers"][0]["title"], "Seed Paper")

    def test_config_default_corpus_path_is_game_local(self) -> None:
        with (
            patch("games.research_papers.config.load_repo_env", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = Settings.from_env()

        self.assertEqual(
            settings.corpus_path,
            "games/research_papers/data/corpus.json",
        )
        self.assertEqual(settings.seed_paper_id, DEFAULT_SEED_PAPER_ID)
        self.assertNotIn("tldr", settings.recommendation_fields.split(","))
        self.assertEqual(settings.recommendation_fields, RECOMMENDATION_FIELDS)


if __name__ == "__main__":
    unittest.main()
