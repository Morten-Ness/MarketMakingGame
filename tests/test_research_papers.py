from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from games.research_papers.config import (
    DEFAULT_SEED_PAPER_ID,
    RECOMMENDATION_FIELDS,
    Settings,
)
from games.research_papers.corpus import (
    CorpusGrower,
    CorpusStore,
    first_untracked_paper,
    recommendation_limits,
)
from games.research_papers.models import Corpus, Paper
from games.research_papers.pdfs import (
    PdfCandidate,
    arxiv_pdf_url,
    is_pdf_response,
    pdf_candidates_for_paper,
    sanitize_filename,
    unique_pdf_path,
)
from games.research_papers.prediction_game import (
    PredictionExercise,
    is_prediction_correct,
    parse_single_option_answer,
    parse_user_option,
    render_visible_exercise,
    run_prediction_game,
    truncate_text,
)
from games.research_papers.raw_text import RawTextExtractor, raw_text_path_for_pdf


def _paper(
    paper_id: str,
    title: str,
    *,
    corpus_id: int | None = None,
    arxiv_id: str | None = None,
    open_access_pdf_url: str | None = None,
    with_pdf: bool = False,
) -> Paper:
    paper = Paper(
        paper_id=paper_id,
        title=title,
        corpus_id=corpus_id,
        external_ids={"ArXiv": arxiv_id} if arxiv_id else {},
        open_access_pdf_url=open_access_pdf_url,
        url=f"https://www.semanticscholar.org/paper/{paper_id}",
    )
    if not with_pdf:
        return paper
    return paper.with_pdf_metadata(
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id or paper_id}",
        pdf_source="arxiv",
        pdf_path="tests/test_research_papers.py",
        pdf_downloaded_at_utc="2026-05-30T00:00:02+00:00",
        pdf_byte_size=1024,
        pdf_sha256="abc123",
    )


class FakePdfDownloader:
    def __init__(self, successful_paper_ids: set[str]) -> None:
        self.successful_paper_ids = successful_paper_ids
        self.attempted_paper_ids: list[str] = []

    def download_for_paper(self, paper: Paper) -> Paper | None:
        self.attempted_paper_ids.append(paper.paper_id)
        if paper.paper_id not in self.successful_paper_ids:
            return None
        return paper.with_pdf_metadata(
            pdf_url=f"https://arxiv.org/pdf/{paper.paper_id}",
            pdf_source="arxiv",
            pdf_path=f"games/research_papers/pdfs/{paper.paper_id}.pdf",
            pdf_downloaded_at_utc="2026-05-30T00:00:03+00:00",
            pdf_byte_size=2048,
            pdf_sha256="def456",
        )


class FakeAnswerLlmClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    @property
    def status(self) -> str:
        return "fake answer llm"

    def generate_json(
        self,
        system_instruction: str,
        payload: object,
        schema: dict[str, object] | None = None,
    ) -> object:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "payload": payload,
                "schema": schema or {},
            }
        )
        return self.response

    def generate_text(self, system_instruction: str, payload: object) -> str:
        raise NotImplementedError


class FakeSemanticScholarClient:
    def __init__(self) -> None:
        self.batch_papers_response: list[Paper] = []
        self.search_papers_response: list[Paper] = []
        self.recommend_papers_response: list[Paper] = []
        self.recommend_papers_by_limit: dict[int, list[Paper]] = {}
        self.batch_papers_by_id: dict[str, Paper] = {}
        self.recommend_positive_paper_ids: list[str] | None = None
        self.recommend_fields: str | None = None
        self.recommend_limits: list[int] = []

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
        self.recommend_limits.append(limit)
        if self.recommend_papers_by_limit:
            return self.recommend_papers_by_limit.get(limit, [])
        return self.recommend_papers_response


class ResearchPaperCorpusTests(unittest.TestCase):
    def test_empty_corpus_grows_from_seed_paper_id(self) -> None:
        client = FakeSemanticScholarClient()
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1)
        client.batch_papers_response = [seed]
        pdf_downloader = FakePdfDownloader({"seed-paper"})
        grower = CorpusGrower(
            client=client,
            pdf_downloader=pdf_downloader,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_initial_limit=25,
            recommendation_max_limit=200,
        )
        corpus = Corpus.empty(
            now_utc="2026-05-30T00:00:00+00:00",
            seed_paper_id=DEFAULT_SEED_PAPER_ID,
            seed_query="Seed Paper",
        )

        result = grower.grow(corpus)

        self.assertEqual(result.reason, "seed")
        self.assertEqual(result.added_paper.paper_id, "seed-paper")
        self.assertTrue(result.added_paper.has_pdf)
        self.assertEqual(result.corpus.paper_ids, ["seed-paper"])
        self.assertEqual(result.corpus.papers[0].rank, 1)

    def test_existing_corpus_grows_from_recommendations(self) -> None:
        client = FakeSemanticScholarClient()
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1, with_pdf=True)
        next_paper = _paper("next-paper", "Next Paper", corpus_id=2)
        client.recommend_papers_response = [seed, next_paper]
        detailed_next_paper = _paper("next-paper", "Detailed Next Paper", corpus_id=2)
        client.batch_papers_by_id = {"next-paper": detailed_next_paper}
        pdf_downloader = FakePdfDownloader({"next-paper"})
        grower = CorpusGrower(
            client=client,
            pdf_downloader=pdf_downloader,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_initial_limit=25,
            recommendation_max_limit=200,
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

        result = grower.grow(corpus)

        self.assertEqual(result.reason, "recommendation")
        self.assertEqual(result.added_paper.paper_id, "next-paper")
        self.assertEqual(result.added_paper.title, "Detailed Next Paper")
        self.assertTrue(result.added_paper.has_pdf)
        self.assertEqual(result.corpus.paper_ids, ["seed-paper", "next-paper"])
        self.assertEqual(result.corpus.papers[1].rank, 2)
        self.assertEqual(client.recommend_positive_paper_ids, ["seed-paper"])
        self.assertEqual(client.recommend_fields, "title")
        self.assertEqual(
            result.corpus.papers[1].recommendation_source_paper_ids,
            ["seed-paper"],
        )

    def test_widens_recommendation_limits_until_downloadable_paper_succeeds(self) -> None:
        client = FakeSemanticScholarClient()
        seed = _paper("seed-paper", "Seed Paper", corpus_id=1, with_pdf=True)
        failed = _paper("failed-paper", "Failed Paper", corpus_id=2)
        success = _paper("success-paper", "Success Paper", corpus_id=3)
        client.recommend_papers_by_limit = {
            25: [failed],
            50: [failed],
            100: [failed, success],
        }
        client.batch_papers_by_id = {
            "failed-paper": failed,
            "success-paper": success,
        }
        pdf_downloader = FakePdfDownloader({"success-paper"})
        grower = CorpusGrower(
            client=client,
            pdf_downloader=pdf_downloader,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_initial_limit=25,
            recommendation_max_limit=200,
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

        result = grower.grow(corpus)

        self.assertEqual(client.recommend_limits, [25, 50, 100])
        self.assertEqual(result.added_paper.paper_id, "success-paper")
        self.assertEqual(result.recommendation_limit_used, 100)
        self.assertEqual(
            pdf_downloader.attempted_paper_ids,
            ["failed-paper", "success-paper"],
        )

    def test_backfills_and_prunes_existing_corpus_before_growth(self) -> None:
        client = FakeSemanticScholarClient()
        missing_pdf = _paper("missing-pdf", "Missing PDF", corpus_id=1)
        pruned = _paper("pruned-paper", "Pruned Paper", corpus_id=2)
        next_paper = _paper("next-paper", "Next Paper", corpus_id=3)
        client.recommend_papers_response = [next_paper]
        client.batch_papers_by_id = {
            "missing-pdf": missing_pdf,
            "pruned-paper": pruned,
            "next-paper": next_paper,
        }
        pdf_downloader = FakePdfDownloader({"missing-pdf", "next-paper"})
        grower = CorpusGrower(
            client=client,
            pdf_downloader=pdf_downloader,
            paper_fields="title,tldr",
            recommendation_fields="title",
            recommendation_initial_limit=25,
            recommendation_max_limit=25,
        )
        corpus = Corpus.empty(
            now_utc="2026-05-30T00:00:00+00:00",
            seed_paper_id="seed-paper",
            seed_query="Seed Paper",
        ).with_added_paper(
            missing_pdf,
            now_utc="2026-05-30T00:00:01+00:00",
            reason="seed",
        ).with_added_paper(
            pruned,
            now_utc="2026-05-30T00:00:02+00:00",
            reason="semantic_scholar_recommendation",
        )

        result = grower.grow(corpus)

        self.assertEqual(result.backfilled_pdf_count, 1)
        self.assertEqual(result.pruned_paper_count, 1)
        self.assertEqual(result.corpus.paper_ids, ["missing-pdf", "next-paper"])
        self.assertTrue(all(row.paper.has_pdf for row in result.corpus.papers))

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
        self.assertEqual(settings.pdf_dir, "games/research_papers/pdfs")
        self.assertEqual(settings.raw_text_dir, "games/research_papers/raw_text")
        self.assertTrue(settings.require_pdf)
        self.assertTrue(settings.prefer_arxiv)
        self.assertEqual(settings.recommendation_initial_limit, 25)
        self.assertEqual(settings.recommendation_max_limit, 200)
        self.assertTrue(settings.enable_prediction_game)
        self.assertEqual(settings.strong_model, "gpt-5")
        self.assertEqual(
            settings.game_log_path,
            "games/research_papers/logs/prediction_games.jsonl",
        )
        self.assertEqual(settings.game_max_text_chars, 160_000)


class ResearchPaperPdfTests(unittest.TestCase):
    def test_arxiv_pdf_url_derives_from_external_ids(self) -> None:
        paper = _paper("paper-id", "Paper", arxiv_id="ArXiv:2605.27295")

        self.assertEqual(arxiv_pdf_url(paper), "https://arxiv.org/pdf/2605.27295")

    def test_pdf_candidates_prefer_arxiv_then_open_access(self) -> None:
        paper = _paper(
            "paper-id",
            "Paper",
            arxiv_id="2605.27295",
            open_access_pdf_url="https://example.test/paper.pdf",
        )

        candidates = pdf_candidates_for_paper(paper, prefer_arxiv=True)

        self.assertEqual(
            candidates,
            [
                PdfCandidate("https://arxiv.org/pdf/2605.27295", "arxiv"),
                PdfCandidate(
                    "https://example.test/paper.pdf",
                    "semantic_scholar_open_access",
                ),
            ],
        )

    def test_sanitizes_and_trims_title_filenames(self) -> None:
        title = 'A/B:C*D?E "F" <G> | H ' + ("long " * 40)

        filename = sanitize_filename(title, max_length=40)

        self.assertLessEqual(len(filename), 40)
        self.assertNotRegex(filename, r'[<>:"/\\|?*]')
        self.assertTrue(filename.startswith("A B C D E"))

    def test_unique_pdf_path_adds_suffix_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir)
            first = unique_pdf_path(pdf_dir, "Paper Title")
            first.write_bytes(b"%PDF-1.7")
            second = unique_pdf_path(pdf_dir, "Paper Title")

        self.assertEqual(first.name, "Paper Title.pdf")
        self.assertEqual(second.name, "Paper Title-2.pdf")

    def test_pdf_validation_accepts_pdf_bytes_and_rejects_html(self) -> None:
        self.assertTrue(is_pdf_response(b"%PDF-1.7\ncontent", "application/pdf"))
        self.assertFalse(is_pdf_response(b"<html>not a pdf</html>", "text/html"))

    def test_recommendation_limits_widen_gradually(self) -> None:
        self.assertEqual(recommendation_limits(25, 200), [25, 50, 100, 200])
        self.assertEqual(recommendation_limits(40, 100), [40, 80, 100])


class ResearchPaperRawTextTests(unittest.TestCase):
    def test_raw_text_path_uses_same_pdf_stem_with_txt_extension(self) -> None:
        path = raw_text_path_for_pdf(
            Path("games/research_papers/pdfs/Example Paper.pdf"),
            Path("games/research_papers/raw_text"),
        )

        self.assertEqual(
            path,
            Path("games/research_papers/raw_text/Example Paper.txt"),
        )

    def test_extract_for_corpus_writes_raw_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_dir = tmp_path / "pdfs"
            pdf_dir.mkdir()
            pdf_path = pdf_dir / "Example Paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.7")
            raw_text_dir = tmp_path / "raw_text"
            paper = _paper("paper-id", "Example Paper", corpus_id=1).with_pdf_metadata(
                pdf_url="https://arxiv.org/pdf/1234.5678",
                pdf_source="arxiv",
                pdf_path=str(pdf_path),
                pdf_downloaded_at_utc="2026-05-30T00:00:00+00:00",
                pdf_byte_size=10,
                pdf_sha256="abc123",
            )
            corpus = Corpus.empty(
                now_utc="2026-05-30T00:00:00+00:00",
                seed_paper_id="paper-id",
                seed_query="Example Paper",
            ).with_added_paper(
                paper,
                now_utc="2026-05-30T00:00:01+00:00",
                reason="seed",
            )
            extractor = RawTextExtractor(
                str(raw_text_dir),
                extract_text_func=lambda _path: "extracted text",
            )

            summary = extractor.extract_for_corpus(corpus)

            self.assertEqual(summary.extracted_count, 1)
            self.assertEqual(summary.skipped_count, 0)
            self.assertEqual(
                (raw_text_dir / "Example Paper.txt").read_text(encoding="utf-8"),
                "extracted text\n",
            )

    def test_extract_for_corpus_skips_current_raw_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_path = tmp_path / "Example Paper.pdf"
            text_path = tmp_path / "Example Paper.txt"
            pdf_path.write_bytes(b"%PDF-1.7")
            text_path.write_text("already extracted\n", encoding="utf-8")
            paper = _paper("paper-id", "Example Paper", corpus_id=1).with_pdf_metadata(
                pdf_url="https://arxiv.org/pdf/1234.5678",
                pdf_source="arxiv",
                pdf_path=str(pdf_path),
                pdf_downloaded_at_utc="2026-05-30T00:00:00+00:00",
                pdf_byte_size=10,
                pdf_sha256="abc123",
            )
            corpus = Corpus.empty(
                now_utc="2026-05-30T00:00:00+00:00",
                seed_paper_id="paper-id",
                seed_query="Example Paper",
            ).with_added_paper(
                paper,
                now_utc="2026-05-30T00:00:01+00:00",
                reason="seed",
            )
            extractor = RawTextExtractor(
                str(tmp_path),
                extract_text_func=lambda _path: "new text",
            )

            summary = extractor.extract_for_corpus(corpus)

            self.assertEqual(summary.extracted_count, 0)
            self.assertEqual(summary.skipped_count, 1)
            self.assertEqual(
                text_path.read_text(encoding="utf-8"),
                "already extracted\n",
            )


class ResearchPaperPredictionGameTests(unittest.TestCase):
    def test_accepts_suitable_and_not_suitable_exercise_json(self) -> None:
        suitable = PredictionExercise.from_payload(_exercise_payload())
        not_suitable = PredictionExercise.from_payload(
            {
                "suitability": "Not suitable",
                "neutral_setup": "",
                "prediction_question": "",
                "options": [],
                "reasoning_prompt": "",
                "reveal": {
                    "correct_option": "",
                    "result_summary": "",
                    "correctness_explanation": "",
                    "learning_note": "",
                    "caveats": "",
                },
                "not_suitable_reason": "The paper is mainly a survey.",
                "alternative_reading_exercise": "Make a concept map.",
            }
        )

        self.assertTrue(suitable.is_suitable)
        self.assertFalse(not_suitable.is_suitable)

    def test_visible_rendering_does_not_include_reveal_content(self) -> None:
        exercise = PredictionExercise.from_payload(_exercise_payload())

        visible = render_visible_exercise(exercise)

        self.assertIn("Prediction question:", visible)
        self.assertNotIn("secret result", visible)
        self.assertNotIn("Correct answer", visible)

    def test_answer_parsing_and_correctness(self) -> None:
        labels = {"A", "B", "C"}

        self.assertEqual(parse_single_option_answer("A", labels), "A")
        self.assertEqual(parse_single_option_answer("a.", labels), "A")
        self.assertIsNone(parse_single_option_answer("I choose B because", labels))
        self.assertEqual(parse_user_option("A", labels), "A")
        self.assertEqual(parse_user_option("a.", labels), "A")
        self.assertEqual(parse_user_option("I choose B because it seems plausible", labels), "B")
        self.assertEqual(parse_user_option("My final prediction is A.", labels), "A")
        self.assertIsNone(parse_user_option("I am torn between things", labels))
        self.assertTrue(is_prediction_correct("B", "B"))
        self.assertFalse(is_prediction_correct("A", "B"))

    def test_long_answer_uses_llm_for_prediction_and_reasoning_feedback(self) -> None:
        exercise = PredictionExercise.from_payload(_exercise_payload())
        printed: list[str] = []
        answer_llm = FakeAnswerLlmClient(
            {
                "parsed_option": "A",
                "reasoning_summary": "The user thinks Method A should win.",
                "reasoning_assessment": (
                    "Your intuition about the setup was plausible, but the paper's "
                    "main table favors Method B."
                ),
                "interpretation_notes": "Final prediction was explicit.",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "prediction_games.jsonl")
            outcome = run_prediction_game(
                exercise=exercise,
                paper=_paper("paper-id", "Example Paper", corpus_id=1, with_pdf=True),
                model_name="gpt-5",
                log_path=log_path,
                llm_client=answer_llm,
                input_func=lambda _prompt: "I reason through it, so my final prediction is A.",
                print_func=printed.append,
            )
            row = json.loads(Path(log_path).read_text(encoding="utf-8").strip())

        self.assertEqual(len(answer_llm.calls), 1)
        self.assertEqual(outcome.parsed_option, "A")
        self.assertFalse(outcome.correct)
        self.assertEqual(outcome.answer_parser, "llm")
        self.assertIn("Your reasoning:", "\n".join(printed))
        self.assertIn("main table favors Method B", "\n".join(printed))
        self.assertEqual(row["reasoningSummary"], "The user thinks Method A should win.")
        self.assertIn("main table favors Method B", row["reasoningAssessment"])

    def test_single_letter_answer_skips_llm_interpretation(self) -> None:
        exercise = PredictionExercise.from_payload(_exercise_payload())
        printed: list[str] = []
        answer_llm = FakeAnswerLlmClient(
            {
                "parsed_option": "A",
                "reasoning_summary": "Should not be used.",
                "reasoning_assessment": "Should not be printed.",
                "interpretation_notes": "",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outcome = run_prediction_game(
                exercise=exercise,
                paper=_paper("paper-id", "Example Paper", corpus_id=1, with_pdf=True),
                model_name="gpt-5",
                log_path=str(Path(tmpdir) / "prediction_games.jsonl"),
                llm_client=answer_llm,
                input_func=lambda _prompt: "B",
                print_func=printed.append,
            )

        self.assertEqual(answer_llm.calls, [])
        self.assertEqual(outcome.parsed_option, "B")
        self.assertTrue(outcome.correct)
        self.assertNotIn("Should not be printed", "\n".join(printed))

    def test_unclear_answer_asks_again_before_reveal_and_logs_result(self) -> None:
        exercise = PredictionExercise.from_payload(_exercise_payload())
        printed: list[str] = []
        answers = iter(["not sure yet", "I choose B because of the setup"])

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "prediction_games.jsonl")
            outcome = run_prediction_game(
                exercise=exercise,
                paper=_paper("paper-id", "Example Paper", corpus_id=1, with_pdf=True),
                model_name="gpt-5",
                log_path=log_path,
                input_func=lambda _prompt: next(answers),
                print_func=printed.append,
            )
            row = json.loads(Path(log_path).read_text(encoding="utf-8").strip())

        self.assertTrue(outcome.revealed)
        self.assertEqual(outcome.parsed_option, "B")
        self.assertTrue(outcome.correct)
        self.assertIn("Please answer with a clear option label", "\n".join(printed))
        self.assertIn("secret result", "\n".join(printed))
        self.assertTrue(row["revealed"])
        self.assertEqual(row["reveal"]["correct_option"], "B")
        self.assertEqual(row["correct"], True)

    def test_truncate_text_preserves_front_and_back(self) -> None:
        text = "front-" + ("middle" * 100) + "-back"

        truncated = truncate_text(text, 80)

        self.assertTrue(truncated.startswith("front-"))
        self.assertTrue(truncated.endswith("-back"))
        self.assertIn("middle of paper omitted", truncated)

    def test_missing_openai_key_fails_before_corpus_growth(self) -> None:
        from games.research_papers import cli

        settings = SimpleNamespace(
            enable_prediction_game=True,
            openai_api_key=None,
        )
        printed: list[str] = []

        with (
            patch("games.research_papers.cli.Settings.from_env", return_value=settings),
            patch("builtins.print", side_effect=printed.append),
            patch("games.research_papers.cli.CorpusStore") as corpus_store,
        ):
            exit_code = cli.main()

        self.assertEqual(exit_code, 2)
        self.assertFalse(corpus_store.called)
        self.assertIn("OPENAI_API_KEY", "\n".join(printed))


def _exercise_payload() -> dict[str, object]:
    return {
        "suitability": "Suitable",
        "neutral_setup": "Two methods are compared on the same benchmark.",
        "prediction_question": "Which method has the better main result?",
        "options": [
            {"label": "A", "text": "Method A is better."},
            {"label": "B", "text": "Method B is better."},
            {"label": "C", "text": "They are similar."},
        ],
        "reasoning_prompt": "Give your firm prediction, optionally with reasoning.",
        "reveal": {
            "correct_option": "B",
            "result_summary": "The secret result is that Method B is better.",
            "correctness_explanation": "The main table favors Method B.",
            "learning_note": "The benchmark rewards the relevant mechanism.",
            "caveats": "The result is benchmark-specific.",
        },
        "not_suitable_reason": "",
        "alternative_reading_exercise": "",
    }


if __name__ == "__main__":
    unittest.main()
