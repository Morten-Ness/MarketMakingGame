from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

from games.verbal_zetamac.config import Settings
from games.verbal_zetamac.engine import (
    QuestionGenerator,
    compute_score,
    parse_spoken_integer,
)
from games.verbal_zetamac.models import AnswerAttempt, GenerationSettings, Question


class VerbalZetamacQuestionGenerationTests(unittest.TestCase):
    def test_generates_addition_and_subtraction_from_addition_range(self) -> None:
        settings = GenerationSettings(
            operations=("addition", "subtraction"),
            addition_min=2,
            addition_max=100,
        )
        generator = QuestionGenerator(settings, random.Random(1))

        addition = generator.generate_one("addition")
        subtraction = generator.generate_one("subtraction")

        self.assertEqual(addition.answer, addition.left + addition.right)
        self.assertGreaterEqual(addition.left, 2)
        self.assertLessEqual(addition.left, 100)
        self.assertGreaterEqual(addition.right, 2)
        self.assertLessEqual(addition.right, 100)
        self.assertEqual(subtraction.answer, subtraction.left - subtraction.right)
        self.assertGreaterEqual(subtraction.answer, 2)
        self.assertLessEqual(subtraction.answer, 100)

    def test_generates_multiplication_and_integer_division(self) -> None:
        settings = GenerationSettings(
            operations=("multiplication", "division"),
            multiplier_min=2,
            multiplier_max=12,
            multiplicand_min=2,
            multiplicand_max=100,
        )
        generator = QuestionGenerator(settings, random.Random(2))

        multiplication = generator.generate_one("multiplication")
        division = generator.generate_one("division")

        self.assertEqual(multiplication.answer, multiplication.left * multiplication.right)
        self.assertGreaterEqual(multiplication.left, 2)
        self.assertLessEqual(multiplication.left, 12)
        self.assertGreaterEqual(multiplication.right, 2)
        self.assertLessEqual(multiplication.right, 100)
        self.assertEqual(division.left % division.right, 0)
        self.assertEqual(division.answer, division.left // division.right)

    def test_uniform_random_generation_uses_configured_operations(self) -> None:
        settings = GenerationSettings(operations=("addition",), addition_min=2, addition_max=2)
        generator = QuestionGenerator(settings, random.Random(3))

        questions = generator.generate(5)

        self.assertTrue(all(question.operation == "addition" for question in questions))


class VerbalZetamacAnswerParsingTests(unittest.TestCase):
    def test_parses_digit_answers(self) -> None:
        self.assertEqual(parse_spoken_integer("72"), 72)
        self.assertEqual(parse_spoken_integer("the answer is 106."), 106)

    def test_parses_word_number_answers(self) -> None:
        self.assertEqual(parse_spoken_integer("seventy two"), 72)
        self.assertEqual(parse_spoken_integer("seventy-two"), 72)
        self.assertEqual(parse_spoken_integer("one hundred six"), 106)
        self.assertEqual(parse_spoken_integer("1 hundred 6"), 106)

    def test_parses_negative_answers(self) -> None:
        self.assertEqual(parse_spoken_integer("minus five"), -5)
        self.assertEqual(parse_spoken_integer("-5"), -5)

    def test_rejects_non_numeric_transcripts(self) -> None:
        self.assertIsNone(parse_spoken_integer("I do not know"))
        self.assertIsNone(parse_spoken_integer("12 or 13"))


class VerbalZetamacScoringTests(unittest.TestCase):
    def test_computes_score_fields(self) -> None:
        question = Question("addition", 2, 2, "+", 4)
        attempts = [
            AnswerAttempt(question, "4", 4, True, 1.0),
            AnswerAttempt(question, "5", 5, False, 2.0),
            AnswerAttempt(question, "four", 4, True, 3.0),
        ]

        score = compute_score(
            duration_seconds=120,
            elapsed_seconds=10.0,
            questions_generated=150,
            attempts=attempts,
        )

        self.assertEqual(score.questions_seen, 3)
        self.assertEqual(score.correct_count, 2)
        self.assertEqual(score.incorrect_count, 1)
        self.assertEqual(score.correct_per_second, 0.2)
        self.assertEqual(score.accuracy, 0.6667)

    def test_empty_score_avoids_division_by_zero(self) -> None:
        score = compute_score(
            duration_seconds=120,
            elapsed_seconds=0.0,
            questions_generated=150,
            attempts=[],
        )

        self.assertEqual(score.questions_seen, 0)
        self.assertEqual(score.correct_per_second, 0.0)
        self.assertEqual(score.accuracy, 0.0)


class VerbalZetamacConfigTests(unittest.TestCase):
    def test_default_score_log_path_is_game_local(self) -> None:
        with (
            patch("games.verbal_zetamac.config.load_repo_env", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = Settings.from_env()

        self.assertEqual(
            settings.score_log_path,
            "games/verbal_zetamac/logs/scores.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
