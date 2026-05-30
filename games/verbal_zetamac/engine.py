from __future__ import annotations

import random
import re

from .models import AnswerAttempt, GenerationSettings, Question, Score


class QuestionGenerator:
    def __init__(
        self,
        settings: GenerationSettings,
        rng: random.Random | None = None,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._rng = rng if rng is not None else random.Random()

    def generate(self, count: int) -> list[Question]:
        if count < 0:
            raise ValueError("Question count cannot be negative.")
        return [self.generate_one() for _ in range(count)]

    def generate_one(self, operation: str | None = None) -> Question:
        operation = operation or self._rng.choice(self._settings.operations)
        if operation == "addition":
            return self._addition_question()
        if operation == "subtraction":
            return self._subtraction_question()
        if operation == "multiplication":
            return self._multiplication_question()
        if operation == "division":
            return self._division_question()
        raise ValueError(f"Unsupported operation: {operation}")

    def _addition_question(self) -> Question:
        left = self._rng.randint(self._settings.addition_min, self._settings.addition_max)
        right = self._rng.randint(self._settings.addition_min, self._settings.addition_max)
        return Question("addition", left, right, "+", left + right)

    def _subtraction_question(self) -> Question:
        first = self._rng.randint(self._settings.addition_min, self._settings.addition_max)
        second = self._rng.randint(self._settings.addition_min, self._settings.addition_max)
        total = first + second
        if self._rng.choice((True, False)):
            return Question("subtraction", total, first, "-", second)
        return Question("subtraction", total, second, "-", first)

    def _multiplication_question(self) -> Question:
        left = self._rng.randint(
            self._settings.multiplier_min,
            self._settings.multiplier_max,
        )
        right = self._rng.randint(
            self._settings.multiplicand_min,
            self._settings.multiplicand_max,
        )
        return Question("multiplication", left, right, "x", left * right)

    def _division_question(self) -> Question:
        first = self._rng.randint(
            self._settings.multiplier_min,
            self._settings.multiplier_max,
        )
        second = self._rng.randint(
            self._settings.multiplicand_min,
            self._settings.multiplicand_max,
        )
        product = first * second
        if self._rng.choice((True, False)):
            return Question("division", product, first, "/", second)
        return Question("division", product, second, "/", first)


def compute_score(
    *,
    duration_seconds: int,
    elapsed_seconds: float,
    questions_generated: int,
    attempts: list[AnswerAttempt],
) -> Score:
    questions_seen = len(attempts)
    correct_count = sum(1 for attempt in attempts if attempt.correct)
    incorrect_count = questions_seen - correct_count
    elapsed = max(0.0, elapsed_seconds)
    return Score(
        duration_seconds=duration_seconds,
        elapsed_seconds=round(elapsed, 3),
        questions_generated=questions_generated,
        questions_seen=questions_seen,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        correct_per_second=round(correct_count / elapsed, 4) if elapsed else 0.0,
        accuracy=round(correct_count / questions_seen, 4) if questions_seen else 0.0,
    )


def parse_spoken_integer(text: str) -> int | None:
    tokens = _tokenize_answer(text)
    segments: list[list[str]] = []
    current: list[str] = []

    for token in tokens:
        if _is_numberish_token(token):
            current.append(token)
        elif current:
            segments.append(current)
            current = []

    if current:
        segments.append(current)

    candidates = [
        value
        for value in (_parse_number_tokens(segment) for segment in segments)
        if value is not None
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


_UNITS = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALES = {"hundred", "thousand"}
_SIGNS = {"minus", "negative"}
_OPTIONAL_NUMBER_WORDS = {"and"}


def _tokenize_answer(text: str) -> list[str]:
    cleaned = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text.lower())
    return re.findall(r"[+-]?\d+|[a-z]+", cleaned)


def _is_numberish_token(token: str) -> bool:
    return (
        _is_integer_token(token)
        or token in _UNITS
        or token in _TENS
        or token in _SCALES
        or token in _SIGNS
        or token in _OPTIONAL_NUMBER_WORDS
    )


def _parse_number_tokens(tokens: list[str]) -> int | None:
    meaningful_tokens = [token for token in tokens if token != "and"]
    if not meaningful_tokens or all(token in _SIGNS for token in meaningful_tokens):
        return None

    if all(_is_integer_token(token) for token in meaningful_tokens):
        if len(meaningful_tokens) != 1:
            return None
        return int(meaningful_tokens[0])

    total = 0
    current = 0
    sign = 1
    seen_number = False
    sign_allowed = True

    for token in tokens:
        if token == "and":
            continue

        if token in _SIGNS:
            if not sign_allowed or seen_number:
                return None
            sign = -1
            sign_allowed = False
            continue

        sign_allowed = False
        if _is_integer_token(token):
            numeric_value = int(token)
            if numeric_value < 0:
                if seen_number:
                    return None
                sign = -1
                numeric_value = abs(numeric_value)
            current += numeric_value
            seen_number = True
            continue

        if token in _UNITS:
            current += _UNITS[token]
            seen_number = True
            continue

        if token in _TENS:
            current += _TENS[token]
            seen_number = True
            continue

        if token == "hundred":
            current = max(current, 1) * 100
            seen_number = True
            continue

        if token == "thousand":
            total += max(current, 1) * 1000
            current = 0
            seen_number = True
            continue

        return None

    if not seen_number:
        return None
    return sign * (total + current)


def _is_integer_token(token: str) -> bool:
    return re.fullmatch(r"[+-]?\d+", token) is not None
