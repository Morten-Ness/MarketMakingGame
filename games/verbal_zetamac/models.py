from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_OPERATIONS = ("addition", "subtraction", "multiplication", "division")


@dataclass(frozen=True)
class GenerationSettings:
    operations: tuple[str, ...] = VALID_OPERATIONS
    addition_min: int = 2
    addition_max: int = 100
    multiplier_min: int = 2
    multiplier_max: int = 12
    multiplicand_min: int = 2
    multiplicand_max: int = 100

    def validate(self) -> None:
        if not self.operations:
            raise ValueError("VERBAL_ZETAMAC_OPERATIONS must include at least one operation.")
        invalid_operations = sorted(set(self.operations) - set(VALID_OPERATIONS))
        if invalid_operations:
            joined = ", ".join(invalid_operations)
            raise ValueError(f"Unsupported verbal Zetamac operations: {joined}.")
        _validate_range("addition", self.addition_min, self.addition_max)
        _validate_range("multiplier", self.multiplier_min, self.multiplier_max)
        _validate_range("multiplicand", self.multiplicand_min, self.multiplicand_max)

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "operations": list(self.operations),
            "addition_min": self.addition_min,
            "addition_max": self.addition_max,
            "multiplier_min": self.multiplier_min,
            "multiplier_max": self.multiplier_max,
            "multiplicand_min": self.multiplicand_min,
            "multiplicand_max": self.multiplicand_max,
        }


@dataclass(frozen=True)
class Question:
    operation: str
    left: int
    right: int
    operator: str
    answer: int

    @property
    def display_text(self) -> str:
        return f"{self.left} {self.operator} {self.right}"

    @property
    def spoken_text(self) -> str:
        operator_words = {
            "+": "plus",
            "-": "minus",
            "x": "times",
            "/": "divided by",
        }
        return f"{self.left} {operator_words[self.operator]} {self.right}"

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "prompt": self.display_text,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class AnswerAttempt:
    question: Question
    transcript: str
    parsed_answer: int | None
    correct: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class Score:
    duration_seconds: int
    elapsed_seconds: float
    questions_generated: int
    questions_seen: int
    correct_count: int
    incorrect_count: int
    correct_per_second: float
    accuracy: float

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "questions_generated": self.questions_generated,
            "questions_seen": self.questions_seen,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "correct_per_second": self.correct_per_second,
            "accuracy": self.accuracy,
        }


def _validate_range(name: str, minimum: int, maximum: int) -> None:
    if minimum > maximum:
        raise ValueError(f"{name} minimum cannot be greater than maximum.")

