from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from shared.llm import JsonLlmClient
from shared.logging import JsonlLogger
from shared.paths import resolve_repo_path

from .models import Paper


SUITABLE = "Suitable"
PARTIALLY_SUITABLE = "Partially suitable"
NOT_SUITABLE = "Not suitable"
SUITABLE_VALUES = {SUITABLE, PARTIALLY_SUITABLE}
VALID_SUITABILITY = {SUITABLE, PARTIALLY_SUITABLE, NOT_SUITABLE}

EXERCISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "suitability",
        "neutral_setup",
        "prediction_question",
        "options",
        "reasoning_prompt",
        "reveal",
        "not_suitable_reason",
        "alternative_reading_exercise",
    ],
    "properties": {
        "suitability": {
            "type": "string",
            "enum": [SUITABLE, PARTIALLY_SUITABLE, NOT_SUITABLE],
        },
        "neutral_setup": {"type": "string"},
        "prediction_question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "text"],
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
        "reasoning_prompt": {"type": "string"},
        "reveal": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "correct_option",
                "result_summary",
                "correctness_explanation",
                "learning_note",
                "caveats",
            ],
            "properties": {
                "correct_option": {"type": "string"},
                "result_summary": {"type": "string"},
                "correctness_explanation": {"type": "string"},
                "learning_note": {"type": "string"},
                "caveats": {"type": "string"},
            },
        },
        "not_suitable_reason": {"type": "string"},
        "alternative_reading_exercise": {"type": "string"},
    },
}

_ANSWER_INTERPRETATION_INSTRUCTIONS = """
You interpret a user's prediction-game answer after they have seen only the neutral
setup, prediction question, and options. Return JSON only.

Extract exactly one firm option label when the user's answer clearly commits to one.
Accept natural wording such as "my final prediction is A" or "I think the RL option
wins, so D". If the user is still undecided or names multiple options without a final
choice, return an empty parsed_option.

When a firm option is present, write a concise reasoning_assessment for the user. It
will be shown only after the reveal, so it may refer to the correct result. Explain
which part of the user's reasoning was directionally right or wrong in comparison to
the paper result. Do not add a new dialogue turn or ask follow-up questions.
""".strip()


def _answer_interpretation_schema(valid_options: set[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "parsed_option",
            "reasoning_summary",
            "reasoning_assessment",
            "interpretation_notes",
        ],
        "properties": {
            "parsed_option": {
                "type": "string",
                "enum": ["", *sorted(valid_options)],
            },
            "reasoning_summary": {"type": "string"},
            "reasoning_assessment": {"type": "string"},
            "interpretation_notes": {"type": "string"},
        },
    }


@dataclass(frozen=True)
class PredictionOption:
    label: str
    text: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PredictionOption":
        label = _required_text(payload, "label").upper().strip(".:")
        text = _required_text(payload, "text")
        if not re.fullmatch(r"[A-Z]", label):
            raise ValueError(f"Invalid prediction option label: {label!r}")
        return cls(label=label, text=text)

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "text": self.text}


@dataclass(frozen=True)
class PredictionReveal:
    correct_option: str
    result_summary: str
    correctness_explanation: str
    learning_note: str
    caveats: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PredictionReveal":
        correct_option = (
            _required_text(payload, "correct_option", allow_empty=True).upper().strip(".:")
        )
        if correct_option and not re.fullmatch(r"[A-Z]", correct_option):
            raise ValueError(f"Invalid correct option: {correct_option!r}")
        return cls(
            correct_option=correct_option,
            result_summary=_required_text(payload, "result_summary", allow_empty=True),
            correctness_explanation=_required_text(
                payload,
                "correctness_explanation",
                allow_empty=True,
            ),
            learning_note=_required_text(payload, "learning_note", allow_empty=True),
            caveats=_required_text(payload, "caveats", allow_empty=True),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "correct_option": self.correct_option,
            "result_summary": self.result_summary,
            "correctness_explanation": self.correctness_explanation,
            "learning_note": self.learning_note,
            "caveats": self.caveats,
        }


@dataclass(frozen=True)
class PredictionExercise:
    suitability: str
    neutral_setup: str
    prediction_question: str
    options: list[PredictionOption]
    reasoning_prompt: str
    reveal: PredictionReveal
    not_suitable_reason: str
    alternative_reading_exercise: str

    @property
    def is_suitable(self) -> bool:
        return self.suitability in SUITABLE_VALUES

    @property
    def option_labels(self) -> set[str]:
        return {option.label for option in self.options}

    @classmethod
    def from_payload(cls, payload: Any) -> "PredictionExercise":
        if not isinstance(payload, dict):
            raise ValueError("Prediction exercise payload must be a JSON object.")

        suitability = _required_text(payload, "suitability")
        if suitability not in VALID_SUITABILITY:
            raise ValueError(f"Invalid suitability: {suitability!r}")

        options_payload = payload.get("options")
        if not isinstance(options_payload, list):
            raise ValueError("Prediction exercise options must be a list.")
        options = [
            PredictionOption.from_payload(option)
            for option in options_payload
            if isinstance(option, dict)
        ]

        reveal_payload = payload.get("reveal")
        if not isinstance(reveal_payload, dict):
            raise ValueError("Prediction exercise reveal must be an object.")
        reveal = PredictionReveal.from_payload(reveal_payload)

        exercise = cls(
            suitability=suitability,
            neutral_setup=_required_text(payload, "neutral_setup", allow_empty=True),
            prediction_question=_required_text(
                payload,
                "prediction_question",
                allow_empty=True,
            ),
            options=options,
            reasoning_prompt=_required_text(payload, "reasoning_prompt", allow_empty=True),
            reveal=reveal,
            not_suitable_reason=_required_text(
                payload,
                "not_suitable_reason",
                allow_empty=True,
            ),
            alternative_reading_exercise=_required_text(
                payload,
                "alternative_reading_exercise",
                allow_empty=True,
            ),
        )
        exercise.validate()
        return exercise

    def validate(self) -> None:
        if self.is_suitable:
            if not self.neutral_setup:
                raise ValueError("Suitable exercises require neutral_setup.")
            if not self.prediction_question:
                raise ValueError("Suitable exercises require prediction_question.")
            if not self.options:
                raise ValueError("Suitable exercises require options.")
            if self.reveal.correct_option not in self.option_labels:
                raise ValueError("Reveal correct_option must match an option label.")
            if not self.reveal.result_summary:
                raise ValueError("Suitable exercises require reveal.result_summary.")
            if not self.reveal.correctness_explanation:
                raise ValueError(
                    "Suitable exercises require reveal.correctness_explanation."
                )
        elif not self.not_suitable_reason:
            raise ValueError("Not suitable exercises require not_suitable_reason.")

    def visible_dict(self) -> dict[str, Any]:
        return {
            "suitability": self.suitability,
            "neutral_setup": self.neutral_setup,
            "prediction_question": self.prediction_question,
            "options": [option.as_dict() for option in self.options],
            "reasoning_prompt": self.reasoning_prompt,
            "not_suitable_reason": self.not_suitable_reason,
            "alternative_reading_exercise": self.alternative_reading_exercise,
        }


@dataclass(frozen=True)
class GameOutcome:
    user_answer_text: str | None
    parsed_option: str | None
    correct: bool | None
    revealed: bool
    answer_parser: str | None = None
    reasoning_summary: str | None = None
    reasoning_assessment: str | None = None


@dataclass(frozen=True)
class UserAnswerInterpretation:
    parsed_option: str | None
    reasoning_summary: str
    reasoning_assessment: str
    interpretation_notes: str

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        valid_options: set[str],
    ) -> "UserAnswerInterpretation":
        if not isinstance(payload, dict):
            raise ValueError("User answer interpretation payload must be a JSON object.")

        parsed_option = _required_text(payload, "parsed_option", allow_empty=True)
        parsed_option = parsed_option.upper().strip(".:")
        if parsed_option and parsed_option not in valid_options:
            raise ValueError(f"Invalid interpreted option: {parsed_option!r}")

        return cls(
            parsed_option=parsed_option or None,
            reasoning_summary=_required_text(
                payload,
                "reasoning_summary",
                allow_empty=True,
            ),
            reasoning_assessment=_required_text(
                payload,
                "reasoning_assessment",
                allow_empty=True,
            ),
            interpretation_notes=_required_text(
                payload,
                "interpretation_notes",
                allow_empty=True,
            ),
        )


def generate_prediction_exercise(
    *,
    llm_client: JsonLlmClient,
    prompt_text: str,
    paper: Paper,
    raw_text: str,
    max_text_chars: int,
) -> PredictionExercise:
    payload = build_generation_payload(
        paper=paper,
        raw_text=raw_text,
        max_text_chars=max_text_chars,
    )
    response = llm_client.generate_json(prompt_text, payload, schema=EXERCISE_SCHEMA)
    return PredictionExercise.from_payload(response)


def build_generation_payload(
    *,
    paper: Paper,
    raw_text: str,
    max_text_chars: int,
) -> dict[str, Any]:
    return {
        "paper": {
            "paper_id": paper.paper_id,
            "corpus_id": paper.corpus_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "tldr": paper.tldr,
            "url": paper.url,
        },
        "raw_text": truncate_text(raw_text, max_text_chars),
    }


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n\n[... middle of paper omitted for length ...]\n\n"
    if max_chars <= len(marker):
        return text[:max_chars]

    head_chars = int((max_chars - len(marker)) * 0.45)
    tail_chars = max_chars - len(marker) - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def render_visible_exercise(exercise: PredictionExercise) -> str:
    if not exercise.is_suitable:
        return "\n".join(
            [
                f"Suitability: {exercise.suitability}",
                "",
                "Reason:",
                exercise.not_suitable_reason,
                "",
                "Alternative reading exercise:",
                exercise.alternative_reading_exercise,
            ]
        ).strip()

    lines = [
        f"Suitability: {exercise.suitability}",
        "",
        "Neutral setup:",
        exercise.neutral_setup,
        "",
        "Prediction question:",
        exercise.prediction_question,
        "",
        "Options:",
    ]
    for option in exercise.options:
        lines.append(f"{option.label}. {option.text}")
    lines.extend(
        [
            "",
            "Before reveal:",
            exercise.reasoning_prompt
            or "Give your firm prediction, optionally with your reasoning.",
        ]
    )
    return "\n".join(lines)


def parse_user_option(text: str, valid_options: set[str]) -> str | None:
    options = "".join(sorted(valid_options))
    if not options:
        return None

    start_match = re.match(
        rf"^\s*(?:option\s+)?([{options}])(?:[\s\).,:;-]|$)",
        text,
        flags=re.I,
    )
    if start_match:
        return start_match.group(1).upper()

    intent_match = re.search(
        rf"\b(?:choose|pick|select|answer|predict|prediction|option|i think|i expect|my answer)\s+(?:is\s+|as\s+|would\s+be\s+|:?\s*)(?:option\s+)?([{options}])\b",
        text,
        flags=re.I,
    )
    if intent_match:
        return intent_match.group(1).upper()

    return None


def parse_single_option_answer(text: str, valid_options: set[str]) -> str | None:
    options = "".join(sorted(valid_options))
    if not options:
        return None

    match = re.fullmatch(
        rf"\s*(?:option\s+)?([{options}])(?:[\s\).,:;-]*)\s*",
        text,
        flags=re.I,
    )
    if not match:
        return None
    return match.group(1).upper()


def interpret_user_answer(
    *,
    llm_client: JsonLlmClient,
    exercise: PredictionExercise,
    answer_text: str,
) -> UserAnswerInterpretation:
    payload = {
        "task": (
            "Extract the user's firm option prediction from their answer, then assess "
            "their reasoning against the hidden paper result. If no single firm option "
            "can be inferred, return an empty parsed_option and do not reveal result "
            "details in the assessment."
        ),
        "visible_exercise": exercise.visible_dict(),
        "hidden_reveal": exercise.reveal.as_dict(),
        "user_answer": answer_text,
        "valid_options": sorted(exercise.option_labels),
    }
    response = llm_client.generate_json(
        _ANSWER_INTERPRETATION_INSTRUCTIONS,
        payload,
        schema=_answer_interpretation_schema(exercise.option_labels),
    )
    return UserAnswerInterpretation.from_payload(response, exercise.option_labels)


def is_prediction_correct(user_option: str, correct_option: str) -> bool:
    return user_option.upper() == correct_option.upper()


def run_prediction_game(
    *,
    exercise: PredictionExercise,
    paper: Paper,
    model_name: str,
    log_path: str,
    llm_client: JsonLlmClient | None = None,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> GameOutcome:
    logger = PredictionGameLogger(log_path)
    print_func("")
    print_func("PREDICTION EXERCISE")
    print_func(render_visible_exercise(exercise))

    if not exercise.is_suitable:
        logger.write(
            paper=paper,
            model_name=model_name,
            exercise=exercise,
            outcome=GameOutcome(
                user_answer_text=None,
                parsed_option=None,
                correct=None,
                revealed=False,
            ),
        )
        return GameOutcome(None, None, None, False)

    while True:
        answer_text = input_func("Your firm prediction: ").strip()
        reasoning_summary = ""
        reasoning_assessment = ""
        answer_parser = "single_option"
        parsed_option = parse_single_option_answer(answer_text, exercise.option_labels)

        if answer_text and not parsed_option and llm_client is not None:
            try:
                interpretation = interpret_user_answer(
                    llm_client=llm_client,
                    exercise=exercise,
                    answer_text=answer_text,
                )
            except Exception as exc:
                print_func(f"Could not interpret that answer with the model: {exc}")
                interpretation = UserAnswerInterpretation(
                    parsed_option=None,
                    reasoning_summary="",
                    reasoning_assessment="",
                    interpretation_notes="model interpretation failed",
                )
            parsed_option = interpretation.parsed_option
            reasoning_summary = interpretation.reasoning_summary
            reasoning_assessment = interpretation.reasoning_assessment
            answer_parser = "llm"

        if not parsed_option and llm_client is None:
            parsed_option = parse_user_option(answer_text, exercise.option_labels)
            answer_parser = "regex"

        if parsed_option:
            break
        print_func("Please answer with a clear option label, such as A or B.")

    correct = is_prediction_correct(parsed_option, exercise.reveal.correct_option)
    print_func("")
    print_func("REVEAL")
    print_func(f"Correct answer: {exercise.reveal.correct_option}")
    print_func("Your prediction was correct." if correct else "Your prediction was not correct.")
    print_func("")
    print_func("Relevant result:")
    print_func(exercise.reveal.result_summary)
    print_func("")
    print_func("Explanation:")
    print_func(exercise.reveal.correctness_explanation)
    if reasoning_assessment:
        print_func("")
        print_func("Your reasoning:")
        print_func(reasoning_assessment)
    if exercise.reveal.learning_note:
        print_func("")
        print_func("What to learn:")
        print_func(exercise.reveal.learning_note)
    if exercise.reveal.caveats:
        print_func("")
        print_func("Caveats:")
        print_func(exercise.reveal.caveats)

    outcome = GameOutcome(
        user_answer_text=answer_text,
        parsed_option=parsed_option,
        correct=correct,
        revealed=True,
        answer_parser=answer_parser,
        reasoning_summary=reasoning_summary,
        reasoning_assessment=reasoning_assessment,
    )
    logger.write(
        paper=paper,
        model_name=model_name,
        exercise=exercise,
        outcome=outcome,
    )
    return outcome


class PredictionGameLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(
        self,
        *,
        paper: Paper,
        model_name: str,
        exercise: PredictionExercise,
        outcome: GameOutcome,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp_utc": _utc_now(),
            "model": model_name,
            "paper": {
                "paperId": paper.paper_id,
                "corpusId": paper.corpus_id,
                "title": paper.title,
                "year": paper.year,
                "url": paper.url,
                "pdfPath": paper.pdf_path,
            },
            "exercise": exercise.visible_dict(),
            "userAnswerText": outcome.user_answer_text,
            "parsedOption": outcome.parsed_option,
            "correct": outcome.correct,
            "revealed": outcome.revealed,
            "answerParser": outcome.answer_parser,
            "reasoningSummary": outcome.reasoning_summary,
            "reasoningAssessment": outcome.reasoning_assessment,
        }
        if outcome.revealed:
            payload["reveal"] = exercise.reveal.as_dict()
        self._logger.write(payload)


def read_prompt(path: str) -> str:
    return resolve_repo_path(path).read_text(encoding="utf-8")


def read_raw_text_for_paper(paper: Paper, raw_text_dir: str) -> str:
    if not paper.pdf_path:
        raise ValueError(f"Paper has no PDF path: {paper.title}")
    pdf_path = resolve_repo_path(paper.pdf_path)
    raw_text_path = resolve_repo_path(raw_text_dir) / f"{pdf_path.stem}.txt"
    return raw_text_path.read_text(encoding="utf-8")


def _required_text(
    payload: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Prediction exercise field {key!r} must be a string.")
    text = value.strip()
    if not allow_empty and not text:
        raise ValueError(f"Prediction exercise field {key!r} cannot be empty.")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
