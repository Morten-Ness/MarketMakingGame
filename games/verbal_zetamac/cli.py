from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timezone

from shared.audio import build_audio_cues, build_listener, build_speaker
from shared.logging import JsonlLogger

from .config import Settings
from .engine import QuestionGenerator, compute_score, parse_spoken_integer
from .models import AnswerAttempt, Score


class ScoreLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(self, score: Score, settings: Settings) -> None:
        self._logger.write(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **score.as_log_dict(),
                "settings": settings.game_settings_log(),
            }
        )


def main() -> int:
    settings = Settings.from_env()
    try:
        settings.generation_settings.validate()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    speaker = build_speaker(settings)
    cue_player = build_audio_cues(settings)
    listener = build_listener(settings, cue_player)
    score_logger = ScoreLogger(settings.score_log_path)

    _print_intro(settings, speaker, listener, cue_player)
    score = _run_game(settings, speaker, listener, cue_player)
    score_logger.write(score, settings)
    _print_summary(score, settings, speaker)
    return 0


def _print_intro(settings: Settings, speaker, listener, cue_player) -> None:
    print("Verbal Zetamac")
    print(f"Duration: {settings.duration_seconds} seconds")
    print(f"Questions generated upfront: {settings.question_count}")
    print(f"Operations: {', '.join(settings.operations)}")
    print(
        "Addition/Subtraction range: "
        f"{settings.addition_min}-{settings.addition_max}"
    )
    print(
        "Multiplication/Division range: "
        f"{settings.multiplier_min}-{settings.multiplier_max} x "
        f"{settings.multiplicand_min}-{settings.multiplicand_max}"
    )
    print(f"Python: {sys.executable}")
    print(speaker.status)
    print(listener.status)
    print(cue_player.status)
    print(f"Score log: {settings.score_log_path}")
    print()
    speaker.speak(
        "Exchange",
        (
            "Verbal Zetamac starting. "
            f"You have {settings.duration_seconds} seconds. "
            "Answer each arithmetic question out loud."
        ),
    )


def _run_game(settings: Settings, speaker, listener, cue_player) -> Score:
    generator = QuestionGenerator(settings.generation_settings, random.Random())
    questions = generator.generate(settings.question_count)
    attempts: list[AnswerAttempt] = []

    print("Generating questions complete. Timer starts now.")
    print()

    started_at = time.monotonic()
    for index, question in enumerate(questions, start=1):
        if time.monotonic() - started_at >= settings.duration_seconds:
            break

        print(f"[{index}] {question.display_text} = ?")
        speaker.speak("Question", question.spoken_text)
        cue_player.user_turn_started()
        transcript = listener.listen()
        parsed_answer = parse_spoken_integer(transcript)
        correct = parsed_answer == question.answer
        elapsed_seconds = time.monotonic() - started_at
        attempts.append(
            AnswerAttempt(
                question=question,
                transcript=transcript,
                parsed_answer=parsed_answer,
                correct=correct,
                elapsed_seconds=round(elapsed_seconds, 3),
            )
        )

        if correct:
            cue_player.command_accepted()
            print("Correct.")
        else:
            cue_player.command_rejected()
            parsed_text = "unparsed" if parsed_answer is None else str(parsed_answer)
            print(f"Incorrect. Heard {parsed_text}; answer was {question.answer}.")
        print()

    elapsed_seconds = time.monotonic() - started_at
    return compute_score(
        duration_seconds=settings.duration_seconds,
        elapsed_seconds=elapsed_seconds,
        questions_generated=len(questions),
        attempts=attempts,
    )


def _print_summary(score: Score, settings: Settings, speaker) -> None:
    print("RESULTS")
    print(f"Questions seen: {score.questions_seen}")
    print(f"Correct: {score.correct_count}")
    print(f"Incorrect: {score.incorrect_count}")
    print(f"Accuracy: {score.accuracy:.1%}")
    print(f"Correct per second: {score.correct_per_second:g}")
    print(f"Elapsed: {score.elapsed_seconds:g}s")
    print(f"Score log: {settings.score_log_path}")
    speaker.speak(
        "Exchange",
        (
            "Game over. "
            f"You answered {score.correct_count} out of "
            f"{score.questions_seen} questions correctly."
        ),
    )

