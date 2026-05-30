from __future__ import annotations

import unittest
from unittest.mock import patch

from shared.audio import MacOSSaySpeaker, normalize_transcribed_numbers


class AudioTranscriptNormalizationTests(unittest.TestCase):
    def test_converts_single_digit_words_to_digits(self) -> None:
        self.assertEqual(normalize_transcribed_numbers("five at 30."), "5 at 30.")
        self.assertEqual(normalize_transcribed_numbers("Six at 30."), "6 at 30.")
        self.assertEqual(
            normalize_transcribed_numbers("I make seven at 30."),
            "I make 7 at 30.",
        )

    def test_preserves_words_that_are_not_single_digits(self) -> None:
        self.assertEqual(
            normalize_transcribed_numbers("lift the thirty offer"),
            "lift the thirty offer",
        )


class MacOSSaySpeakerTests(unittest.TestCase):
    def test_prefixes_silence_control_when_preroll_enabled(self) -> None:
        speaker = MacOSSaySpeaker("Samantha", preroll_ms=120)

        self.assertEqual(
            speaker._prepare_say_text("6 plus 7"),
            "[[slnc 120]] 6 plus 7",
        )

    def test_omits_silence_control_when_preroll_disabled(self) -> None:
        speaker = MacOSSaySpeaker("Samantha", preroll_ms=0)

        self.assertEqual(speaker._prepare_say_text("6 plus 7"), "6 plus 7")

    def test_speak_passes_preroll_text_to_say(self) -> None:
        speaker = MacOSSaySpeaker("Samantha", preroll_ms=80)

        with (
            patch("builtins.print"),
            patch("shared.audio.subprocess.run") as run,
        ):
            speaker.speak("Question", "6 plus 7")

        self.assertEqual(
            run.call_args.args[0],
            ["say", "-v", "Samantha", "[[slnc 80]] 6 plus 7"],
        )


if __name__ == "__main__":
    unittest.main()
