from __future__ import annotations

import unittest

from shared.audio import normalize_transcribed_numbers


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


if __name__ == "__main__":
    unittest.main()

