from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import env_bool, env_float, env_int, load_repo_env

from .models import GenerationSettings, VALID_OPERATIONS


def _env_operations(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    enable_tts: bool
    tts_backend: str
    tts_preroll_ms: int
    macos_say_voice: str
    enable_voice_input: bool
    enable_audio_cues: bool
    audio_cue_volume: float
    kokoro_model_path: str
    kokoro_voices_path: str
    whisper_model: str
    vad_silence_ms: int
    voice_max_seconds: int
    duration_seconds: int
    question_count: int
    operations: tuple[str, ...]
    addition_min: int
    addition_max: int
    multiplier_min: int
    multiplier_max: int
    multiplicand_min: int
    multiplicand_max: int
    score_log_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_repo_env()
        return cls(
            enable_tts=env_bool("ENABLE_TTS", False),
            tts_backend=os.getenv("TTS_BACKEND", "auto"),
            tts_preroll_ms=env_int("TTS_PREROLL_MS", 120),
            macos_say_voice=os.getenv("MACOS_SAY_VOICE", "Samantha"),
            enable_voice_input=env_bool("ENABLE_VOICE_INPUT", False),
            enable_audio_cues=env_bool("ENABLE_AUDIO_CUES", True),
            audio_cue_volume=env_float("AUDIO_CUE_VOLUME", 0.2),
            kokoro_model_path=os.getenv(
                "KOKORO_MODEL_PATH",
                "models/kokoro-v1.0.int8.onnx",
            ),
            kokoro_voices_path=os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin"),
            whisper_model=os.getenv("WHISPER_MODEL", "tiny.en"),
            vad_silence_ms=env_int("VAD_SILENCE_MS", 500),
            voice_max_seconds=env_int("VOICE_MAX_SECONDS", 30),
            duration_seconds=env_int("VERBAL_ZETAMAC_DURATION_SECONDS", 120),
            question_count=env_int("VERBAL_ZETAMAC_QUESTION_COUNT", 150),
            operations=_env_operations("VERBAL_ZETAMAC_OPERATIONS", VALID_OPERATIONS),
            addition_min=env_int("VERBAL_ZETAMAC_ADDITION_MIN", 2),
            addition_max=env_int("VERBAL_ZETAMAC_ADDITION_MAX", 100),
            multiplier_min=env_int("VERBAL_ZETAMAC_MULTIPLIER_MIN", 2),
            multiplier_max=env_int("VERBAL_ZETAMAC_MULTIPLIER_MAX", 12),
            multiplicand_min=env_int("VERBAL_ZETAMAC_MULTIPLICAND_MIN", 2),
            multiplicand_max=env_int("VERBAL_ZETAMAC_MULTIPLICAND_MAX", 100),
            score_log_path=os.getenv(
                "VERBAL_ZETAMAC_SCORE_LOG_PATH",
                "games/verbal_zetamac/logs/scores.jsonl",
            ),
        )

    @property
    def generation_settings(self) -> GenerationSettings:
        return GenerationSettings(
            operations=self.operations,
            addition_min=self.addition_min,
            addition_max=self.addition_max,
            multiplier_min=self.multiplier_min,
            multiplier_max=self.multiplier_max,
            multiplicand_min=self.multiplicand_min,
            multiplicand_max=self.multiplicand_max,
        )

    def game_settings_log(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "question_count": self.question_count,
            **self.generation_settings.as_log_dict(),
        }
