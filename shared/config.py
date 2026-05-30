from __future__ import annotations

import os
from dataclasses import dataclass

from .paths import repo_path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows the text-only fallback to run before dependency install.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


def load_repo_env() -> bool:
    return load_dotenv(repo_path(".env"))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class SharedRuntimeSettings:
    gemini_api_key: str | None
    gemini_model: str
    bot_temperature: float
    use_gemini: bool
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

    @classmethod
    def from_env(cls) -> "SharedRuntimeSettings":
        load_repo_env()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            bot_temperature=env_float("BOT_TEMPERATURE", 0.2),
            use_gemini=env_bool("USE_GEMINI", True),
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
        )
