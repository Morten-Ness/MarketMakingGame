from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # Allows the text-only fallback to run before dependency install.
    def load_dotenv() -> bool:
        return False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    bot_temperature: float
    use_gemini: bool
    enable_tts: bool
    tts_backend: str
    macos_say_voice: str
    enable_voice_input: bool
    kokoro_model_path: str
    kokoro_voices_path: str
    whisper_model: str
    vad_silence_ms: int
    voice_max_seconds: int
    max_turns: int
    end_on_trade: bool
    allow_pass: bool
    min_tighten_increment: float
    scratchpad_log_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        return cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            bot_temperature=_env_float("BOT_TEMPERATURE", 0.2),
            use_gemini=_env_bool("USE_GEMINI", True),
            enable_tts=_env_bool("ENABLE_TTS", False),
            tts_backend=os.getenv("TTS_BACKEND", "auto"),
            macos_say_voice=os.getenv("MACOS_SAY_VOICE", "Samantha"),
            enable_voice_input=_env_bool("ENABLE_VOICE_INPUT", False),
            kokoro_model_path=os.getenv(
                "KOKORO_MODEL_PATH",
                "models/kokoro-v1.0.int8.onnx",
            ),
            kokoro_voices_path=os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin"),
            whisper_model=os.getenv("WHISPER_MODEL", "tiny.en"),
            vad_silence_ms=_env_int("VAD_SILENCE_MS", 500),
            voice_max_seconds=_env_int("VOICE_MAX_SECONDS", 30),
            max_turns=_env_int("MAX_TURNS", 9),
            end_on_trade=_env_bool("END_ON_TRADE", False),
            allow_pass=_env_bool("ALLOW_PASS", True),
            min_tighten_increment=_env_float("MIN_TIGHTEN_INCREMENT", 0.5),
            scratchpad_log_path=os.getenv(
                "SCRATCHPAD_LOG_PATH",
                "logs/scratchpads.jsonl",
            ),
        )
