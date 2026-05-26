from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import env_bool, env_float, env_int, load_repo_env


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
    enable_audio_cues: bool
    audio_cue_volume: float
    kokoro_model_path: str
    kokoro_voices_path: str
    whisper_model: str
    vad_silence_ms: int
    voice_max_seconds: int
    max_turns: int
    end_on_trade: bool
    bot_count: int
    randomize_turn_order: bool
    allow_pass: bool
    min_tighten_increment: float
    scratchpad_log_path: str
    game_summary_log_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_repo_env()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            bot_temperature=env_float("BOT_TEMPERATURE", 0.2),
            use_gemini=env_bool("USE_GEMINI", True),
            enable_tts=env_bool("ENABLE_TTS", False),
            tts_backend=os.getenv("TTS_BACKEND", "auto"),
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
            max_turns=env_int("MAX_TURNS", 9),
            end_on_trade=env_bool("END_ON_TRADE", False),
            bot_count=env_int("BOT_COUNT", 2),
            randomize_turn_order=env_bool("RANDOMIZE_TURN_ORDER", True),
            allow_pass=env_bool("ALLOW_PASS", True),
            min_tighten_increment=env_float("MIN_TIGHTEN_INCREMENT", 0.5),
            scratchpad_log_path=os.getenv(
                "SCRATCHPAD_LOG_PATH",
                "games/dice_sum_market/logs/scratchpads.jsonl",
            ),
            game_summary_log_path=os.getenv(
                "GAME_SUMMARY_LOG_PATH",
                "games/dice_sum_market/logs/game_summaries.jsonl",
            ),
        )

