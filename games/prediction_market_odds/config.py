from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import env_bool, env_float, env_int, load_repo_env


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    auto_next_game: bool
    bot_count: int
    randomize_turn_order: bool
    allow_pass: bool
    min_tighten_increment: float
    gamma_base_url: str
    market_fetch_limit: int
    market_fetch_pages: int
    market_max_offset: int
    min_liquidity: float
    min_volume: float
    allowed_categories: tuple[str, ...]
    cache_path: str
    scratchpad_log_path: str
    game_summary_log_path: str
    played_markets_log_path: str

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
            max_turns=env_int("PREDICTION_MARKET_MAX_TURNS", 8),
            end_on_trade=env_bool("PREDICTION_MARKET_END_ON_TRADE", False),
            auto_next_game=env_bool("PREDICTION_MARKET_AUTO_NEXT_GAME", False),
            bot_count=env_int("PREDICTION_MARKET_BOT_COUNT", 3),
            randomize_turn_order=env_bool("PREDICTION_MARKET_RANDOMIZE_TURN_ORDER", True),
            allow_pass=env_bool("PREDICTION_MARKET_ALLOW_PASS", False),
            min_tighten_increment=env_float("PREDICTION_MARKET_MIN_TIGHTEN_INCREMENT", 1.0),
            gamma_base_url=os.getenv(
                "POLYMARKET_GAMMA_BASE_URL",
                "https://gamma-api.polymarket.com",
            ).rstrip("/"),
            market_fetch_limit=env_int("POLYMARKET_MARKET_FETCH_LIMIT", 100),
            market_fetch_pages=env_int("POLYMARKET_MARKET_FETCH_PAGES", 3),
            market_max_offset=env_int("POLYMARKET_MARKET_MAX_OFFSET", 500),
            min_liquidity=env_float("POLYMARKET_MIN_LIQUIDITY", 0.0),
            min_volume=env_float("POLYMARKET_MIN_VOLUME", 0.0),
            allowed_categories=_env_csv(
                "POLYMARKET_ALLOWED_CATEGORIES",
                ("AI", "Geopolitics", "Tech", "Finance", "Science"),
            ),
            cache_path=os.getenv(
                "PREDICTION_MARKET_CACHE_PATH",
                "games/prediction_market_odds/data/market_cache.json",
            ),
            scratchpad_log_path=os.getenv(
                "PREDICTION_MARKET_SCRATCHPAD_LOG_PATH",
                "games/prediction_market_odds/logs/scratchpads.jsonl",
            ),
            game_summary_log_path=os.getenv(
                "PREDICTION_MARKET_GAME_SUMMARY_LOG_PATH",
                "games/prediction_market_odds/logs/game_summaries.jsonl",
            ),
            played_markets_log_path=os.getenv(
                "PREDICTION_MARKET_PLAYED_MARKETS_LOG_PATH",
                "games/prediction_market_odds/logs/played_markets.jsonl",
            ),
        )
