from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Literal


USER_NAME = "User"
DEFAULT_BOT_COUNT = 3
BOT_NAME_POOL = (
    "Bot_Alpha",
    "Bot_Beta",
    "Bot_Gamma",
    "Bot_Delta",
    "Bot_Epsilon",
)


@dataclass(frozen=True)
class BotProfile:
    voice: str
    style: str


BOT_PROFILE_POOL: tuple[BotProfile, ...] = (
    BotProfile(voice="af_bella", style="Aggressive Probability Trader"),
    BotProfile(voice="am_adam", style="Conservative Forecaster"),
    BotProfile(voice="bf_emma", style="Tight Spreader"),
    BotProfile(voice="af_bella", style="Skeptical Event Trader"),
    BotProfile(voice="am_adam", style="Fast Flow Trader"),
)


def build_bot_names(bot_count: int) -> tuple[str, ...]:
    names = list(BOT_NAME_POOL[:bot_count])
    if bot_count > len(names):
        names.extend(f"Bot_{index}" for index in range(len(names) + 1, bot_count + 1))
    return tuple(names)


def build_bot_profiles(bot_names: tuple[str, ...]) -> dict[str, BotProfile]:
    return {
        bot_name: profile
        for bot_name, profile in zip(bot_names, cycle(BOT_PROFILE_POOL))
    }


@dataclass(frozen=True)
class MarketSnapshot:
    market_id: str
    question: str
    slug: str | None
    event_id: str | None
    event_slug: str | None
    category: str | None
    description: str | None
    end_date: str | None
    yes_probability: float
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    volume: float | None
    fetched_at_utc: str

    @property
    def yes_probability_percent(self) -> float:
        return round(self.yes_probability * 100, 2)

    def public_dict(self) -> dict[str, object]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "slug": self.slug,
            "event_id": self.event_id,
            "event_slug": self.event_slug,
            "category": self.category,
            "description": self.description,
            "end_date": self.end_date,
        }

    def as_log_dict(self) -> dict[str, object]:
        return {
            **self.public_dict(),
            "yes_probability": self.yes_probability,
            "yes_probability_percent": self.yes_probability_percent,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "last_trade_price": self.last_trade_price,
            "volume": self.volume,
            "volume_1mo": self.volume,
            "fetched_at_utc": self.fetched_at_utc,
        }


@dataclass(frozen=True)
class Quote:
    owner: str
    bid: float
    offer: float

    def as_public_dict(self) -> dict[str, object]:
        return {"owner": self.owner, "bid": self.bid, "offer": self.offer}


@dataclass(frozen=True)
class Trade:
    buyer: str
    seller: str
    price: float
    quantity: int
    turn_index: int
    source: str

    def as_public_dict(self) -> dict[str, object]:
        return {
            "buyer": self.buyer,
            "seller": self.seller,
            "price": self.price,
            "quantity": self.quantity,
            "turn_index": self.turn_index,
            "source": self.source,
        }


CommandKind = Literal["quote", "pass", "hit", "lift", "unknown"]


@dataclass(frozen=True)
class ParsedCommand:
    kind: CommandKind
    raw_text: str
    bid: float | None = None
    offer: float | None = None
    price: float | None = None


@dataclass(frozen=True)
class ActionResult:
    accepted: bool
    message: str
    trade: Trade | None = None
    finished: bool = False
