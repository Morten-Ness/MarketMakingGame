from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Literal


USER_NAME = "User"
DEFAULT_BOT_COUNT = 2
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
    BotProfile(voice="af_bella", style="Aggressive Arbitrageur"),
    BotProfile(voice="am_adam", style="Conservative Market Maker"),
    BotProfile(voice="bf_emma", style="Tight Spreader"),
    BotProfile(voice="af_bella", style="Patient Relative-Value Trader"),
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
