from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


USER_NAME = "User"


@dataclass(frozen=True)
class BotProfile:
    voice: str
    style: str


BOT_PROFILES: dict[str, BotProfile] = {
    "Bot_Alpha": BotProfile(voice="af_bella", style="Aggressive Arbitrageur"),
    "Bot_Beta": BotProfile(voice="am_adam", style="Conservative Market Maker"),
    "Bot_Gamma": BotProfile(voice="bf_emma", style="Tight Spreader"),
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
