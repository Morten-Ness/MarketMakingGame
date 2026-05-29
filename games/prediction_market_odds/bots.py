from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from shared.llm import GeminiClient

from .engine import format_probability
from .models import BotProfile


@dataclass(frozen=True)
class BotDecision:
    scratchpad: str
    verbal_action: str


class BotClient(Protocol):
    @property
    def status(self) -> str:
        ...

    def decide(
        self,
        participant: str,
        profile: BotProfile,
        public_state: dict[str, object],
    ) -> BotDecision:
        ...


class HeuristicBotClient:
    """Offline fallback that makes broad no-research probability markets."""

    status = "Bot brain: local heuristic fallback (programmatic)."

    def decide(
        self,
        participant: str,
        profile: BotProfile,
        public_state: dict[str, object],
    ) -> BotDecision:
        market = public_state.get("market", {})
        question = str(market.get("question", "")) if isinstance(market, dict) else ""
        category = str(market.get("category", "")) if isinstance(market, dict) else ""
        fair_value = _heuristic_probability(question, category, participant)
        best_bid = public_state.get("best_bid")
        best_offer = public_state.get("best_offer")
        trading_rules = public_state.get("trading_rules", {})
        allow_pass = (
            bool(trading_rules.get("allow_pass", False))
            if isinstance(trading_rules, dict)
            else False
        )
        min_tighten_increment = (
            float(trading_rules.get("min_tighten_increment", 1.0))
            if isinstance(trading_rules, dict)
            else 1.0
        )

        edge = 7.0
        half_width = 12.0
        if "Aggressive" in profile.style:
            edge = 5.0
            half_width = 10.0
        elif "Conservative" in profile.style:
            edge = 10.0
            half_width = 16.0
        elif "Tight" in profile.style:
            edge = 6.0
            half_width = 8.0

        if isinstance(best_offer, dict):
            offer = float(best_offer["offer"])
            if offer <= fair_value - edge:
                return BotDecision(
                    scratchpad=(
                        f"No external research. My text-only fair value is "
                        f"{fair_value:.0f}%. Best offer is {offer:.0f}%, which "
                        f"clears my buy edge of {edge:.0f} points."
                    ),
                    verbal_action=f"I lift the {format_probability(offer)} offer.",
                )

        if isinstance(best_bid, dict):
            bid = float(best_bid["bid"])
            if bid >= fair_value + edge:
                return BotDecision(
                    scratchpad=(
                        f"No external research. My text-only fair value is "
                        f"{fair_value:.0f}%. Best bid is {bid:.0f}%, which "
                        f"clears my sell edge of {edge:.0f} points."
                    ),
                    verbal_action=f"I hit the {format_probability(bid)} bid.",
                )

        bid = max(0.0, round(fair_value - half_width))
        offer = min(100.0, round(fair_value + half_width))
        if bid >= offer:
            offer = min(100.0, bid + 1.0)

        if isinstance(best_bid, dict) and isinstance(best_offer, dict):
            current_bid = float(best_bid["bid"])
            current_offer = float(best_offer["offer"])
            bid = max(bid, current_bid)
            offer = min(offer, current_offer)
            if not allow_pass and not _tightens_market_enough(
                bid,
                offer,
                current_bid,
                current_offer,
                min_tighten_increment,
            ):
                bid, offer = _tighten_market(
                    bid,
                    offer,
                    current_bid,
                    current_offer,
                    min_tighten_increment,
                    prefer_bid=fair_value >= (current_bid + current_offer) / 2,
                )
            if bid >= offer:
                if not allow_pass:
                    return _forced_trade_decision(fair_value, current_bid, current_offer)
                return BotDecision(
                    scratchpad=(
                        f"No external research. My fair value is {fair_value:.0f}%, "
                        "and the current market is too tight to improve without trading."
                    ),
                    verbal_action="I pass.",
                )

        return BotDecision(
            scratchpad=(
                f"No external research. I estimate fair value around {fair_value:.0f}% "
                f"from the question text and quote a {profile.style.lower()} spread."
            ),
            verbal_action=f"I make {format_probability(bid)} at {format_probability(offer)}.",
        )


class GeminiBotClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.35) -> None:
        self._client = GeminiClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    @property
    def status(self) -> str:
        return f"Bot brain: {self._client.status}."

    def decide(
        self,
        participant: str,
        profile: BotProfile,
        public_state: dict[str, object],
    ) -> BotDecision:
        system_instruction = _system_instruction(participant, profile, public_state)
        user_prompt = {
            "public_state": public_state,
            "task": (
                "Choose exactly one legal trading action: make/update a market, "
                "hit a bid, lift an offer, or pass only if passing is legal."
            ),
        }
        return _parse_decision_payload(
            self._client.generate_json(system_instruction, user_prompt)
        )


def _system_instruction(
    participant: str,
    profile: BotProfile,
    public_state: dict[str, object],
) -> str:
    game = public_state.get("game", {})
    market = public_state.get("market", {})
    trading_rules = public_state.get("trading_rules", {})
    allow_pass = (
        bool(trading_rules.get("allow_pass", False))
        if isinstance(trading_rules, dict)
        else False
    )
    action_examples = [
        '- "I make 20% at 35%."',
        '- "I hit the 20% bid."',
        '- "I lift the 35% offer."',
    ]
    if allow_pass:
        action_examples.insert(1, '- "I pass."')
    else:
        action_examples.append("- Passing is not legal in this game.")

    return f"""
You are {participant}, a competitive interview-practice prediction-market bot.
Your trading style is: {profile.style}.

Authoritative game configuration:
{json.dumps(game, indent=2)}

Public market information:
{json.dumps(market, indent=2)}

Authoritative trading rules:
{json.dumps(trading_rules, indent=2)}

The hidden underlying is the current Polymarket implied Yes probability for this
market, in percentage points from 0 to 100.

You do not know the hidden probability. You must not browse, search, use tools,
claim current odds, or rely on external research. Reason only from the public
market title, description, category, end date, order book, trades, and action
tape shown in public_state.

Risk policy:
- Quote in whole percentage points.
- Only hit or lift when the public quote is meaningfully mispriced versus your
  own text-only probability estimate.
- Prefer making or tightening a two-sided probability market when edge is weak.
- If allow_pass is false, do not pass. You must hit, lift, or quote a market
  that tightens the current best market by at least min_tighten_increment on
  at least one side.
- If there is an existing best market, any new quote must be inside or equal to
  it: your bid must be at least the current best bid, and your offer must be no
  higher than the current best offer.

Return only strict JSON matching this exact schema:
{{
  "scratchpad": "brief private computation and rationale, never spoken aloud",
  "verbal_action": "one spoken trading command"
}}

The verbal_action must be one of:
{chr(10).join(action_examples)}
""".strip()


def _heuristic_probability(question: str, category: str, participant: str) -> float:
    text = f"{question} {category}".lower()
    value = 45.0
    if any(word in text for word in ("by ", "before ", "this year", "will ")):
        value -= 8.0
    if any(word in text for word in ("win", "election", "nominee", "championship")):
        value += 2.0
    if any(word in text for word in ("ceasefire", "resign", "ban", "default", "bankrupt")):
        value -= 5.0
    if any(word in text for word in ("crypto", "bitcoin", "ethereum")):
        value += 3.0
    if any(word in text for word in ("sports", "nba", "nfl", "mlb", "nhl", "soccer")):
        value += 5.0

    digest = hashlib.sha256(f"{participant}:{question}".encode("utf-8")).digest()
    noise = digest[0] % 17 - 8
    return min(88.0, max(8.0, value + noise))


def _tightens_market_enough(
    bid: float,
    offer: float,
    current_bid: float,
    current_offer: float,
    min_tighten_increment: float,
) -> bool:
    tolerance = 1e-9
    return (
        bid - current_bid + tolerance >= min_tighten_increment
        or current_offer - offer + tolerance >= min_tighten_increment
    )


def _tighten_market(
    bid: float,
    offer: float,
    current_bid: float,
    current_offer: float,
    min_tighten_increment: float,
    prefer_bid: bool,
) -> tuple[float, float]:
    bid_tightened = current_bid + min_tighten_increment
    offer_tightened = current_offer - min_tighten_increment

    if prefer_bid and bid_tightened < offer:
        return max(bid, bid_tightened), offer
    if not prefer_bid and offer_tightened > bid:
        return bid, min(offer, offer_tightened)
    if bid_tightened < current_offer:
        return bid_tightened, min(offer, current_offer)
    if offer_tightened > current_bid:
        return max(bid, current_bid), offer_tightened
    return current_offer, current_bid


def _forced_trade_decision(
    fair_value: float,
    current_bid: float,
    current_offer: float,
) -> BotDecision:
    midpoint = (current_bid + current_offer) / 2
    if fair_value >= midpoint:
        return BotDecision(
            scratchpad=(
                f"My no-research fair value is {fair_value:.0f}%. Passing is "
                "disabled and there is no room to tighten without crossing, so I buy."
            ),
            verbal_action=f"I lift the {format_probability(current_offer)} offer.",
        )
    return BotDecision(
        scratchpad=(
            f"My no-research fair value is {fair_value:.0f}%. Passing is "
            "disabled and there is no room to tighten without crossing, so I sell."
        ),
        verbal_action=f"I hit the {format_probability(current_bid)} bid.",
    )


def _parse_decision_payload(payload: object) -> BotDecision:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Gemini returned non-object JSON: {payload!r}")
    scratchpad = str(payload.get("scratchpad", "")).strip()
    verbal_action = str(payload.get("verbal_action", "")).strip()
    if not verbal_action:
        raise RuntimeError(f"Gemini JSON omitted verbal_action: {payload!r}")

    return BotDecision(scratchpad=scratchpad, verbal_action=verbal_action)

