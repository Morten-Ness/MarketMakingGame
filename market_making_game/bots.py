from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .engine import UNKNOWN_DIE_EV, format_price
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
        private_die: int,
    ) -> BotDecision:
        ...


class HeuristicBotClient:
    """Offline fallback that behaves like a simple signal-aware market maker."""

    status = "Bot brain: local heuristic fallback (programmatic)."

    def decide(
        self,
        participant: str,
        profile: BotProfile,
        public_state: dict[str, object],
        private_die: int,
    ) -> BotDecision:
        asset = public_state.get("asset", {})
        dice_count = int(asset.get("dice_count", 5)) if isinstance(asset, dict) else 5
        die_sides = int(asset.get("die_sides", 6)) if isinstance(asset, dict) else 6
        unknown_dice_count = max(dice_count - 1, 0)
        unknown_die_ev = (die_sides + 1) / 2
        fair_value = private_die + unknown_dice_count * unknown_die_ev
        min_value, max_value = _asset_range(asset, dice_count, die_sides)
        trading_rules = public_state.get("trading_rules", {})
        allow_pass = (
            bool(trading_rules.get("allow_pass", True))
            if isinstance(trading_rules, dict)
            else True
        )
        min_tighten_increment = (
            float(trading_rules.get("min_tighten_increment", 0.5))
            if isinstance(trading_rules, dict)
            else 0.5
        )
        remainder_text = (
            f"{unknown_dice_count} * {unknown_die_ev:g} = "
            f"{unknown_dice_count * unknown_die_ev:g}"
        )
        best_bid = public_state.get("best_bid")
        best_offer = public_state.get("best_offer")

        edge = 1.5
        if "Aggressive" in profile.style:
            edge = 1.25
        elif "Conservative" in profile.style:
            edge = 2.0
        elif "Tight" in profile.style:
            edge = 1.75

        if isinstance(best_offer, dict):
            offer = float(best_offer["offer"])
            if offer <= fair_value - edge:
                return BotDecision(
                    scratchpad=(
                        f"My die is {private_die}. Remainder EV is {remainder_text}. "
                        f"Baseline EV is {fair_value:.2f}. Best offer is {offer:.2f}, "
                        f"which clears my risk-adjusted buy edge of {edge:.2f}."
                    ),
                    verbal_action=f"I lift the {format_price(offer)} offer.",
                )

        if isinstance(best_bid, dict):
            bid = float(best_bid["bid"])
            if bid >= fair_value + edge:
                return BotDecision(
                    scratchpad=(
                        f"My die is {private_die}. Remainder EV is {remainder_text}. "
                        f"Baseline EV is {fair_value:.2f}. Best bid is {bid:.2f}, "
                        f"which clears my risk-adjusted sell edge of {edge:.2f}."
                    ),
                    verbal_action=f"I hit the {format_price(bid)} bid.",
                )

        half_width = 1.75
        if "Aggressive" in profile.style:
            half_width = 1.5
        if "Conservative" in profile.style:
            half_width = 2.25
        elif "Tight" in profile.style:
            half_width = 1.25

        bid = max(min_value, int(fair_value - half_width))
        offer = min(max_value, int(fair_value + half_width + 0.9999))
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
                    return _forced_trade_decision(
                        private_die=private_die,
                        fair_value=fair_value,
                        current_bid=current_bid,
                        current_offer=current_offer,
                    )
                return BotDecision(
                    scratchpad=(
                        f"My die is {private_die}. Baseline EV is {fair_value:.2f}. "
                        "The current market is too tight for me to improve both sides "
                        "without crossing, and I do not have enough edge to trade."
                    ),
                    verbal_action="I pass.",
                )
        if bid >= offer:
            offer = bid + 1

        return BotDecision(
            scratchpad=(
                f"My die is {private_die}. Remainder EV is {remainder_text}. "
                f"Baseline EV is {fair_value:.2f}. No available quote gives enough "
                f"edge, so I will make a {profile.style.lower()} market."
            ),
            verbal_action=f"I make {format_price(bid)} at {format_price(offer)}.",
        )


class GeminiBotClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.35) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install requirements.txt or disable Gemini."
            ) from exc

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def status(self) -> str:
        return f"Bot brain: Gemini API ({self._model}, temperature={self._temperature})."

    def decide(
        self,
        participant: str,
        profile: BotProfile,
        public_state: dict[str, object],
        private_die: int,
    ) -> BotDecision:
        system_instruction = _system_instruction(participant, profile, public_state)
        user_prompt = {
            "private_die": private_die,
            "private_signal": public_state.get("private_signal"),
            "public_state": public_state,
            "task": (
                "Choose exactly one legal trading action: make/update a market, "
                "pass, hit a bid, or lift an offer."
            ),
        }

        response = self._client.models.generate_content(
            model=self._model,
            contents=json.dumps(user_prompt, indent=2),
            config=self._types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=self._temperature,
            ),
        )
        return _parse_decision(response.text or "")


def _system_instruction(
    participant: str,
    profile: BotProfile,
    public_state: dict[str, object],
) -> str:
    game = public_state.get("game", {})
    asset = public_state.get("asset", {})
    information_structure = public_state.get("information_structure", {})
    trading_rules = public_state.get("trading_rules", {})
    allow_pass = (
        bool(trading_rules.get("allow_pass", True))
        if isinstance(trading_rules, dict)
        else True
    )
    action_examples = [
        '- "I make 16 at 18."',
        '- "I hit the 16 bid."',
        '- "I lift the 18 offer."',
    ]
    if allow_pass:
        action_examples.insert(1, '- "I pass."')
    else:
        action_examples.append("- Passing is not legal in this game.")
    return f"""
You are {participant}, a competitive interview-practice market-making bot.
Your trading style is: {profile.style}.

Authoritative game configuration:
{json.dumps(game, indent=2)}

Authoritative underlying asset:
{json.dumps(asset, indent=2)}

Authoritative information structure:
{json.dumps(information_structure, indent=2)}

Authoritative trading rules:
{json.dumps(trading_rules, indent=2)}

You know your own private die value from private_signal. You do not know anyone
else's die value or any hidden die values.
If private_signal_sharing_required is true, at least some participants share
the same private die information, but you are not told who shares with whom or
whether your own signal is shared.
Use the public order book, full chronological action_tape, and trade history,
but do not claim to know hidden dice.

Risk policy:
- Be risk-aware and patient. The goal is realistic interview practice, not instant execution.
- Your baseline fair value is your private die value plus the expected value of all dice whose values you do not know.
- Read action_tape in sequence order before deciding. It contains all public quotes, passes, attempted trades, executed trades, and rejected actions so far.
- Only hit or lift when the public quote is clearly mispriced by at least 1.5 points versus your fair value.
- If your style is Conservative Market Maker, require about 2.0 points of edge before crossing.
- If your style is Aggressive Arbitrageur, you may cross at about 1.25 points of edge, but do not cross for tiny edges.
- Prefer making/updating a two-sided market or passing when edge is weak.
- If allow_pass is false, do not pass. You must hit, lift, or quote a market that tightens the current best market by at least min_tighten_increment on at least one side.
- If there is an existing best market, any new quote must be inside or equal to it:
  your bid must be at least the current best bid, and your offer must be no higher than the current best offer.
- Do not make one-sided extensions such as improving the bid while worsening the offer, or improving the offer while worsening the bid.
- Avoid immediately trading against the first quote unless your private signal gives a strong reason.
- A reasonable two-sided market is usually 2 to 4 points wide, adjusted for your style and position.

Return only strict JSON matching this exact schema:
{{
  "scratchpad": "brief private computation and rationale, never spoken aloud",
  "verbal_action": "one spoken trading command"
}}

The verbal_action must be one of:
{chr(10).join(action_examples)}
""".strip()


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
    private_die: int,
    fair_value: float,
    current_bid: float,
    current_offer: float,
) -> BotDecision:
    midpoint = (current_bid + current_offer) / 2
    if fair_value >= midpoint:
        return BotDecision(
            scratchpad=(
                f"My die is {private_die}. Baseline EV is {fair_value:.2f}. "
                "Passing is disabled and there is no room to tighten without crossing, "
                "so I will buy the offer."
            ),
            verbal_action=f"I lift the {format_price(current_offer)} offer.",
        )
    return BotDecision(
        scratchpad=(
            f"My die is {private_die}. Baseline EV is {fair_value:.2f}. "
            "Passing is disabled and there is no room to tighten without crossing, "
            "so I will sell to the bid."
        ),
        verbal_action=f"I hit the {format_price(current_bid)} bid.",
    )


def _asset_range(asset: object, dice_count: int, die_sides: int) -> tuple[int, int]:
    if isinstance(asset, dict):
        value_range = asset.get("range")
        if (
            isinstance(value_range, list)
            and len(value_range) == 2
            and all(isinstance(value, (int, float)) for value in value_range)
        ):
            return int(value_range[0]), int(value_range[1])
    return dice_count, dice_count * die_sides


def _parse_decision(text: str) -> BotDecision:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned invalid JSON: {text!r}") from exc

    scratchpad = str(payload.get("scratchpad", "")).strip()
    verbal_action = str(payload.get("verbal_action", "")).strip()
    if not verbal_action:
        raise RuntimeError(f"Gemini JSON omitted verbal_action: {payload!r}")

    return BotDecision(scratchpad=scratchpad, verbal_action=verbal_action)
