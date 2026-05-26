from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Iterable

from .models import (
    DEFAULT_BOT_COUNT,
    ActionResult,
    ParsedCommand,
    Quote,
    Trade,
    USER_NAME,
    build_bot_names,
)


DICE_COUNT = 5
DIE_SIDES = 6
MIN_TRUE_VALUE = DICE_COUNT
MAX_TRUE_VALUE = DICE_COUNT * DIE_SIDES
UNKNOWN_DIE_EV = (DIE_SIDES + 1) / 2
DEFAULT_TURN_ORDER = (*build_bot_names(DEFAULT_BOT_COUNT), USER_NAME)


def format_price(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_trading_command(text: str) -> ParsedCommand:
    raw = text.strip()
    normalized = re.sub(r"[!,?;:]", " ", raw.lower())
    normalized = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return ParsedCommand(kind="unknown", raw_text=raw)

    if re.search(r"\b(pass|skip|no market|nothing|stand aside)\b", normalized):
        return ParsedCommand(kind="pass", raw_text=raw)

    hit_match = re.search(
        r"\b(hit|sell(?:ing)?(?: to)?(?: the)? bid)\b(?:\s+(?:the\s+)?)?(\d+(?:\.\d+)?)?",
        normalized,
    )
    if hit_match:
        price = float(hit_match.group(2)) if hit_match.group(2) else None
        return ParsedCommand(kind="hit", raw_text=raw, price=price)

    lift_match = re.search(
        r"\b(lift|buy(?:ing)?(?: the)? offer|take(?: the)? offer)\b(?:\s+(?:the\s+)?)?(\d+(?:\.\d+)?)?",
        normalized,
    )
    if lift_match:
        price = float(lift_match.group(2)) if lift_match.group(2) else None
        return ParsedCommand(kind="lift", raw_text=raw, price=price)

    quote_patterns = (
        r"\b(\d+(?:\.\d+)?)\s*(?:at|@|/|x|by)\s*(\d+(?:\.\d+)?)\b",
        r"\bbid\s+(\d+(?:\.\d+)?)\s+(?:offer|ask)\s+(\d+(?:\.\d+)?)\b",
        r"\b(\d+(?:\.\d+)?)\s+bid\s+(\d+(?:\.\d+)?)\s+(?:offer|ask)\b",
    )
    for pattern in quote_patterns:
        quote_match = re.search(pattern, normalized)
        if quote_match:
            bid = float(quote_match.group(1))
            offer = float(quote_match.group(2))
            return ParsedCommand(kind="quote", raw_text=raw, bid=bid, offer=offer)

    return ParsedCommand(kind="unknown", raw_text=raw)


@dataclass
class MarketMakingGame:
    max_turns: int = 9
    end_on_trade: bool = False
    allow_pass: bool = True
    min_tighten_increment: float = 0.5
    turn_order: tuple[str, ...] = DEFAULT_TURN_ORDER
    rng: random.Random = field(default_factory=random.Random)
    dice: list[int] = field(init=False)
    true_value: int = field(init=False)
    private_die_indexes: dict[str, int] = field(init=False)
    private_dice: dict[str, int] = field(init=False)
    quotes: dict[str, Quote] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    positions: dict[str, int] = field(init=False)
    cash: dict[str, float] = field(init=False)
    turn_count: int = 0
    finished: bool = False

    def __post_init__(self) -> None:
        if len(set(self.turn_order)) != len(self.turn_order):
            raise ValueError("Turn order must not contain duplicate participants.")
        if USER_NAME not in self.turn_order:
            raise ValueError(f"Turn order must include {USER_NAME}.")

        self.dice = [self.rng.randint(1, DIE_SIDES) for _ in range(DICE_COUNT)]
        self.true_value = sum(self.dice)
        self.private_die_indexes = self._assign_private_die_indexes()
        self.private_dice = {
            participant: self.dice[index]
            for participant, index in self.private_die_indexes.items()
        }
        self.positions = {participant: 0 for participant in self.turn_order}
        self.cash = {participant: 0.0 for participant in self.turn_order}
        self.history.append(
            f"The exchange rolled {DICE_COUNT} {DIE_SIDES}-sided dice "
            "and distributed private signals."
        )
        self.history.append(f"Turn order is {' -> '.join(self.turn_order)}.")
        if self.private_signal_sharing_required():
            self.history.append(
                "There are more participants than dice, so at least some private "
                "signals are shared. The exchange does not reveal who shares with whom."
            )
        else:
            self.history.append("Each participant has a unique private die signal.")

    def _assign_private_die_indexes(self) -> dict[str, int]:
        shuffled_die_indexes = list(range(DICE_COUNT))
        self.rng.shuffle(shuffled_die_indexes)

        assignments: dict[str, int] = {}
        for participant, die_index in zip(self.turn_order, shuffled_die_indexes):
            assignments[participant] = die_index

        for participant in self.turn_order[len(shuffled_die_indexes) :]:
            assignments[participant] = self.rng.choice(shuffled_die_indexes)

        return assignments

    def current_participant(self) -> str:
        return self.turn_order[self.turn_count % len(self.turn_order)]

    def private_die_for(self, participant: str) -> int | None:
        return self.private_dice.get(participant)

    def private_die_number_for(self, participant: str) -> int | None:
        index = self.private_die_indexes.get(participant)
        if index is None:
            return None
        return index + 1

    def turn_position_for(self, participant: str) -> int | None:
        if participant not in self.turn_order:
            return None
        return self.turn_order.index(participant) + 1

    def private_signal_sharing_required(self) -> bool:
        return len(self.turn_order) > DICE_COUNT

    def private_signal_shared_count(self) -> int:
        return max(len(self.turn_order) - DICE_COUNT, 0)

    def public_state_for(self, participant: str) -> dict[str, object]:
        action_tape = [
            {"sequence": index, "event": event}
            for index, event in enumerate(self.history)
        ]
        participants = list(self.turn_order)
        turn_positions = {
            name: index + 1
            for index, name in enumerate(participants)
        }
        assigned_die_indexes = set(self.private_die_indexes.values())
        sharing_required = self.private_signal_sharing_required()
        return {
            "participant": participant,
            "turn_index": self.turn_count,
            "max_turns": self.max_turns,
            "turn_order": list(self.turn_order),
            "game": {
                "current_turn_index": self.turn_count,
                "current_turn_number": self.turn_count + 1,
                "max_turns": self.max_turns,
                "turns_remaining_including_current": max(
                    self.max_turns - self.turn_count,
                    0,
                ),
                "end_on_trade": self.end_on_trade,
                "allow_pass": self.allow_pass,
                "min_tighten_increment": self.min_tighten_increment,
                "current_participant": self.current_participant(),
                "participants": participants,
                "participant_count": len(participants),
                "other_participants": [
                    name for name in participants if name != participant
                ],
                "turn_order": participants,
                "turn_positions": turn_positions,
                "your_turn_position": self.turn_position_for(participant),
            },
            "order_book": [quote.as_public_dict() for quote in self.quotes.values()],
            "best_bid": self._best_bid(exclude=participant).as_public_dict()
            if self._best_bid(exclude=participant)
            else None,
            "best_offer": self._best_offer(exclude=participant).as_public_dict()
            if self._best_offer(exclude=participant)
            else None,
            "action_tape": action_tape,
            "recent_history": [item["event"] for item in action_tape],
            "trades": [trade.as_public_dict() for trade in self.trades],
            "trade_count": len(self.trades),
            "own_position": self.positions[participant],
            "own_cash": round(self.cash[participant], 2),
            "asset": {
                "description": f"sum of {DICE_COUNT} {DIE_SIDES}-sided dice",
                "dice_count": DICE_COUNT,
                "die_sides": DIE_SIDES,
                "range": [MIN_TRUE_VALUE, MAX_TRUE_VALUE],
                "unconditional_expected_value": DICE_COUNT * UNKNOWN_DIE_EV,
            },
            "information_structure": {
                "your_private_die_number": self.private_die_number_for(participant),
                "your_private_die_value": self.private_die_for(participant),
                "hidden_die_count": DICE_COUNT - len(assigned_die_indexes),
                "private_signal_sharing_required": sharing_required,
                "minimum_number_of_shared_private_signals": self.private_signal_shared_count(),
                "rule": (
                    "Each participant knows only their own private die value. If "
                    "there are more participants than dice, some participants share "
                    "the same private die information, but the exchange does not "
                    "reveal who shares with whom. Do not infer or claim hidden die "
                    "values or shared-signal identities before showdown."
                ),
            },
            "private_signal": {
                "die_number": self.private_die_number_for(participant),
                "die_value": self.private_die_for(participant),
            },
            "trading_rules": self.trading_rules(),
        }

    def trading_rules(self) -> dict[str, object]:
        allowed_actions = ["make_or_update_market", "hit_bid", "lift_offer"]
        if self.allow_pass:
            allowed_actions.append("pass")

        return {
            "allowed_actions": allowed_actions,
            "allow_pass": self.allow_pass,
            "min_tighten_increment": self.min_tighten_increment,
            "quote_rule": (
                "A new quote must be inside or equal to the current best external "
                "market. Crossing the best bid or offer executes a trade."
            ),
            "no_pass_quote_rule": (
                "When passing is disabled and an external best market exists, "
                "a non-crossing quote must tighten the best bid by at least "
                "min_tighten_increment or tighten the best offer by at least "
                "min_tighten_increment. If no external market exists, make an "
                "initial valid two-sided market."
            ),
        }

    def apply_action(self, participant: str, command_text: str | ParsedCommand) -> ActionResult:
        command = (
            parse_trading_command(command_text)
            if isinstance(command_text, str)
            else command_text
        )
        if command.kind == "unknown":
            return self._reject_action(
                participant,
                f"Could not parse trading command: {command.raw_text!r}",
            )

        if command.kind == "pass":
            if not self.allow_pass:
                return self._reject_action(
                    participant,
                    "Passing is not allowed in this game. Hit, lift, or tighten the market.",
                )
            self._record_and_advance(f"{participant} passed.")
            return ActionResult(True, f"{participant} passed.", finished=self.finished)

        if command.kind == "quote":
            return self._handle_quote(participant, command)

        if command.kind == "hit":
            return self._hit_bid(participant, command.price)

        if command.kind == "lift":
            return self._lift_offer(participant, command.price)

        return ActionResult(False, "Unsupported action.", finished=self.finished)

    def order_book_line(self) -> str:
        if not self.quotes:
            return "Book is empty."

        quotes = sorted(self.quotes.values(), key=lambda quote: quote.owner)
        parts = [
            f"{quote.owner}: {format_price(quote.bid)} at {format_price(quote.offer)}"
            for quote in quotes
        ]
        return " | ".join(parts)

    def settlement(self) -> dict[str, dict[str, float | int]]:
        return {
            participant: {
                "position": self.positions[participant],
                "cash": round(self.cash[participant], 2),
                "pnl": round(
                    self.cash[participant] + self.positions[participant] * self.true_value,
                    2,
                ),
            }
            for participant in self.turn_order
        }

    def reveal(self) -> dict[str, object]:
        return {
            "dice": {
                f"die_{index}": value
                for index, value in enumerate(self.dice, start=1)
            },
            "true_value": self.true_value,
            "settlement": self.settlement(),
        }

    def _handle_quote(self, participant: str, command: ParsedCommand) -> ActionResult:
        assert command.bid is not None
        assert command.offer is not None

        bid = command.bid
        offer = command.offer
        if bid < MIN_TRUE_VALUE or offer > MAX_TRUE_VALUE:
            return self._reject_action(
                participant,
                message=(
                    f"Market must stay inside the dice-sum range "
                    f"{MIN_TRUE_VALUE}-{MAX_TRUE_VALUE}."
                ),
            )
        if bid >= offer:
            return self._reject_action(
                participant,
                "A market must have bid strictly below offer.",
            )

        best_offer = self._best_offer(exclude=participant)
        if best_offer and bid >= best_offer.offer:
            return self._execute_trade(
                buyer=participant,
                seller=best_offer.owner,
                price=best_offer.offer,
                source=f"{participant} crossed the offer while quoting.",
            )

        best_bid = self._best_bid(exclude=participant)
        if best_bid and offer <= best_bid.bid:
            return self._execute_trade(
                buyer=best_bid.owner,
                seller=participant,
                price=best_bid.bid,
                source=f"{participant} crossed the bid while quoting.",
            )

        if best_bid and best_offer and (bid < best_bid.bid or offer > best_offer.offer):
            current_market = (
                f"{format_price(best_bid.bid)} at {format_price(best_offer.offer)}"
            )
            attempted_market = f"{format_price(bid)} at {format_price(offer)}"
            return self._reject_action(
                participant,
                message=(
                    f"{attempted_market} is outside the current best market "
                    f"{current_market}. Quote inside that market, hit/lift it"
                    f"{', or pass' if self.allow_pass else ''}."
                ),
            )

        if (
            not self.allow_pass
            and best_bid
            and best_offer
            and not self._tightens_market_enough(bid, offer, best_bid, best_offer)
        ):
            current_market = (
                f"{format_price(best_bid.bid)} at {format_price(best_offer.offer)}"
            )
            attempted_market = f"{format_price(bid)} at {format_price(offer)}"
            return self._reject_action(
                participant,
                message=(
                    f"{attempted_market} does not tighten the current best market "
                    f"{current_market} by at least "
                    f"{format_price(self.min_tighten_increment)} on either side. "
                    "Hit, lift, or tighten at least one side."
                ),
            )

        self.quotes[participant] = Quote(owner=participant, bid=bid, offer=offer)
        message = f"{participant} made {format_price(bid)} at {format_price(offer)}."
        self._record_and_advance(message)
        return ActionResult(True, message, finished=self.finished)

    def _hit_bid(self, participant: str, requested_price: float | None) -> ActionResult:
        quote = self._best_bid(exclude=participant, requested_price=requested_price)
        if not quote:
            price_text = (
                f" at {format_price(requested_price)}"
                if requested_price is not None
                else ""
            )
            return self._reject_action(
                participant,
                f"No external bid available{price_text}.",
            )

        return self._execute_trade(
            buyer=quote.owner,
            seller=participant,
            price=quote.bid,
            source=f"{participant} hit {quote.owner}'s bid.",
        )

    def _lift_offer(self, participant: str, requested_price: float | None) -> ActionResult:
        quote = self._best_offer(exclude=participant, requested_price=requested_price)
        if not quote:
            price_text = (
                f" at {format_price(requested_price)}"
                if requested_price is not None
                else ""
            )
            return self._reject_action(
                participant,
                f"No external offer available{price_text}.",
            )

        return self._execute_trade(
            buyer=participant,
            seller=quote.owner,
            price=quote.offer,
            source=f"{participant} lifted {quote.owner}'s offer.",
        )

    def _execute_trade(self, buyer: str, seller: str, price: float, source: str) -> ActionResult:
        quantity = 1
        self.positions[buyer] += quantity
        self.positions[seller] -= quantity
        self.cash[buyer] -= price * quantity
        self.cash[seller] += price * quantity

        trade = Trade(
            buyer=buyer,
            seller=seller,
            price=price,
            quantity=quantity,
            turn_index=self.turn_count,
            source=source,
        )
        self.trades.append(trade)
        self.quotes.pop(seller, None)
        self.quotes.pop(buyer, None)

        message = (
            f"TRADE: {buyer} bought {quantity} from {seller} "
            f"at {format_price(price)}. {source}"
        )
        self._record_and_advance(message, trade=trade)
        return ActionResult(True, message, trade=trade, finished=self.finished)

    def _record_and_advance(self, event: str, trade: Trade | None = None) -> None:
        self.history.append(event)
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            self.finished = True
        if trade and self.end_on_trade:
            self.finished = True

    def _reject_action(self, participant: str, message: str) -> ActionResult:
        self.history.append(f"REJECTED: {participant}: {message}")
        return ActionResult(False, message, finished=self.finished)

    def _tightens_market_enough(
        self,
        bid: float,
        offer: float,
        best_bid: Quote,
        best_offer: Quote,
    ) -> bool:
        tolerance = 1e-9
        bid_tightening = bid - best_bid.bid
        offer_tightening = best_offer.offer - offer
        return (
            bid_tightening + tolerance >= self.min_tighten_increment
            or offer_tightening + tolerance >= self.min_tighten_increment
        )

    def _best_bid(
        self,
        exclude: str | None = None,
        requested_price: float | None = None,
    ) -> Quote | None:
        candidates = self._quote_candidates(exclude=exclude)
        if requested_price is not None:
            candidates = [
                quote for quote in candidates if math.isclose(quote.bid, requested_price)
            ]
        return max(candidates, key=lambda quote: quote.bid, default=None)

    def _best_offer(
        self,
        exclude: str | None = None,
        requested_price: float | None = None,
    ) -> Quote | None:
        candidates = self._quote_candidates(exclude=exclude)
        if requested_price is not None:
            candidates = [
                quote for quote in candidates if math.isclose(quote.offer, requested_price)
            ]
        return min(candidates, key=lambda quote: quote.offer, default=None)

    def _quote_candidates(self, exclude: str | None = None) -> list[Quote]:
        return [quote for quote in self.quotes.values() if quote.owner != exclude]


def active_bot_names(turn_order: Iterable[str]) -> list[str]:
    return [participant for participant in turn_order if participant != USER_NAME]
