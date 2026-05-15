from __future__ import annotations

import json
import sys
from pathlib import Path

from .audio import build_listener, build_speaker
from .bots import BotClient, BotDecision, GeminiBotClient, HeuristicBotClient
from .config import Settings
from .engine import (
    DEFAULT_TURN_ORDER,
    DIE_SIDES,
    DICE_COUNT,
    MAX_TRUE_VALUE,
    MIN_TRUE_VALUE,
    UNKNOWN_DIE_EV,
    MarketMakingGame,
    active_bot_names,
)
from .models import BOT_PROFILES, USER_NAME


class ScratchpadLogger:
    def __init__(self, path: str) -> None:
        self._path = Path(path) if path else None

    def write(self, participant: str, decision: BotDecision, turn_index: int) -> None:
        if not self._path:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "turn_index": turn_index,
            "participant": participant,
            "scratchpad": decision.scratchpad,
            "verbal_action": decision.verbal_action,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def main() -> int:
    settings = Settings.from_env()
    game = MarketMakingGame(
        max_turns=settings.max_turns,
        end_on_trade=settings.end_on_trade,
        allow_pass=settings.allow_pass,
        min_tighten_increment=settings.min_tighten_increment,
        turn_order=DEFAULT_TURN_ORDER,
    )
    speaker = build_speaker(settings)
    listener = build_listener(settings)
    bot_client = _build_bot_client(settings)
    scratchpad_logger = ScratchpadLogger(settings.scratchpad_log_path)

    _print_intro(game, settings, speaker, listener, bot_client)

    while not game.finished:
        participant = game.current_participant()
        if participant == USER_NAME:
            command_text = _read_user_action(game, listener)
        else:
            profile = BOT_PROFILES[participant]
            private_die = game.private_die_for(participant)
            assert private_die is not None
            try:
                decision = bot_client.decide(
                    participant=participant,
                    profile=profile,
                    public_state=game.public_state_for(participant),
                    private_die=private_die,
                )
            except Exception as exc:
                print(f"[bot fallback] {participant} decision failed: {exc}")
                decision = HeuristicBotClient().decide(
                    participant=participant,
                    profile=profile,
                    public_state=game.public_state_for(participant),
                    private_die=private_die,
                )
            scratchpad_logger.write(participant, decision, game.turn_count)
            speaker.speak(participant, decision.verbal_action, profile)
            command_text = decision.verbal_action

        result = game.apply_action(participant, command_text)
        if not result.accepted:
            print(result.message)
            if participant == USER_NAME:
                continue

            fallback_text = _fallback_bot_command(game, participant, profile)
            speaker.speak(participant, fallback_text, profile)
            result = game.apply_action(participant, fallback_text)
            if not result.accepted:
                print(result.message)
                fallback_text = _last_resort_bot_command(game, participant)
                speaker.speak(participant, fallback_text, profile)
                result = game.apply_action(participant, fallback_text)

        print(result.message)
        print(f"Book: {game.order_book_line()}")
        print()

    _print_showdown(game)
    return 0


def _build_bot_client(settings: Settings) -> BotClient:
    if settings.use_gemini and settings.gemini_api_key:
        try:
            return GeminiBotClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                temperature=settings.bot_temperature,
            )
        except RuntimeError as exc:
            print(f"[gemini disabled] {exc}")
    elif settings.use_gemini:
        print("[gemini disabled] GEMINI_API_KEY is not set; using heuristic bots.")

    return HeuristicBotClient()


def _print_intro(
    game: MarketMakingGame,
    settings: Settings,
    speaker,
    listener,
    bot_client: BotClient,
) -> None:
    active_bots = ", ".join(active_bot_names(game.turn_order))
    print("Quant Market-Making 5-Dice Simulator")
    print(f"Active bots: {active_bots}")
    print(f"Turns: {settings.max_turns}; end on first trade: {settings.end_on_trade}")
    print(
        f"Pass allowed: {settings.allow_pass}; "
        f"minimum tightening: {settings.min_tighten_increment:g}"
    )
    print(
        f"Underlying: sum of {DICE_COUNT} {DIE_SIDES}-sided dice, "
        f"range {MIN_TRUE_VALUE}-{MAX_TRUE_VALUE}, EV {DICE_COUNT * UNKNOWN_DIE_EV:g}"
    )
    print(
        f"Your private signal: Die #{game.private_die_number_for(USER_NAME)} = "
        f"{game.private_die_for(USER_NAME)}"
    )
    print("Commands: '16 at 18', 'pass', 'hit the 16', 'lift the 18'")
    print(f"Python: {sys.executable}")
    print(bot_client.status)
    print(speaker.status)
    print(listener.status)
    print()
    speaker.speak("Exchange", _spoken_intro(game))


def _read_user_action(game: MarketMakingGame, listener) -> str:
    while True:
        print("[YOUR TURN]")
        print(f"Book: {game.order_book_line()}")
        print(f"Your Die #3: {game.private_die_for(USER_NAME)}")
        command_text = listener.listen()
        if command_text:
            return command_text
        print("Please enter a trading command.")


def _spoken_intro(game: MarketMakingGame) -> str:
    turn_word = "turn" if game.max_turns == 1 else "turns"
    end_condition = (
        "It will end immediately on the first trade."
        if game.end_on_trade
        else "It will not end on the first trade."
    )
    pass_rule = (
        "Passing is allowed."
        if game.allow_pass
        else (
            "Passing is not allowed. You must hit, lift, or tighten the current "
            f"market by at least {format_spoken_price(game.min_tighten_increment)} "
            "on at least one side."
        )
    )
    private_die_number = game.private_die_number_for(USER_NAME)
    private_die_value = game.private_die_for(USER_NAME)
    private_die_label = (
        f"die number {_speakable_number(private_die_number)}"
        if private_die_number is not None
        else "your private die"
    )

    return (
        "Game starting. "
        f"This game has {_speakable_number(game.max_turns)} {turn_word}. "
        f"{end_condition} "
        f"{pass_rule} "
        "The underlying is the sum of "
        f"{_dice_phrase(DICE_COUNT, DIE_SIDES)}. "
        f"Your private information is {private_die_label}, "
        f"which is {_speakable_number(private_die_value)}."
    )


def _fallback_bot_command(game: MarketMakingGame, participant: str, profile) -> str:
    private_die = game.private_die_for(participant)
    if private_die is None:
        return _last_resort_bot_command(game, participant)

    decision = HeuristicBotClient().decide(
        participant=participant,
        profile=profile,
        public_state=game.public_state_for(participant),
        private_die=private_die,
    )
    return decision.verbal_action


def _last_resort_bot_command(game: MarketMakingGame, participant: str) -> str:
    state = game.public_state_for(participant)
    best_offer = state.get("best_offer")
    if isinstance(best_offer, dict):
        return f"I lift the {best_offer['offer']} offer."

    best_bid = state.get("best_bid")
    if isinstance(best_bid, dict):
        return f"I hit the {best_bid['bid']} bid."

    return f"I make {MIN_TRUE_VALUE} at {MAX_TRUE_VALUE}."


def format_spoken_price(value: float) -> str:
    numeric_value = float(value)
    return (
        _speakable_number(int(numeric_value))
        if numeric_value.is_integer()
        else f"{numeric_value:g}"
    )


def _dice_phrase(count: int, sides: int) -> str:
    die_word = "die" if count == 1 else "dice"
    return f"{_speakable_number(count)} {_sided_die_phrase(sides)} {die_word}"


def _sided_die_phrase(sides: int) -> str:
    if sides == 1:
        return "one-sided"
    return f"{_speakable_number(sides)}-sided"


def _speakable_number(value: int | None) -> str:
    if value is None:
        return "unknown"
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(value, str(value))


def _print_showdown(game: MarketMakingGame) -> None:
    reveal = game.reveal()
    dice = reveal["dice"]
    print("SHOWDOWN")
    print(
        "Dice: "
        f"{dice['die_1']}, {dice['die_2']}, {dice['die_3']}, "
        f"{dice['die_4']}, {dice['die_5']}"
    )
    print(f"True value: {reveal['true_value']}")
    print("PnL:")
    for participant, row in reveal["settlement"].items():
        print(
            f"  {participant}: position {row['position']}, "
            f"cash {row['cash']}, pnl {row['pnl']}"
        )
