from __future__ import annotations

import random
import sys
from datetime import datetime, timezone

from shared.audio import build_audio_cues, build_listener, build_speaker
from shared.logging import JsonlLogger

from .bots import BotClient, BotDecision, GeminiBotClient, HeuristicBotClient
from .config import Settings
from .engine import (
    DIE_SIDES,
    DICE_COUNT,
    MAX_TRUE_VALUE,
    MIN_TRUE_VALUE,
    UNKNOWN_DIE_EV,
    MarketMakingGame,
    active_bot_names,
)
from .models import USER_NAME, build_bot_names, build_bot_profiles


class ScratchpadLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(self, participant: str, decision: BotDecision, turn_index: int) -> None:
        self._logger.write(
            {
                "turn_index": turn_index,
                "participant": participant,
                "scratchpad": decision.scratchpad,
                "verbal_action": decision.verbal_action,
            }
        )


class GameSummaryLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(self, game: MarketMakingGame) -> None:
        settlement = game.settlement()
        self._logger.write(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "turns_used": game.turn_count,
                "bot_count": len(active_bot_names(game.turn_order)),
                "user_turn_position": game.turn_position_for(USER_NAME),
                "user_final_pnl": settlement[USER_NAME]["pnl"],
            }
        )


def main() -> int:
    settings = Settings.from_env()
    try:
        bot_names = build_bot_names(settings.bot_count)
        bot_profiles = build_bot_profiles(bot_names)
        turn_order = _build_turn_order(settings, bot_names)
        game = MarketMakingGame(
            max_turns=settings.max_turns,
            end_on_trade=settings.end_on_trade,
            allow_pass=settings.allow_pass,
            min_tighten_increment=settings.min_tighten_increment,
            turn_order=turn_order,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2
    speaker = build_speaker(settings)
    cue_player = build_audio_cues(settings)
    listener = build_listener(settings, cue_player)
    bot_client = _build_bot_client(settings)
    scratchpad_logger = ScratchpadLogger(settings.scratchpad_log_path)
    game_summary_logger = GameSummaryLogger(settings.game_summary_log_path)

    _print_intro(game, settings, speaker, listener, cue_player, bot_client)

    while not game.finished:
        participant = game.current_participant()
        if participant == USER_NAME:
            command_text = _read_user_action(game, listener, cue_player)
        else:
            profile = bot_profiles[participant]
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
        if participant == USER_NAME and result.accepted:
            cue_player.command_accepted()
        if not result.accepted:
            if participant == USER_NAME:
                cue_player.command_rejected()
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

    _print_showdown(game, speaker)
    game_summary_logger.write(game)
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


def _build_turn_order(settings: Settings, bot_names: tuple[str, ...]) -> tuple[str, ...]:
    _validate_bot_count(settings.bot_count)
    participants = [*bot_names, USER_NAME]
    if settings.randomize_turn_order:
        random.shuffle(participants)
    return tuple(participants)


def _validate_bot_count(bot_count: int) -> None:
    if bot_count < 1:
        raise ValueError("BOT_COUNT must be at least 1.")


def _print_intro(
    game: MarketMakingGame,
    settings: Settings,
    speaker,
    listener,
    cue_player,
    bot_client: BotClient,
) -> None:
    active_bots = ", ".join(active_bot_names(game.turn_order))
    print("Quant Market-Making 5-Dice Simulator")
    print(f"Active bots: {active_bots}")
    print(f"Bot count: {settings.bot_count}; participants: {len(game.turn_order)}")
    print(
        f"Private signal sharing required: {game.private_signal_sharing_required()}"
    )
    print(f"Turns: {settings.max_turns}; end on first trade: {settings.end_on_trade}")
    print(f"Turn order randomized: {settings.randomize_turn_order}")
    print(f"Turn order: {' -> '.join(game.turn_order)}")
    print(
        f"Your turn position: {game.turn_position_for(USER_NAME)} "
        f"of {len(game.turn_order)}"
    )
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
    print(cue_player.status)
    print(f"Scratchpad log: {settings.scratchpad_log_path}")
    print(f"Game summary log: {settings.game_summary_log_path}")
    print()
    speaker.speak("Exchange", _spoken_intro(game))


def _read_user_action(game: MarketMakingGame, listener, cue_player) -> str:
    while True:
        print("[YOUR TURN]")
        print(f"Book: {game.order_book_line()}")
        print(
            f"Your Die #{game.private_die_number_for(USER_NAME)}: "
            f"{game.private_die_for(USER_NAME)}"
        )
        cue_player.user_turn_started()
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
    private_die_value = game.private_die_for(USER_NAME)
    user_turn_position = game.turn_position_for(USER_NAME)
    bot_count = len(active_bot_names(game.turn_order))
    opponent_word = "opponent" if bot_count == 1 else "opponents"
    participant_word = "participant" if len(game.turn_order) == 1 else "participants"
    bot_count_phrase = (
        f"There is {_speakable_number(bot_count)} bot {opponent_word}"
        if bot_count == 1
        else f"There are {_speakable_number(bot_count)} bot {opponent_word}"
    )
    sharing_rule = (
        "There are more participants than dice, so at least some participants "
        "share the same private die information. You will not be told who shares "
        "with whom."
        if game.private_signal_sharing_required()
        else "Each participant has a unique private die signal."
    )

    return (
        "Game starting. "
        f"This game has {_speakable_number(game.max_turns)} {turn_word}. "
        f"{end_condition} "
        f"{pass_rule} "
        f"{bot_count_phrase} and "
        f"{_speakable_number(len(game.turn_order))} total {participant_word}. "
        f"{sharing_rule} "
        f"You are {_ordinal_phrase(user_turn_position)} in the turn order, "
        f"out of {_speakable_number(len(game.turn_order))} {participant_word}. "
        f"The turn order is {_spoken_turn_order(game.turn_order)}. "
        "The underlying is the sum of "
        f"{_dice_phrase(DICE_COUNT, DIE_SIDES)}. "
        f"Your private die value is {_speakable_number(private_die_value)}."
    )


def _spoken_turn_order(turn_order: tuple[str, ...]) -> str:
    names = [_participant_spoken_name(name) for name in turn_order]
    if not names:
        return "unknown"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])}, then {names[-1]}"


def _participant_spoken_name(name: str) -> str:
    if name == USER_NAME:
        return "you"
    return name.replace("_", " ")


def _ordinal_phrase(value: int | None) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
        11: "eleventh",
        12: "twelfth",
    }
    if value is None:
        return "unknown"
    return words.get(value, str(value))


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


def _print_showdown(game: MarketMakingGame, speaker) -> None:
    reveal = game.reveal()
    dice = reveal["dice"]
    print("SHOWDOWN")
    print(f"Dice: {', '.join(str(value) for value in dice.values())}")
    print(f"True value: {reveal['true_value']}")
    print("PnL:")
    for participant, row in reveal["settlement"].items():
        print(
            f"  {participant}: position {row['position']}, "
            f"cash {row['cash']}, pnl {row['pnl']}"
        )

    user_row = reveal["settlement"][USER_NAME]
    speaker.speak(
        "Exchange",
        (
            "Game over. "
            f"The true value is {reveal['true_value']}. "
            f"Your final P and L is {format_spoken_price(float(user_row['pnl']))}."
        ),
    )
