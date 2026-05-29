from __future__ import annotations

import random
import sys
from datetime import datetime, timezone

from shared.audio import build_audio_cues, build_listener, build_speaker
from shared.logging import JsonlLogger

from .bots import BotClient, BotDecision, GeminiBotClient, HeuristicBotClient
from .config import Settings
from .data import MarketRepository, MarketSelectionError
from .engine import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    PredictionMarketGame,
    active_bot_names,
    format_probability,
)
from .models import BotProfile, MarketSnapshot, USER_NAME, build_bot_names, build_bot_profiles


class ScratchpadLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(
        self,
        market: MarketSnapshot,
        participant: str,
        decision: BotDecision,
        turn_index: int,
    ) -> None:
        self._logger.write(
            {
                "turn_index": turn_index,
                "participant": participant,
                "market_id": market.market_id,
                "event_id": market.event_id,
                "scratchpad": decision.scratchpad,
                "verbal_action": decision.verbal_action,
            }
        )


class PlayedMarketLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(self, market: MarketSnapshot) -> None:
        self._logger.write(
            {
                "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                **market.as_log_dict(),
            }
        )


class GameSummaryLogger:
    def __init__(self, path: str) -> None:
        self._logger = JsonlLogger(path)

    def write(self, game: PredictionMarketGame) -> None:
        settlement = game.settlement()
        self._logger.write(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "turns_used": game.turn_count,
                "bot_count": len(active_bot_names(game.turn_order)),
                "user_turn_position": game.turn_position_for(USER_NAME),
                "user_final_pnl": settlement[USER_NAME]["pnl"],
                **game.market.as_log_dict(),
            }
        )


def main() -> int:
    settings = Settings.from_env()
    speaker = build_speaker(settings)
    cue_player = build_audio_cues(settings)
    listener = build_listener(settings, cue_player)
    bot_client = _build_bot_client(settings)
    market_repository = MarketRepository(settings)
    scratchpad_logger = ScratchpadLogger(settings.scratchpad_log_path)
    played_market_logger = PlayedMarketLogger(settings.played_markets_log_path)
    game_summary_logger = GameSummaryLogger(settings.game_summary_log_path)

    while True:
        try:
            game, bot_profiles = _build_game(settings, market_repository)
        except (MarketSelectionError, ValueError) as exc:
            print(f"Configuration error: {exc}")
            return 2

        played_market_logger.write(game.market)
        _print_intro(game, settings, speaker, listener, cue_player, bot_client)
        _run_game(
            game=game,
            bot_profiles=bot_profiles,
            bot_client=bot_client,
            scratchpad_logger=scratchpad_logger,
            listener=listener,
            cue_player=cue_player,
            speaker=speaker,
        )
        game_summary_logger.write(game)

        if not settings.auto_next_game:
            return 0

        print()
        print("Auto-next game is enabled; starting a new market.")
        print()


def _build_game(
    settings: Settings,
    market_repository: MarketRepository,
) -> tuple[PredictionMarketGame, dict[str, BotProfile]]:
    market = market_repository.select_market()
    bot_names = build_bot_names(settings.bot_count)
    bot_profiles = build_bot_profiles(bot_names)
    turn_order = _build_turn_order(settings, bot_names)
    game = PredictionMarketGame(
        market=market,
        max_turns=settings.max_turns,
        end_on_trade=settings.end_on_trade,
        allow_pass=settings.allow_pass,
        min_tighten_increment=settings.min_tighten_increment,
        turn_order=turn_order,
    )
    return game, bot_profiles


def _run_game(
    *,
    game: PredictionMarketGame,
    bot_profiles,
    bot_client: BotClient,
    scratchpad_logger: ScratchpadLogger,
    listener,
    cue_player,
    speaker,
) -> None:
    while not game.finished:
        participant = game.current_participant()
        if participant == USER_NAME:
            command_text = _read_user_action(game, listener, cue_player)
        else:
            profile = bot_profiles[participant]
            try:
                decision = bot_client.decide(
                    participant=participant,
                    profile=profile,
                    public_state=game.public_state_for(participant),
                )
            except Exception as exc:
                print(f"[bot fallback] {participant} decision failed: {exc}")
                decision = HeuristicBotClient().decide(
                    participant=participant,
                    profile=profile,
                    public_state=game.public_state_for(participant),
                )
            scratchpad_logger.write(game.market, participant, decision, game.turn_count)
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
        raise ValueError("PREDICTION_MARKET_BOT_COUNT must be at least 1.")


def _print_intro(
    game: PredictionMarketGame,
    settings: Settings,
    speaker,
    listener,
    cue_player,
    bot_client: BotClient,
) -> None:
    active_bots = ", ".join(active_bot_names(game.turn_order))
    print("Prediction-Market Odds Market-Making Simulator")
    print(f"Active bots: {active_bots}")
    print(f"Bot count: {settings.bot_count}; participants: {len(game.turn_order)}")
    print(f"Auto-next game: {settings.auto_next_game}")
    print(f"Turns: {settings.max_turns}; end on first trade: {settings.end_on_trade}")
    print(f"Turn order randomized: {settings.randomize_turn_order}")
    print(f"Turn order: {' -> '.join(game.turn_order)}")
    print(
        f"Your turn position: {game.turn_position_for(USER_NAME)} "
        f"of {len(game.turn_order)}"
    )
    print(
        f"Pass allowed: {settings.allow_pass}; "
        f"minimum tightening: {settings.min_tighten_increment:g}%"
    )
    print(
        "Underlying: current hidden Polymarket implied Yes probability, "
        f"range {format_probability(MIN_PROBABILITY)}-{format_probability(MAX_PROBABILITY)}"
    )
    print(f"Market ID: {game.market.market_id}")
    print(f"Question: {game.market.question}")
    if game.market.category:
        print(f"Category: {game.market.category}")
    if game.market.end_date:
        print(f"End date: {game.market.end_date}")
    if game.market.description:
        print(f"Description: {_shorten(game.market.description, 500)}")
    print("Hidden value: Polymarket implied Yes probability is revealed at showdown.")
    print("Commands: '20 at 35', '20% at 35%', 'hit the 20', 'lift the 35'")
    print(f"Python: {sys.executable}")
    print(bot_client.status)
    print(speaker.status)
    print(listener.status)
    print(cue_player.status)
    print(f"Market cache: {settings.cache_path}")
    print(f"Scratchpad log: {settings.scratchpad_log_path}")
    print(f"Played markets log: {settings.played_markets_log_path}")
    print(f"Game summary log: {settings.game_summary_log_path}")
    print()
    speaker.speak("Exchange", _spoken_intro(game))


def _read_user_action(game: PredictionMarketGame, listener, cue_player) -> str:
    while True:
        print("[YOUR TURN]")
        print(f"Market: {game.market.question}")
        print(f"Book: {game.order_book_line()}")
        cue_player.user_turn_started()
        command_text = listener.listen()
        if command_text:
            return command_text
        print("Please enter a trading command.")


def _spoken_intro(game: PredictionMarketGame) -> str:
    turn_word = "turn" if game.max_turns == 1 else "turns"
    pass_rule = (
        "Passing is allowed."
        if game.allow_pass
        else (
            "Passing is not allowed. You must hit, lift, or tighten the current "
            f"market by at least {format_spoken_probability(game.min_tighten_increment)} "
            "on at least one side."
        )
    )
    bot_count = len(active_bot_names(game.turn_order))
    opponent_word = "opponent" if bot_count == 1 else "opponents"
    participant_word = "participant" if len(game.turn_order) == 1 else "participants"
    bot_count_phrase = (
        f"There is {_speakable_number(bot_count)} bot {opponent_word}"
        if bot_count == 1
        else f"There are {_speakable_number(bot_count)} bot {opponent_word}"
    )
    return (
        "Prediction market game starting. "
        f"This game has {_speakable_number(game.max_turns)} {turn_word}. "
        f"{pass_rule} "
        f"{bot_count_phrase} and "
        f"{_speakable_number(len(game.turn_order))} total {participant_word}. "
        f"You are {_ordinal_phrase(game.turn_position_for(USER_NAME))} in the turn order. "
        "The hidden value is the current Polymarket implied Yes probability. "
        f"The question is: {game.market.question}."
    )


def _fallback_bot_command(game: PredictionMarketGame, participant: str, profile) -> str:
    decision = HeuristicBotClient().decide(
        participant=participant,
        profile=profile,
        public_state=game.public_state_for(participant),
    )
    return decision.verbal_action


def _last_resort_bot_command(game: PredictionMarketGame, participant: str) -> str:
    state = game.public_state_for(participant)
    best_offer = state.get("best_offer")
    if isinstance(best_offer, dict):
        return f"I lift the {best_offer['offer']} offer."

    best_bid = state.get("best_bid")
    if isinstance(best_bid, dict):
        return f"I hit the {best_bid['bid']} bid."

    return "I make 0 at 100."


def format_spoken_probability(value: float) -> str:
    numeric_value = float(value)
    return (
        f"{_speakable_number(int(numeric_value))} percent"
        if numeric_value.is_integer()
        else f"{numeric_value:g} percent"
    )


def _print_showdown(game: PredictionMarketGame, speaker) -> None:
    reveal = game.reveal()
    print("SHOWDOWN")
    print(f"Market: {game.market.question}")
    print(f"Polymarket implied Yes probability: {format_probability(game.true_value)}")
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
            "The Polymarket implied Yes probability was "
            f"{format_spoken_probability(game.true_value)}. "
            f"Your final P and L is {float(user_row['pnl']):g}."
        ),
    )


def _shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


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
