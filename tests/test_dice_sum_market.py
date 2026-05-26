from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

from games.dice_sum_market.config import Settings
from games.dice_sum_market.engine import MarketMakingGame, parse_trading_command
from games.dice_sum_market.models import USER_NAME


class TradingCommandParserTests(unittest.TestCase):
    def test_parses_quote_commands(self) -> None:
        cases = [
            ("16 at 18", 16.0, 18.0),
            ("bid 16 offer 18", 16.0, 18.0),
            ("17.5 at 19", 17.5, 19.0),
        ]

        for text, expected_bid, expected_offer in cases:
            with self.subTest(text=text):
                command = parse_trading_command(text)

                self.assertEqual(command.kind, "quote")
                self.assertEqual(command.bid, expected_bid)
                self.assertEqual(command.offer, expected_offer)

    def test_parses_pass_hit_and_lift_commands(self) -> None:
        self.assertEqual(parse_trading_command("pass").kind, "pass")

        hit = parse_trading_command("hit the 16 bid")
        self.assertEqual(hit.kind, "hit")
        self.assertEqual(hit.price, 16.0)

        lift = parse_trading_command("lift the 18 offer")
        self.assertEqual(lift.kind, "lift")
        self.assertEqual(lift.price, 18.0)


class MarketMakingGameTests(unittest.TestCase):
    def test_rejects_invalid_quote_without_advancing_turn(self) -> None:
        game = MarketMakingGame(
            max_turns=5,
            turn_order=("Maker", USER_NAME),
            rng=random.Random(1),
        )

        result = game.apply_action("Maker", "4 at 31")

        self.assertFalse(result.accepted)
        self.assertEqual(game.turn_count, 0)
        self.assertEqual(game.quotes, {})

    def test_crossed_quote_executes_trade_and_settles_pnl(self) -> None:
        game = MarketMakingGame(
            max_turns=5,
            turn_order=("Maker", USER_NAME),
            rng=random.Random(1),
        )

        quote_result = game.apply_action("Maker", "16 at 18")
        trade_result = game.apply_action(USER_NAME, "18 at 20")

        self.assertTrue(quote_result.accepted)
        self.assertTrue(trade_result.accepted)
        self.assertIsNotNone(trade_result.trade)
        self.assertEqual(trade_result.trade.buyer, USER_NAME)
        self.assertEqual(trade_result.trade.seller, "Maker")
        self.assertEqual(trade_result.trade.price, 18)
        self.assertEqual(game.positions[USER_NAME], 1)
        self.assertEqual(game.positions["Maker"], -1)

        settlement = game.settlement()
        self.assertEqual(
            settlement[USER_NAME]["pnl"],
            round(-18 + game.true_value, 2),
        )
        self.assertEqual(
            settlement["Maker"]["pnl"],
            round(18 - game.true_value, 2),
        )


class DiceGameConfigTests(unittest.TestCase):
    def test_default_log_paths_are_game_local(self) -> None:
        with (
            patch("games.dice_sum_market.config.load_repo_env", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            settings = Settings.from_env()

        self.assertEqual(
            settings.scratchpad_log_path,
            "games/dice_sum_market/logs/scratchpads.jsonl",
        )
        self.assertEqual(
            settings.game_summary_log_path,
            "games/dice_sum_market/logs/game_summaries.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
