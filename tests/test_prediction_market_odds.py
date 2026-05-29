from __future__ import annotations

import unittest
import math
from dataclasses import replace

from games.prediction_market_odds.config import Settings
from games.prediction_market_odds.data import (
    MarketRepository,
    PolymarketClient,
    snapshot_from_market_payload,
)
from games.prediction_market_odds.engine import (
    LOG_ODDS_PNL_MULTIPLIER,
    PredictionMarketGame,
    log_odds_from_probability,
    parse_trading_command,
)
from games.prediction_market_odds.models import MarketSnapshot, USER_NAME


def _market(probability: float = 0.42) -> MarketSnapshot:
    return MarketSnapshot(
        market_id="123",
        question="Will the example event happen?",
        slug="example-event",
        event_id="event-1",
        event_slug="example-event-parent",
        category="Politics",
        description="Example resolution criteria.",
        end_date="2026-12-31T00:00:00Z",
        yes_probability=probability,
        best_bid=probability - 0.01,
        best_ask=probability + 0.01,
        last_trade_price=probability,
        volume=1000.0,
        fetched_at_utc="2026-05-29T00:00:00+00:00",
    )


def _market_with_identity(
    market_id: str,
    category: str,
    volume: float,
) -> MarketSnapshot:
    return MarketSnapshot(
        market_id=market_id,
        question=f"Will {market_id} happen?",
        slug=market_id,
        event_id=f"event-{market_id}",
        event_slug=f"event-{market_id}",
        category=category,
        description=None,
        end_date=None,
        yes_probability=0.42,
        best_bid=0.41,
        best_ask=0.43,
        last_trade_price=0.42,
        volume=volume,
        fetched_at_utc="2026-05-29T00:00:00+00:00",
    )


class PredictionMarketParserTests(unittest.TestCase):
    def test_parses_probability_quotes(self) -> None:
        cases = [
            ("20 at 35", 20.0, 35.0),
            ("20% at 35%", 20.0, 35.0),
            ("bid 20 offer 35", 20.0, 35.0),
        ]

        for text, expected_bid, expected_offer in cases:
            with self.subTest(text=text):
                command = parse_trading_command(text)

                self.assertEqual(command.kind, "quote")
                self.assertEqual(command.bid, expected_bid)
                self.assertEqual(command.offer, expected_offer)

    def test_parses_hit_and_lift_with_optional_prices(self) -> None:
        self.assertEqual(parse_trading_command("hit").kind, "hit")

        hit = parse_trading_command("hit the 20% bid")
        self.assertEqual(hit.kind, "hit")
        self.assertEqual(hit.price, 20.0)

        lift = parse_trading_command("lift the 35 percent offer")
        self.assertEqual(lift.kind, "lift")
        self.assertEqual(lift.price, 35.0)


class PredictionMarketGameTests(unittest.TestCase):
    def test_rejects_non_integer_probability_quotes(self) -> None:
        game = PredictionMarketGame(
            market=_market(),
            max_turns=5,
            turn_order=("Maker", USER_NAME),
        )

        result = game.apply_action("Maker", "20.5 at 35")

        self.assertFalse(result.accepted)
        self.assertEqual(game.turn_count, 0)

    def test_crossed_quote_executes_trade_and_settles_to_log_odds_pnl(self) -> None:
        game = PredictionMarketGame(
            market=_market(0.42),
            max_turns=5,
            turn_order=("Maker", USER_NAME),
        )

        quote_result = game.apply_action("Maker", "20 at 35")
        trade_result = game.apply_action(USER_NAME, "35 at 50")

        self.assertTrue(quote_result.accepted)
        self.assertTrue(trade_result.accepted)
        self.assertIsNotNone(trade_result.trade)
        self.assertEqual(game.true_value, 42.0)
        self.assertEqual(game.positions[USER_NAME], 1)
        expected_pnl = round(
            (
                log_odds_from_probability(42)
                - log_odds_from_probability(35)
            )
            * LOG_ODDS_PNL_MULTIPLIER,
            4,
        )
        self.assertEqual(game.settlement()[USER_NAME]["pnl"], expected_pnl)
        self.assertEqual(game.settlement()["Maker"]["pnl"], -expected_pnl)

    def test_log_odds_conversion_uses_natural_logit(self) -> None:
        self.assertEqual(log_odds_from_probability(50), 0.0)
        self.assertAlmostEqual(log_odds_from_probability(75), math.log(3))


class PolymarketPayloadTests(unittest.TestCase):
    def test_extracts_yes_probability_from_outcome_prices(self) -> None:
        snapshot = snapshot_from_market_payload(
            {
                "id": "123",
                "question": "Will the example event happen?",
                "slug": "example-event",
                "active": True,
                "closed": False,
                "enableOrderBook": True,
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.42", "0.58"]',
                "bestBid": "0.41",
                "bestAsk": "0.43",
                "lastTradePrice": "0.42",
                "volumeNum": "12345",
                "volume1mo": "2345",
                "events": [
                    {
                        "id": "event-1",
                        "slug": "example-event-parent",
                        "category": "Politics",
                    }
                ],
            }
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.yes_probability, 0.42)
        self.assertEqual(snapshot.yes_probability_percent, 42.0)
        self.assertEqual(snapshot.event_id, "event-1")
        self.assertEqual(snapshot.volume, 2345.0)

    def test_uses_requested_category_when_payload_category_is_empty(self) -> None:
        snapshot = snapshot_from_market_payload(
            {
                "id": "123",
                "question": "Will an AI model pass the benchmark?",
                "active": True,
                "closed": False,
                "enableOrderBook": True,
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.42", "0.58"]',
            },
            fallback_category="AI",
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.category, "AI")

    def test_market_client_requests_tag_filtered_volume_sort(self) -> None:
        client = PolymarketClient("https://example.test")
        captured = {}

        def fake_get_json(path, params):
            captured["path"] = path
            captured["params"] = params
            return []

        client._get_json = fake_get_json
        client.list_markets(
            limit=50,
            offset=0,
            min_liquidity=0,
            min_volume=0,
            tag_id="439",
        )

        self.assertEqual(captured["path"], "/markets")
        self.assertEqual(captured["params"]["tag_id"], "439")
        self.assertEqual(captured["params"]["related_tags"], "true")
        self.assertEqual(captured["params"]["order"], "volume1mo")
        self.assertEqual(captured["params"]["ascending"], "false")

    def test_repository_selects_highest_monthly_volume_in_randomized_category(self) -> None:
        settings = Settings.from_env()
        repository = MarketRepository(settings)
        snapshots = [
            _market(probability=0.2) for _ in range(4)
        ]
        snapshots[0] = _market_with_identity("ai-low", "AI", 10)
        snapshots[1] = _market_with_identity("ai-high", "AI", 100)
        snapshots[2] = _market_with_identity("finance-low", "Finance", 20)
        snapshots[3] = _market_with_identity("finance-high", "Finance", 200)

        repository._settings = replace(settings, allowed_categories=("AI",))

        selected = repository._select_unplayed_market(snapshots, set())

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.market_id, "ai-high")


if __name__ == "__main__":
    unittest.main()
