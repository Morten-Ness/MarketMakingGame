from __future__ import annotations

import json
import random
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from shared.paths import resolve_repo_path

from .config import Settings
from .models import MarketSnapshot


REQUESTED_CATEGORY_TAGS: dict[str, tuple[str, ...]] = {
    "ai": ("439", "835", "817", "537"),
    "geopolitics": ("1396", "842", "366", "1363", "101426", "103027"),
    "tech": ("101999", "178"),
    "finance": ("120", "1266", "100328"),
    "science": ("103037",),
}

REQUESTED_CATEGORY_LABELS: dict[str, str] = {
    "ai": "AI",
    "geopolitics": "Geopolitics",
    "tech": "Tech",
    "finance": "Finance",
    "science": "Science",
}

CATEGORY_ALIASES = {
    "artificialintelligence": "ai",
    "geopoltics": "geopolitics",
    "geopolitical": "geopolitics",
    "technology": "tech",
    "financial": "finance",
    "climatescience": "science",
}


class MarketSelectionError(RuntimeError):
    pass


class PolymarketClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def list_markets(
        self,
        *,
        limit: int,
        offset: int,
        min_liquidity: float,
        min_volume: float,
        tag_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str | int | float] = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
            "order": "volume1mo",
            "ascending": "false",
        }
        if tag_id:
            params["tag_id"] = tag_id
            params["related_tags"] = "true"
        if min_liquidity > 0:
            params["liquidity_num_min"] = min_liquidity
        if min_volume > 0:
            params["volume_num_min"] = min_volume

        payload = self._get_json("/markets", params)
        if not isinstance(payload, list):
            raise MarketSelectionError("Polymarket /markets returned a non-list response.")
        return [item for item in payload if isinstance(item, dict)]

    def get_market(self, market_id: str) -> dict[str, Any] | None:
        payload = self._get_json(f"/markets/{urllib.parse.quote(market_id)}", {})
        return payload if isinstance(payload, dict) else None

    def _get_json(self, path: str, params: dict[str, object]) -> Any:
        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MarketMakingGame/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise MarketSelectionError(f"Could not reach Polymarket API: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise MarketSelectionError("Polymarket API returned invalid JSON.") from exc


class MarketRepository:
    def __init__(self, settings: Settings, rng: random.Random | None = None) -> None:
        self._settings = settings
        self._rng = rng or random.Random()
        self._client = PolymarketClient(settings.gamma_base_url)

    def select_market(self) -> MarketSnapshot:
        played_keys = self._read_played_keys()
        fetched = self._fetch_candidate_snapshots()
        self._write_cache(fetched)

        selected = self._select_unplayed_market(fetched, played_keys)
        if not selected:
            cached = self._read_cached_snapshots()
            selected = self._select_unplayed_market(cached, played_keys)

        if not selected:
            raise MarketSelectionError(
                "No unplayed Polymarket markets were available. "
                "Increase POLYMARKET_MARKET_FETCH_LIMIT, "
                "POLYMARKET_MARKET_FETCH_PAGES, or clear old played logs."
            )

        refreshed = self._refresh_snapshot(selected)
        return refreshed or selected

    def _select_unplayed_market(
        self,
        snapshots: list[MarketSnapshot],
        played_keys: set[str],
    ) -> MarketSnapshot | None:
        category_keys = list(_allowed_category_keys(self._settings.allowed_categories))
        self._rng.shuffle(category_keys)
        for category_key in category_keys:
            category = REQUESTED_CATEGORY_LABELS[category_key]
            for snapshot in _sort_by_monthly_volume(
                [
                    candidate
                    for candidate in snapshots
                    if (candidate.category or "").lower() == category.lower()
                ]
            ):
                if not _is_played(snapshot, played_keys):
                    return snapshot
        return None

    def _fetch_candidate_snapshots(self) -> list[MarketSnapshot]:
        snapshots: list[MarketSnapshot] = []
        seen: set[str] = set()
        category_keys = _allowed_category_keys(self._settings.allowed_categories)
        pages = max(self._settings.market_fetch_pages, 1)
        limit = max(self._settings.market_fetch_limit, 1)
        max_offset = max(self._settings.market_max_offset, 0)

        for category_key in category_keys:
            category_label = REQUESTED_CATEGORY_LABELS[category_key]
            for tag_id in REQUESTED_CATEGORY_TAGS[category_key]:
                for page_index in range(pages):
                    offset = min(page_index * limit, max_offset)
                    for payload in self._client.list_markets(
                        limit=limit,
                        offset=offset,
                        min_liquidity=self._settings.min_liquidity,
                        min_volume=self._settings.min_volume,
                        tag_id=tag_id,
                    ):
                        snapshot = snapshot_from_market_payload(
                            payload,
                            fallback_category=category_label,
                        )
                        if snapshot and snapshot.market_id not in seen:
                            snapshots.append(snapshot)
                            seen.add(snapshot.market_id)

        if not snapshots:
            raise MarketSelectionError(
                "Polymarket returned no usable binary Yes/No markets for "
                f"categories: {', '.join(self._settings.allowed_categories)}."
            )
        return _sort_by_category_then_volume(snapshots, category_keys)

    def _refresh_snapshot(self, snapshot: MarketSnapshot) -> MarketSnapshot | None:
        payload = self._client.get_market(snapshot.market_id)
        if not payload:
            return None
        return snapshot_from_market_payload(payload, fallback_category=snapshot.category)

    def _read_cached_snapshots(self) -> list[MarketSnapshot]:
        path = resolve_repo_path(self._settings.cache_path)
        if not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        markets = payload.get("markets") if isinstance(payload, dict) else None
        if not isinstance(markets, list):
            return []

        snapshots: list[MarketSnapshot] = []
        for row in markets:
            if not isinstance(row, dict):
                continue
            try:
                snapshot = _snapshot_from_cache_row(row)
                if _category_allowed(snapshot, self._settings.allowed_categories):
                    snapshots.append(snapshot)
            except (KeyError, TypeError, ValueError):
                continue
        return _sort_by_category_then_volume(
            snapshots,
            _allowed_category_keys(self._settings.allowed_categories),
        )

    def _write_cache(self, snapshots: list[MarketSnapshot]) -> None:
        path = resolve_repo_path(self._settings.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at_utc": _utc_now(),
            "source": self._settings.gamma_base_url,
            "allowed_categories": [
                REQUESTED_CATEGORY_LABELS[key]
                for key in _allowed_category_keys(self._settings.allowed_categories)
            ],
            "sort": "volume1mo descending within category",
            "markets": [snapshot.as_log_dict() for snapshot in snapshots],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_played_keys(self) -> set[str]:
        keys: set[str] = set()
        for path_text in (
            self._settings.played_markets_log_path,
            self._settings.game_summary_log_path,
        ):
            path = resolve_repo_path(path_text)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.update(_played_keys_from_payload(payload))
        return keys


def snapshot_from_market_payload(
    payload: dict[str, Any],
    fallback_category: str | None = None,
) -> MarketSnapshot | None:
    if not _truthy(payload.get("active")) or _truthy(payload.get("closed")):
        return None
    if payload.get("enableOrderBook") is False:
        return None

    market_id = str(payload.get("id") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not market_id or not question:
        return None

    outcomes = _parse_jsonish_list(payload.get("outcomes"))
    outcome_prices = _parse_jsonish_list(payload.get("outcomePrices"))
    yes_probability = _extract_yes_probability(
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        best_bid=_to_float(payload.get("bestBid")),
        best_ask=_to_float(payload.get("bestAsk")),
        last_trade_price=_to_float(payload.get("lastTradePrice")),
    )
    if yes_probability is None:
        return None

    event = _first_event(payload)
    return MarketSnapshot(
        market_id=market_id,
        question=question,
        slug=_optional_text(payload.get("slug")),
        event_id=_optional_text(event.get("id") if event else None),
        event_slug=_optional_text(event.get("slug") if event else None),
        category=(
            _optional_text(fallback_category)
            or _optional_text(payload.get("category") or (event or {}).get("category"))
        ),
        description=_optional_text(payload.get("description")),
        end_date=_optional_text(payload.get("endDateIso") or payload.get("endDate")),
        yes_probability=round(yes_probability, 4),
        best_bid=_to_float(payload.get("bestBid")),
        best_ask=_to_float(payload.get("bestAsk")),
        last_trade_price=_to_float(payload.get("lastTradePrice")),
        volume=_to_float(
            payload.get("volume1mo")
            or payload.get("volume1moClob")
            or payload.get("volumeNum")
            or payload.get("volume")
        ),
        fetched_at_utc=_utc_now(),
    )


def _extract_yes_probability(
    *,
    outcomes: list[object],
    outcome_prices: list[object],
    best_bid: float | None,
    best_ask: float | None,
    last_trade_price: float | None,
) -> float | None:
    yes_index = None
    for index, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == "yes":
            yes_index = index
            break

    if yes_index is not None and yes_index < len(outcome_prices):
        value = _to_float(outcome_prices[yes_index])
        if _valid_probability(value):
            return value

    if _valid_probability(last_trade_price):
        return last_trade_price
    if _valid_probability(best_bid) and _valid_probability(best_ask):
        return (best_bid + best_ask) / 2
    return None


def _snapshot_from_cache_row(row: dict[str, Any]) -> MarketSnapshot:
    return MarketSnapshot(
        market_id=str(row["market_id"]),
        question=str(row["question"]),
        slug=_optional_text(row.get("slug")),
        event_id=_optional_text(row.get("event_id")),
        event_slug=_optional_text(row.get("event_slug")),
        category=_optional_text(row.get("category")),
        description=_optional_text(row.get("description")),
        end_date=_optional_text(row.get("end_date")),
        yes_probability=float(row["yes_probability"]),
        best_bid=_to_float(row.get("best_bid")),
        best_ask=_to_float(row.get("best_ask")),
        last_trade_price=_to_float(row.get("last_trade_price")),
        volume=_to_float(row.get("volume_1mo") or row.get("volume")),
        fetched_at_utc=str(row.get("fetched_at_utc") or _utc_now()),
    )


def _allowed_category_keys(categories: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for category in categories:
        normalized = _normalize_category(category)
        key = CATEGORY_ALIASES.get(normalized, normalized)
        if key in REQUESTED_CATEGORY_TAGS and key not in keys:
            keys.append(key)
    if not keys:
        raise MarketSelectionError(
            "No supported Polymarket categories configured. Supported categories: "
            "AI, Geopolitics, Tech, Finance, Science."
        )
    return tuple(keys)


def _sort_by_category_then_volume(
    snapshots: list[MarketSnapshot],
    category_keys: tuple[str, ...],
) -> list[MarketSnapshot]:
    order = {
        REQUESTED_CATEGORY_LABELS[key].lower(): index
        for index, key in enumerate(category_keys)
    }
    return sorted(
        snapshots,
        key=lambda snapshot: (
            order.get((snapshot.category or "").lower(), len(order)),
            -(snapshot.volume or 0.0),
        ),
    )


def _sort_by_monthly_volume(snapshots: list[MarketSnapshot]) -> list[MarketSnapshot]:
    return sorted(snapshots, key=lambda snapshot: -(snapshot.volume or 0.0))


def _category_allowed(snapshot: MarketSnapshot, allowed_categories: tuple[str, ...]) -> bool:
    allowed = {
        REQUESTED_CATEGORY_LABELS[key].lower()
        for key in _allowed_category_keys(allowed_categories)
    }
    return (snapshot.category or "").lower() in allowed


def _normalize_category(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _parse_jsonish_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _first_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    events = payload.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return events[0]
    event = payload.get("event")
    return event if isinstance(event, dict) else None


def _played_keys_from_payload(payload: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key_name, prefix in (
        ("market_id", "market"),
        ("event_id", "event"),
        ("slug", "slug"),
        ("event_slug", "event_slug"),
    ):
        value = payload.get(key_name)
        if value:
            keys.add(f"{prefix}:{value}")

    market = payload.get("market")
    if isinstance(market, dict):
        keys.update(_played_keys_from_payload(market))
    return keys


def _is_played(snapshot: MarketSnapshot, played_keys: set[str]) -> bool:
    return any(
        key in played_keys
        for key in (
            f"market:{snapshot.market_id}",
            f"event:{snapshot.event_id}",
            f"slug:{snapshot.slug}",
            f"event_slug:{snapshot.event_slug}",
        )
        if not key.endswith(":None")
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _valid_probability(value: float | None) -> bool:
    return value is not None and 0 <= value <= 1


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
