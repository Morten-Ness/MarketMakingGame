from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import Paper


class SemanticScholarError(RuntimeError):
    pass


class SemanticScholarClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def batch_papers(self, paper_ids: list[str], *, fields: str) -> list[Paper]:
        if not paper_ids:
            return []
        payload = self._request_json(
            "/graph/v1/paper/batch",
            method="POST",
            params={"fields": fields},
            body={"ids": paper_ids},
        )
        if not isinstance(payload, list):
            raise SemanticScholarError("Semantic Scholar batch endpoint returned non-list JSON.")

        papers: list[Paper] = []
        for item in payload:
            if isinstance(item, dict):
                paper = Paper.from_api_payload(item)
                if paper is not None:
                    papers.append(paper)
        return papers

    def search_papers(self, query: str, *, fields: str, limit: int = 10) -> list[Paper]:
        payload = self._request_json(
            "/graph/v1/paper/search",
            method="GET",
            params={"query": query, "fields": fields, "limit": limit},
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []

        papers: list[Paper] = []
        for item in data:
            if isinstance(item, dict):
                paper = Paper.from_api_payload(item)
                if paper is not None:
                    papers.append(paper)
        return papers

    def recommend_papers(
        self,
        *,
        positive_paper_ids: list[str],
        negative_paper_ids: list[str],
        fields: str,
        limit: int,
    ) -> list[Paper]:
        payload = self._request_json(
            "/recommendations/v1/papers/",
            method="POST",
            params={"fields": fields, "limit": limit},
            body={
                "positivePaperIds": positive_paper_ids,
                "negativePaperIds": negative_paper_ids,
            },
        )
        recommended = payload.get("recommendedPapers") if isinstance(payload, dict) else None
        if not isinstance(recommended, list):
            raise SemanticScholarError(
                "Semantic Scholar recommendations endpoint returned unexpected JSON."
            )

        papers: list[Paper] = []
        for item in recommended:
            if isinstance(item, dict):
                paper = Paper.from_api_payload(item)
                if paper is not None:
                    papers.append(paper)
        return papers

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        params: dict[str, object] | None = None,
        body: dict[str, object] | None = None,
    ) -> Any:
        query = urllib.parse.urlencode(params or {})
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"

        headers = {
            "Accept": "application/json",
            "User-Agent": "MarketMakingGame/0.1",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        if self._api_key:
            headers["x-api-key"] = self._api_key

        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = _read_error_message(exc)
            raise SemanticScholarError(
                f"Semantic Scholar API returned HTTP {exc.code}: {message}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SemanticScholarError(f"Could not reach Semantic Scholar API: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SemanticScholarError("Semantic Scholar API returned invalid JSON.") from exc


def _read_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return exc.reason or "unknown error"
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return str(payload)

