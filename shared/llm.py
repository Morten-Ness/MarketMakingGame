from __future__ import annotations

import json
import re
from typing import Any, Protocol


class JsonLlmClient(Protocol):
    @property
    def status(self) -> str:
        ...

    def generate_json(self, system_instruction: str, payload: object) -> Any:
        ...

    def generate_text(self, system_instruction: str, payload: object) -> str:
        ...


class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.35) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed for this Python interpreter. "
                "Run `pip install -r requirements.txt` in the active environment "
                "or disable Gemini."
            ) from exc

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @property
    def status(self) -> str:
        return f"Gemini API ({self._model}, temperature={self._temperature})"

    def generate_json(self, system_instruction: str, payload: object) -> Any:
        response = self._client.models.generate_content(
            model=self._model,
            contents=_payload_to_contents(payload),
            config=self._types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=self._temperature,
            ),
        )
        return parse_json_response(response.text or "")

    def generate_text(self, system_instruction: str, payload: object) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=_payload_to_contents(payload),
            config=self._types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self._temperature,
            ),
        )
        return response.text or ""


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {text!r}") from exc


def _payload_to_contents(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2)
