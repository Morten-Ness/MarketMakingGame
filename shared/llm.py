from __future__ import annotations

import json
import re
from typing import Any, Protocol


class JsonLlmClient(Protocol):
    @property
    def status(self) -> str:
        ...

    def generate_json(
        self,
        system_instruction: str,
        payload: object,
        schema: dict[str, Any] | None = None,
    ) -> Any:
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

    def generate_json(
        self,
        system_instruction: str,
        payload: object,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        del schema
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


class OpenAIResponsesClient:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed for this Python interpreter. "
                "Run `pip install -r requirements.txt` in the active environment."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def status(self) -> str:
        return f"OpenAI Responses API ({self._model})"

    def generate_json(
        self,
        system_instruction: str,
        payload: object,
        schema: dict[str, Any] | None = None,
    ) -> Any:
        text_config: dict[str, Any]
        if schema:
            text_config = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "schema": schema,
                    "strict": True,
                }
            }
        else:
            text_config = {"format": {"type": "json_object"}}

        response = self._client.responses.create(
            model=self._model,
            instructions=system_instruction,
            input=_payload_to_contents(payload),
            text=text_config,
        )
        return parse_json_response(_response_output_text(response))

    def generate_text(self, system_instruction: str, payload: object) -> str:
        response = self._client.responses.create(
            model=self._model,
            instructions=system_instruction,
            input=_payload_to_contents(payload),
        )
        return _response_output_text(response)


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


def _response_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        output = payload.get("output") if isinstance(payload, dict) else None
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if isinstance(content_item, dict) and isinstance(
                        content_item.get("text"),
                        str,
                    ):
                        chunks.append(content_item["text"])
            if chunks:
                return "".join(chunks)

    return ""
