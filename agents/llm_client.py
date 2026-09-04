"""Shared LLM access: structured output, vision input, honest key handling.

Every agent in this pipeline goes through here so that model selection, JSON
schema enforcement and missing-key behaviour are defined in exactly one place.

Key policy (CLAUDE.md, environment facts): no API key is hardcoded anywhere. If
OPENAI_API_KEY is unset we raise :class:`MissingAPIKeyError` with a clear
message. The orchestrator catches it and records *which* pipeline steps could
not be verified as a result -- it never silently skips them.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from config import models


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is attempted with no API key configured."""

    def __init__(self, agent: str) -> None:
        self.agent = agent
        super().__init__(
            f"{agent}: no API key. Set the {models.ENV_API_KEY} environment variable "
            f"(and optionally {models.ENV_BASE_URL} for an OpenAI-compatible endpoint). "
            "No key is read from disk or hardcoded anywhere in this repo."
        )


class LLMResponseError(RuntimeError):
    """The provider returned something that did not satisfy the requested schema."""


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_payload, mime_type) for an image on disk."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime


def image_block(image_path: str | Path, detail: str = "high") -> dict:
    """Build a vision content block for the chat completions API."""
    payload, mime = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{payload}", "detail": detail},
    }


def text_block(text: str) -> dict:
    """Build a plain-text content block."""
    return {"type": "text", "text": text}


def get_client(agent: str):
    """Construct an OpenAI-compatible client, or raise a clear, actionable error."""
    key = models.api_key()
    if not key:
        raise MissingAPIKeyError(agent)
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "The openai package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    kwargs: dict[str, Any] = {"api_key": key}
    url = models.base_url()
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def structured_completion(
    *,
    agent: str,
    model: str,
    system_prompt: str,
    user_content: list[dict],
    schema: dict,
    schema_name: str,
    temperature: float | None = 0.0,
    client: Any = None,
) -> dict:
    """Call the model and return a dict matching ``schema``.

    Uses strict JSON-schema structured output. If the provider rejects
    ``response_format=json_schema`` we fall back to JSON-object mode with the
    schema inlined in the system prompt. Either way the action enum is
    re-checked locally by ``agents.action_vocabulary.parse_action`` at the call
    site, so an out-of-vocabulary action becomes a reported failure rather than
    a silent invention.

    ``client`` exists so tests can inject a transport double. Production callers
    leave it None.
    """
    client = client or get_client(agent)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    if temperature is not None:
        request["temperature"] = temperature

    try:
        completion = client.chat.completions.create(**request)
    except Exception as exc:  # provider rejected json_schema, or transport error
        if not _looks_like_schema_unsupported(exc):
            raise
        request["response_format"] = {"type": "json_object"}
        request["messages"] = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\nRespond with a single JSON object matching this schema exactly:\n"
                    + json.dumps(schema, indent=2)
                ),
            },
            messages[1],
        ]
        completion = client.chat.completions.create(**request)

    raw = completion.choices[0].message.content
    if not raw:
        raise LLMResponseError(f"{agent}: model returned an empty response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"{agent}: model response was not valid JSON: {raw[:400]}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError(f"{agent}: expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _looks_like_schema_unsupported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text or "unsupported" in text
