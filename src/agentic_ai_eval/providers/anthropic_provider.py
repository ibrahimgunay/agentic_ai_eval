"""Anthropic (Claude) provider.

Uses the official ``anthropic`` SDK. Structured output is obtained by asking the
model for a single JSON object matching the target schema and extracting it
robustly — this keeps behaviour identical to the OpenAI and Gemini providers and
tolerant of SDK-version differences.
"""

from __future__ import annotations

import os

from .base import LLMProvider, ModelConfig, augment_system_for_json, extract_json

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception:  # pragma: no cover - import/credential issues -> unavailable
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def parse_json(self, *, system: str, user: str, schema: dict, config: ModelConfig) -> dict | None:
        text = self._message(
            system=augment_system_for_json(system, schema),
            # Prefill an opening brace to nudge Claude straight into JSON.
            user=user,
            config=config,
            prefill="{",
        )
        return extract_json("{" + text if not text.lstrip().startswith("{") else text)

    def complete_text(self, *, system: str, user: str, config: ModelConfig) -> str:
        return self._message(system=system, user=user, config=config)

    # ------------------------------------------------------------------ #

    def _message(self, *, system: str, user: str, config: ModelConfig, prefill: str | None = None) -> str:
        if self._client is None:
            return ""
        messages: list[dict] = [{"role": "user", "content": user}]
        if prefill is not None:
            messages.append({"role": "assistant", "content": prefill})
        try:
            resp = self._client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system,
                messages=messages,  # type: ignore[arg-type]
            )
            return "".join(
                getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
            )
        except Exception:  # pragma: no cover - any API failure degrades to empty
            return ""
