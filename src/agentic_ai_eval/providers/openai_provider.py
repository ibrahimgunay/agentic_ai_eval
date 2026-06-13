"""OpenAI (GPT) provider.

Uses the official ``openai`` SDK with JSON mode (``response_format`` =
``json_object``) for structured output, falling back to robust extraction.
"""

from __future__ import annotations

import os

from .base import LLMProvider, ModelConfig, augment_system_for_json, extract_json

DEFAULT_MODEL = "gpt-4.1"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._client = None
        if self._api_key:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self._api_key)
            except Exception:  # pragma: no cover
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def parse_json(self, *, system: str, user: str, schema: dict, config: ModelConfig) -> dict | None:
        text = self._chat(
            system=augment_system_for_json(system, schema),
            user=user,
            config=config,
            json_mode=True,
        )
        return extract_json(text)

    def complete_text(self, *, system: str, user: str, config: ModelConfig) -> str:
        return self._chat(system=system, user=user, config=config, json_mode=False)

    # ------------------------------------------------------------------ #

    def _chat(self, *, system: str, user: str, config: ModelConfig, json_mode: bool) -> str:
        if self._client is None:
            return ""
        kwargs: dict = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception:  # pragma: no cover
            # Some models reject max_tokens/temperature; retry without the knobs.
            try:
                resp = self._client.chat.completions.create(
                    model=config.model,
                    messages=kwargs["messages"],
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                )
                return resp.choices[0].message.content or ""
            except Exception:
                return ""
