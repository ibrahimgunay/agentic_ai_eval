"""Google Gemini provider.

Uses the ``google-generativeai`` SDK with JSON response mode
(``response_mime_type = application/json``) for structured output.
"""

from __future__ import annotations

import os

from .base import LLMProvider, ModelConfig, augment_system_for_json, extract_json

DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )
        self._genai = None
        if self._api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._genai = genai
            except Exception:  # pragma: no cover
                self._genai = None

    @property
    def available(self) -> bool:
        return self._genai is not None

    def parse_json(self, *, system: str, user: str, schema: dict, config: ModelConfig) -> dict | None:
        text = self._generate(
            system=augment_system_for_json(system, schema),
            user=user,
            config=config,
            json_mode=True,
        )
        return extract_json(text)

    def complete_text(self, *, system: str, user: str, config: ModelConfig) -> str:
        return self._generate(system=system, user=user, config=config, json_mode=False)

    # ------------------------------------------------------------------ #

    def _generate(self, *, system: str, user: str, config: ModelConfig, json_mode: bool) -> str:
        if self._genai is None:
            return ""
        gen_config: dict = {"temperature": config.temperature, "max_output_tokens": config.max_tokens}
        if json_mode:
            gen_config["response_mime_type"] = "application/json"
        try:
            model = self._genai.GenerativeModel(
                model_name=config.model,
                system_instruction=system,
                generation_config=gen_config,
            )
            resp = model.generate_content(user)
            return getattr(resp, "text", "") or ""
        except Exception:  # pragma: no cover
            return ""
