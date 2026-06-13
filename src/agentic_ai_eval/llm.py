"""Vendor-neutral LLM client used by the LLM-backed pipeline stages.

`LLMClient` is a thin facade over a pluggable :mod:`providers` backend
(Anthropic, OpenAI, or Gemini). One place knows about model ids, effort, and
structured outputs; the rest of the codebase stays SDK-agnostic and never
imports a vendor library.

Design goals:
  * **Provider-agnostic.** Set ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, or
    ``GOOGLE_API_KEY`` (or pin one with ``AGENTIC_EVAL_PROVIDER``) and the same
    pipeline runs unchanged.
  * **Graceful OFFLINE mode.** With no credentials, every call returns a
    deterministic, schema-valid stub. This keeps ``ingest``/``analyze``/
    ``generate`` runnable in CI and tests without network or spend — exactly the
    property you want from an eval framework (reproducibility first).
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from .providers import DEFAULT_MODELS, ModelConfig, resolve_provider

# Optional explicit overrides. When unset, each provider's own default is used.
DEFAULT_MODEL = os.environ.get("AGENTIC_EVAL_MODEL")
DEFAULT_JUDGE_MODEL = os.environ.get("AGENTIC_EVAL_JUDGE_MODEL")
DEFAULT_EFFORT = os.environ.get("AGENTIC_EVAL_EFFORT", "high")

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Provider-agnostic structured-output + text client.

    Args:
        provider: ``"anthropic"`` | ``"openai"`` | ``"gemini"`` | ``"offline"``.
            Defaults to ``AGENTIC_EVAL_PROVIDER`` or auto-detection by API key.
        model: Generation model id. Defaults to the provider's default model.
        judge_model: Model id used for LLM-as-judge grading (defaults to ``model``).
        effort: Reasoning effort where the provider supports it.
        api_key: Explicit key. Pass ``""`` to force offline mode.
    """

    def __init__(
        self,
        model: str | None = None,
        judge_model: str | None = None,
        effort: str = DEFAULT_EFFORT,
        api_key: str | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        self._provider = resolve_provider(provider, api_key=api_key)
        default_model = DEFAULT_MODELS.get(self._provider.name, "offline")
        self.model = model or DEFAULT_MODEL or default_model
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL or self.model
        self.effort = effort

    @property
    def online(self) -> bool:
        return self._provider.available

    @property
    def provider_name(self) -> str:
        return self._provider.name

    # ------------------------------------------------------------------ #
    # Structured output
    # ------------------------------------------------------------------ #

    def parse(
        self,
        *,
        schema: type[T],
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 8000,
        temperature: float = 0.0,
    ) -> T:
        """Return an instance of ``schema`` populated by the model.

        Offline (or on any provider failure) returns ``schema``-with-defaults, so
        callers must design schemas that are meaningful when empty (lists default
        to ``[]``). This is what lets every stage fall back deterministically.
        """
        if not self.online:
            return self._offline_stub(schema)

        config = ModelConfig(
            model=model or self.model,
            effort=self.effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = self._provider.parse_json(
            system=system, user=user, schema=schema.model_json_schema(), config=config
        )
        if data is None:
            return self._offline_stub(schema)
        try:
            return schema.model_validate(data)
        except Exception:
            return self._offline_stub(schema)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        if not self.online:
            return ""
        config = ModelConfig(
            model=model or self.model,
            effort=self.effort,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._provider.complete_text(system=system, user=user, config=config)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _offline_stub(schema: type[T]) -> T:
        """Build a schema instance using field defaults; fill required scalars."""
        values: dict = {}
        for field_name, field in schema.model_fields.items():
            if field.is_required():
                ann = field.annotation
                if ann is str:
                    values[field_name] = ""
                elif ann in (int, float):
                    values[field_name] = 0
                elif ann is bool:
                    values[field_name] = False
                else:
                    values[field_name] = None
        try:
            return schema(**values)
        except Exception:
            return schema.model_construct()  # last resort: skip validation
