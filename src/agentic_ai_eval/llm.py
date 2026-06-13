"""Thin Anthropic client wrapper used by the LLM-backed pipeline stages.

Design goals:
  * One place that knows about model ids, effort, adaptive thinking, and
    structured outputs — so the rest of the codebase stays SDK-agnostic.
  * Graceful OFFLINE mode: with no ANTHROPIC_API_KEY, every call returns a
    deterministic, schema-valid stub. This keeps `ingest`/`analyze`/`generate`
    runnable in CI and in tests without network or spend, which is exactly the
    property you want from an eval framework (reproducibility first).

We default to Claude Opus 4.8 (`claude-opus-4-8`) with adaptive thinking and
`effort` controlled via `output_config`, per the current Anthropic API.
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

DEFAULT_MODEL = os.environ.get("AGENTIC_EVAL_MODEL", "claude-opus-4-8")
DEFAULT_JUDGE_MODEL = os.environ.get("AGENTIC_EVAL_JUDGE_MODEL", "claude-opus-4-8")
DEFAULT_EFFORT = os.environ.get("AGENTIC_EVAL_EFFORT", "high")

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wraps the Anthropic SDK with structured-output and text helpers."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        effort: str = DEFAULT_EFFORT,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.judge_model = judge_model
        self.effort = effort
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self._api_key)
            except Exception:  # pragma: no cover - import/credential issues -> offline
                self._client = None

    @property
    def online(self) -> bool:
        return self._client is not None

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
    ) -> T:
        """Return an instance of `schema` populated by the model.

        Offline, returns `schema()`-with-defaults so callers must design schemas
        that are still meaningful when empty (they are: lists default to []).
        """
        if not self.online:
            return self._offline_stub(schema)

        try:
            resp = self._client.messages.parse(  # type: ignore[union-attr]
                model=model or self.model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
            parsed = getattr(resp, "parsed_output", None)
            if isinstance(parsed, schema):
                return parsed
            return self._offline_stub(schema)
        except Exception:  # pragma: no cover - any API failure degrades to stub
            return self._offline_stub(schema)

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4000,
    ) -> str:
        if not self.online:
            return ""
        try:
            resp = self._client.messages.create(  # type: ignore[union-attr]
                model=model or self.model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except Exception:  # pragma: no cover
            return ""

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
