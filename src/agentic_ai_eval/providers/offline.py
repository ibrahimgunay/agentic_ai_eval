"""Deterministic offline provider.

Returns ``None`` for structured calls and ``""`` for text, signalling callers to
use their deterministic fallbacks (heuristic ingest, knowledge-base analysis,
template cases, neutral labelled judge). This is what makes the whole pipeline
reproducible and CI-friendly with zero credentials and zero spend.
"""

from __future__ import annotations

from .base import LLMProvider, ModelConfig


class OfflineProvider(LLMProvider):
    name = "offline"

    @property
    def available(self) -> bool:
        return False

    def parse_json(self, *, system: str, user: str, schema: dict, config: ModelConfig) -> dict | None:
        return None

    def complete_text(self, *, system: str, user: str, config: ModelConfig) -> str:
        return ""
