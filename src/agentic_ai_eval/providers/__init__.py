"""Pluggable LLM providers and the resolver that selects one.

Provider selection order:

  1. An explicit ``provider`` argument / ``AGENTIC_EVAL_PROVIDER`` env var
     (one of ``anthropic`` | ``openai`` | ``gemini`` | ``offline``).
  2. Auto-detection: the first provider whose API key is present, in the order
     Anthropic, OpenAI, Gemini.
  3. Offline (deterministic) if nothing is configured.

An empty-string API key forces offline mode — handy for tests and CI.
"""

from __future__ import annotations

import os

from .anthropic_provider import DEFAULT_MODEL as ANTHROPIC_MODEL
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ModelConfig
from .gemini_provider import DEFAULT_MODEL as GEMINI_MODEL
from .gemini_provider import GeminiProvider
from .offline import OfflineProvider
from .openai_provider import DEFAULT_MODEL as OPENAI_MODEL
from .openai_provider import OpenAIProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "offline": OfflineProvider,
}

#: Each provider's default model, used when no model is configured.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": ANTHROPIC_MODEL,
    "openai": OPENAI_MODEL,
    "gemini": GEMINI_MODEL,
    "offline": "offline",
}

_ENV_KEYS: list[tuple[str, str]] = [
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("gemini", "GOOGLE_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
]


def available_providers() -> list[str]:
    """Provider names that currently have credentials in the environment."""
    found: list[str] = []
    for name, env in _ENV_KEYS:
        if os.environ.get(env) and name not in found:
            found.append(name)
    return found


def resolve_provider(
    provider: str | None = None,
    *,
    api_key: str | None = None,
) -> LLMProvider:
    """Construct the appropriate provider.

    ``api_key=""`` (empty string) forces offline mode regardless of environment.
    """
    if api_key == "":
        return OfflineProvider()

    name = (provider or os.environ.get("AGENTIC_EVAL_PROVIDER") or "").strip().lower()
    if name in _REGISTRY:
        candidate = _REGISTRY[name](api_key=api_key) if name != "offline" else OfflineProvider()
        return candidate if candidate.available or name == "offline" else OfflineProvider()

    # Auto-detect from the first present key.
    for pname in ("anthropic", "openai", "gemini"):
        cls = _REGISTRY[pname]
        inst = cls(api_key=api_key)
        if inst.available:
            return inst

    return OfflineProvider()


__all__ = [
    "LLMProvider",
    "ModelConfig",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OfflineProvider",
    "resolve_provider",
    "available_providers",
    "DEFAULT_MODELS",
]
