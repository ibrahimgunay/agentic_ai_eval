"""Provider resolution and the LLMClient facade. All offline."""

from __future__ import annotations

import pytest

from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.providers import (
    OfflineProvider,
    available_providers,
    resolve_provider,
)
from agentic_ai_eval.providers.base import augment_system_for_json, extract_json


def test_empty_api_key_forces_offline():
    assert resolve_provider(api_key="").name == "offline"
    client = LLMClient(api_key="")
    assert client.online is False
    assert client.provider_name == "offline"


def test_explicit_offline_provider():
    client = LLMClient(provider="offline")
    assert client.online is False


def test_unknown_keys_resolve_offline(monkeypatch):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "AGENTIC_EVAL_PROVIDER"):
        monkeypatch.delenv(env, raising=False)
    assert resolve_provider().name == "offline"
    assert available_providers() == []


def test_available_providers_detects_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert "openai" in available_providers()


def test_offline_provider_returns_none_and_empty():
    p = OfflineProvider()
    from agentic_ai_eval.providers import ModelConfig

    cfg = ModelConfig(model="x")
    assert p.parse_json(system="s", user="u", schema={}, config=cfg) is None
    assert p.complete_text(system="s", user="u", config=cfg) == ""


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1, "b": 2} — done.') == {"a": 1, "b": 2}
    assert extract_json("no json here") is None


def test_augment_system_includes_schema():
    out = augment_system_for_json("base", {"type": "object"})
    assert "JSON" in out and "object" in out


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_named_provider_without_key_degrades_offline(provider, monkeypatch):
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    # No credentials -> resolver must hand back an offline provider, not crash.
    assert resolve_provider(provider).name == "offline"
