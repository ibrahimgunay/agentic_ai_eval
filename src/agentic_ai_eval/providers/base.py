"""Provider abstraction: one narrow interface every LLM backend implements.

The rest of the codebase never imports a vendor SDK directly. It talks to an
`LLMProvider` through two methods:

  * `parse_json(...)`  -> a dict conforming to a JSON schema (structured output)
  * `complete_text(...)` -> free-form text

Each concrete provider (Anthropic, OpenAI, Gemini) implements these using its
own SDK and its own native JSON/structured-output mode. The shared contract is
deliberately small so adding a fourth provider is a single file, and so every
call site stays vendor-neutral. Reproducibility is a first-class concern: an
`OfflineProvider` returns deterministic stubs when no credentials are present,
which keeps the whole pipeline runnable in CI and in tests without spend.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Generation settings passed through to a provider call."""

    model: str
    effort: str = "high"          # reasoning effort, where the provider supports it
    temperature: float = 0.0      # 0 = maximally reproducible
    max_tokens: int = 8000


class LLMProvider(ABC):
    """A vendor-neutral LLM backend.

    Implementations must be safe to construct without network access; any
    credential or import failure should surface as ``available == False`` rather
    than raising, so the pipeline can degrade to offline mode cleanly.
    """

    #: Stable identifier, e.g. "anthropic", "openai", "gemini", "offline".
    name: str = "base"

    def __init__(self, api_key: str | None = None) -> None:
        # Subclasses that need credentials override this; the base accepts the
        # argument so the resolver can construct any provider uniformly.
        self._api_key = api_key

    @property
    @abstractmethod
    def available(self) -> bool:
        """True if this provider can actually serve requests."""

    @abstractmethod
    def parse_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        config: ModelConfig,
    ) -> dict | None:
        """Return a dict matching ``schema`` (JSON Schema), or None on failure."""

    @abstractmethod
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        config: ModelConfig,
    ) -> str:
        """Return free-form model text (empty string on failure)."""


# --------------------------------------------------------------------------- #
# Shared helpers for prompt-based JSON (used by every online provider so that
# structured output behaves identically regardless of vendor SDK quirks).
# --------------------------------------------------------------------------- #

_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else. It must validate "
    "against this JSON schema:\n{schema}\n"
    "Do not wrap it in markdown fences or prose."
)


def augment_system_for_json(system: str, schema: dict) -> str:
    """Append a strict JSON-only instruction with the target schema."""
    return system + _JSON_INSTRUCTION.format(schema=json.dumps(schema, indent=2))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response.

    Tolerant of markdown fences and leading/trailing prose — the failure mode we
    actually see in practice — while still returning None when there is no
    object to be found.
    """
    if not text:
        return None
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    # Also try the substring spanning the first '{' to the last '}'.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        cand = cand.strip()
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
