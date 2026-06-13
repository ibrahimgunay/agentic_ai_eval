"""Turn a free-form description, sketch, or diagram into a typed `SystemSpec`.

Three input shapes are supported:
  * Natural-language description ("a customer-support agent that routes...").
  * A Mermaid flowchart (``graph TD; user --> router; router --> rag; ...``).
  * Already-structured JSON/YAML matching SystemSpec (loaded directly).

Online (any provider key set) we ask the model to extract a SystemSpec with
structured outputs. Offline we fall back to a deterministic heuristic parser so
the pipeline always produces *something* usable — diagrams parse exactly, and
prose is keyword-mapped to component types.
"""

from __future__ import annotations

import re

from .llm import LLMClient
from .schema import Component, ComponentType, SystemSpec

_INGEST_SYSTEM = """You are a staff research engineer who designs evaluation \
systems for agentic AI features. Given a description, sketch, or diagram of an \
agentic AI pipeline, extract a precise, structured SystemSpec.

Decompose the feature into its real components using these types: planner, \
router, tool_use, retrieval, memory, subagent, guardrail, generation, \
output_formatter, human_in_loop, other. Identify the data flow (which \
component feeds which), the entrypoint, hard constraints (latency, policy, \
format), and assumptions. Be specific and faithful to the description; do not \
invent components that aren't implied."""

# Keyword -> ComponentType heuristics for the offline prose parser.
_KEYWORD_MAP: list[tuple[ComponentType, tuple[str, ...]]] = [
    (ComponentType.ROUTER, ("rout", "classif", "intent", "dispatch", "triage")),
    (ComponentType.PLANNER, ("plan", "decompos", "orchestrat", "reason", "step")),
    (ComponentType.RETRIEVAL, ("retriev", "rag", "search", "knowledge base", "vector", "embed")),
    (ComponentType.TOOL_USE, ("tool", "function call", "api", "action", "execute")),
    (ComponentType.MEMORY, ("memory", "history", "state", "context window", "scratchpad")),
    (ComponentType.SUBAGENT, ("sub-agent", "subagent", "delegate", "worker agent")),
    (ComponentType.GUARDRAIL, ("guardrail", "safety", "policy", "moderat", "filter", "validat")),
    (ComponentType.HUMAN_IN_LOOP, ("human", "approval", "confirm", "review")),
    (ComponentType.OUTPUT_FORMATTER, ("format", "structured output", "schema", "json")),
    (ComponentType.GENERATION, ("generat", "respond", "answer", "summar", "draft")),
]


def ingest(text: str, *, name: str | None = None, client: LLMClient | None = None) -> SystemSpec:
    """Parse any supported input shape into a SystemSpec."""
    client = client or LLMClient()
    text = text.strip()

    if _looks_like_mermaid(text):
        spec = parse_mermaid(text, name=name)
        if client.online:
            spec = _enrich_with_llm(text, spec, client)
        return spec

    if client.online:
        spec = client.parse(
            schema=SystemSpec,
            system=_INGEST_SYSTEM,
            user=f"Feature name hint: {name or 'unknown'}\n\nDescription/diagram:\n{text}",
        )
        if spec.components:
            if name and not spec.name:
                spec.name = name
            return spec

    return parse_prose(text, name=name)


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #

_MERMAID_EDGE = re.compile(r"([A-Za-z0-9_]+)\s*-->\s*(?:\|[^|]*\|\s*)?([A-Za-z0-9_]+)")
_MERMAID_NODE_LABEL = re.compile(r"([A-Za-z0-9_]+)\s*[\[\(\{]+\"?([^\"\]\)\}]+)\"?[\]\)\}]+")


def _looks_like_mermaid(text: str) -> bool:
    head = text.lower()
    return "-->" in text and ("graph" in head or "flowchart" in head or head.startswith("graph"))


def parse_mermaid(text: str, *, name: str | None = None) -> SystemSpec:
    """Parse a Mermaid flowchart into components + edges."""
    labels: dict[str, str] = {m.group(1): m.group(2).strip() for m in _MERMAID_NODE_LABEL.finditer(text)}
    edges = [(a, b) for a, b in _MERMAID_EDGE.findall(text)]

    node_ids: list[str] = []
    for a, b in edges:
        for n in (a, b):
            if n not in node_ids:
                node_ids.append(n)
    for nid in labels:
        if nid not in node_ids:
            node_ids.append(nid)

    components: list[Component] = []
    for nid in node_ids:
        label = labels.get(nid, nid)
        ctype = _infer_type(f"{nid} {label}")
        components.append(
            Component(
                id=_slug(nid),
                type=ctype,
                name=label,
                inputs=[_slug(a) for a, b in edges if b == nid],
                outputs=[_slug(b) for a, b in edges if a == nid],
            )
        )

    entrypoint = next((c.id for c in components if not c.inputs), components[0].id if components else None)
    return SystemSpec(
        name=name or "diagram-pipeline",
        summary="Parsed from a Mermaid flowchart.",
        components=components,
        entrypoint=entrypoint,
    )


# --------------------------------------------------------------------------- #
# Prose (offline heuristic)
# --------------------------------------------------------------------------- #


def parse_prose(text: str, *, name: str | None = None) -> SystemSpec:
    """Keyword-map a description to components when no LLM is available."""
    lowered = text.lower()
    seen: list[ComponentType] = []
    for ctype, keywords in _KEYWORD_MAP:
        if any(k in lowered for k in keywords) and ctype not in seen:
            seen.append(ctype)

    if ComponentType.GENERATION not in seen:
        seen.append(ComponentType.GENERATION)

    # Chain components in a sensible default order based on the canonical flow.
    order = [
        ComponentType.ROUTER,
        ComponentType.PLANNER,
        ComponentType.RETRIEVAL,
        ComponentType.MEMORY,
        ComponentType.TOOL_USE,
        ComponentType.SUBAGENT,
        ComponentType.GENERATION,
        ComponentType.GUARDRAIL,
        ComponentType.OUTPUT_FORMATTER,
        ComponentType.HUMAN_IN_LOOP,
    ]
    ordered = [c for c in order if c in seen]

    components: list[Component] = []
    for i, ctype in enumerate(ordered):
        cid = ctype.value
        components.append(
            Component(
                id=cid,
                type=ctype,
                name=ctype.value.replace("_", " ").title(),
                inputs=[ordered[i - 1].value] if i > 0 else [],
                outputs=[ordered[i + 1].value] if i < len(ordered) - 1 else [],
            )
        )

    return SystemSpec(
        name=name or "described-pipeline",
        summary=text[:280],
        goal=text[:280],
        components=components,
        entrypoint=components[0].id if components else None,
        constraints=_extract_constraints(text),
    )


def _enrich_with_llm(text: str, spec: SystemSpec, client: LLMClient) -> SystemSpec:
    """Let the model fill goal/constraints/descriptions on a diagram-derived spec."""
    enriched = client.parse(
        schema=SystemSpec,
        system=_INGEST_SYSTEM
        + "\n\nA component graph is already provided; preserve its ids and edges, "
        "and enrich names, descriptions, goal, and constraints.",
        user=f"Existing spec:\n{spec.model_dump_json(indent=2)}\n\nSource diagram:\n{text}",
    )
    return enriched if enriched.components else spec


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def _infer_type(text: str) -> ComponentType:
    lowered = text.lower()
    for ctype, keywords in _KEYWORD_MAP:
        if any(k in lowered for k in keywords):
            return ctype
    if "user" in lowered or "input" in lowered:
        return ComponentType.OTHER
    return ComponentType.OTHER


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower()).strip("_") or "node"


_CONSTRAINT_HINTS = ("must", "latency", "within", "<=", "seconds", "ms", "p95", "policy", "never", "always")


def _extract_constraints(text: str) -> list[str]:
    out: list[str] = []
    for line in re.split(r"[.\n]", text):
        line = line.strip()
        if line and any(h in line.lower() for h in _CONSTRAINT_HINTS):
            out.append(line)
    return out[:10]
