"""Tests for ingestion (prose + mermaid) and analysis. All offline."""

from __future__ import annotations

from agentic_ai_eval import analyze, ingest
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.schema import ComponentType, Severity

OFFLINE = LLMClient(api_key="")


def test_prose_ingest_extracts_components():
    spec = ingest(
        "An agent that routes intents, retrieves docs via RAG, calls tools, "
        "and applies a safety guardrail before generating an answer.",
        name="support",
        client=OFFLINE,
    )
    types = {c.type for c in spec.components}
    assert ComponentType.ROUTER in types
    assert ComponentType.RETRIEVAL in types
    assert ComponentType.TOOL_USE in types
    assert ComponentType.GUARDRAIL in types
    assert ComponentType.GENERATION in types
    assert spec.entrypoint is not None


def test_components_are_chained():
    spec = ingest("routes, then retrieves, then generates", client=OFFLINE)
    # First component has no inputs; last has no outputs.
    assert spec.components[0].inputs == []
    assert spec.components[-1].outputs == []


def test_mermaid_ingest_parses_edges():
    diagram = """graph TD
        user --> router
        router --> rag["RAG retriever"]
        rag --> gen["answer generation"]
    """
    spec = ingest(diagram, name="diagram", client=OFFLINE)
    ids = {c.id for c in spec.components}
    assert {"user", "router", "rag", "gen"} <= ids
    rag = spec.component("rag")
    assert rag is not None
    assert rag.type == ComponentType.RETRIEVAL
    assert "router" in rag.inputs
    assert "gen" in rag.outputs


def test_analysis_flags_side_effect_safety_risk():
    spec = ingest(
        "An agent with a guardrail and a tool that can issue refunds.",
        client=OFFLINE,
    )
    # Mark a tool side-effecting to trigger the critical risk path.
    tool_comp = next(c for c in spec.components if c.type == ComponentType.TOOL_USE)
    from agentic_ai_eval.schema import Tool

    tool_comp.tools.append(Tool(name="issue_refund", side_effects=True))

    analysis = analyze(spec, client=OFFLINE)
    assert analysis.failure_modes
    assert any(r.severity == Severity.CRITICAL for r in analysis.risks)


def test_constraints_extracted_from_prose():
    spec = ingest(
        "The agent must respond within 6 seconds. It must never reveal PII.",
        client=OFFLINE,
    )
    assert any("6 seconds" in c or "never reveal" in c.lower() for c in spec.constraints)
