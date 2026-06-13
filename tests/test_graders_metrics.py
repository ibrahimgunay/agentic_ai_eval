"""Tests for graders and metrics. All deterministic / offline."""

from __future__ import annotations

from agentic_ai_eval.evals import metrics
from agentic_ai_eval.evals.graders import grade
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.schema import EvalCase, Grader, GraderKind, Trace

OFFLINE = LLMClient(api_key="")


def _case(**kw) -> EvalCase:
    return EvalCase(id="c1", input="hi", **kw)


def test_exact_match():
    g = Grader(kind=GraderKind.EXACT_MATCH, expected="42")
    assert grade(g, _case(), Trace(case_id="c1", output="42")).passed
    assert not grade(g, _case(), Trace(case_id="c1", output="43")).passed


def test_contains_case_insensitive():
    g = Grader(kind=GraderKind.CONTAINS, expected="Refund")
    assert grade(g, _case(), Trace(case_id="c1", output="we issued a refund")).passed


def test_regex_bad_pattern_fails_gracefully():
    g = Grader(kind=GraderKind.REGEX, pattern="[")  # invalid
    res = grade(g, _case(), Trace(case_id="c1", output="x"))
    assert not res.passed
    assert "bad regex" in res.rationale


def test_json_schema_validation():
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
    g = Grader(kind=GraderKind.JSON_SCHEMA, json_schema=schema)
    ok = grade(g, _case(), Trace(case_id="c1", output='{"answer": "hello"}'))
    assert ok.passed and ok.score == 1.0
    bad = grade(g, _case(), Trace(case_id="c1", output='{"wrong": 1}'))
    assert not bad.passed
    not_json = grade(g, _case(), Trace(case_id="c1", output="not json"))
    assert not not_json.passed and not_json.score == 0.0


def test_tool_trajectory_partial_credit():
    g = Grader(kind=GraderKind.TOOL_TRAJECTORY, expected_tools=["a", "b", "c"])
    perfect = grade(g, _case(), Trace(case_id="c1", tool_calls=["a", "b", "c"]))
    assert perfect.passed and perfect.score == 1.0
    partial = grade(g, _case(), Trace(case_id="c1", tool_calls=["a", "c"]))
    assert not partial.passed
    assert 0.0 < partial.score < 1.0


def test_numeric_threshold_latency():
    g = Grader(kind=GraderKind.NUMERIC_THRESHOLD, metric="latency_ms", threshold=1000, direction="lte")
    assert grade(g, _case(), Trace(case_id="c1", latency_ms=800)).passed
    assert not grade(g, _case(), Trace(case_id="c1", latency_ms=1200)).passed


def test_llm_judge_offline_is_neutral_and_labeled():
    g = Grader(kind=GraderKind.LLM_JUDGE, rubric="is it good?")
    res = grade(g, _case(), Trace(case_id="c1", output="x"), client=OFFLINE)
    assert res.score == 0.5
    assert "offline" in res.rationale.lower()


def test_lcs_metric():
    assert metrics.tool_trajectory_match([], []) == 1.0
    assert metrics.tool_trajectory_match(["a"], []) == 0.0
    assert metrics.tool_trajectory_match(["a", "b"], ["a", "b"]) == 1.0


def test_cost_estimate():
    t = Trace(case_id="c1", input_tokens=1_000_000, output_tokens=0)
    assert metrics.estimated_cost_usd(t) == 5.0
