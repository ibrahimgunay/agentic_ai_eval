"""Grader implementations: each turns (case, trace) into a GraderResult.

Graders are the unit of measurement. We support deterministic graders
(exact/contains/regex/json-schema/tool-trajectory/numeric-threshold) and an
LLM-as-judge grader that scores against a rubric using structured outputs.

The judge degrades gracefully: offline (no API key) it returns a neutral,
clearly-labeled score so a suite still *runs* end to end — but the result is
marked so reports never mistake a stub for a real measurement.
"""

from __future__ import annotations

import json
import re

from ..llm import LLMClient
from ..schema import EvalCase, Grader, GraderKind, GraderResult, Trace
from . import metrics
from .judge import judge as run_jury


def grade(grader: Grader, case: EvalCase, trace: Trace, *, client: LLMClient | None = None) -> GraderResult:
    """Dispatch to the right grader implementation."""
    kind = grader.kind
    if kind == GraderKind.EXACT_MATCH:
        return _exact_match(grader, trace)
    if kind == GraderKind.CONTAINS:
        return _contains(grader, trace)
    if kind == GraderKind.REGEX:
        return _regex(grader, trace)
    if kind == GraderKind.JSON_SCHEMA:
        return _json_schema(grader, trace)
    if kind == GraderKind.TOOL_TRAJECTORY:
        return _tool_trajectory(grader, case, trace)
    if kind == GraderKind.NUMERIC_THRESHOLD:
        return _numeric_threshold(grader, case, trace)
    if kind == GraderKind.LLM_JUDGE:
        return _llm_judge(grader, case, trace, client or LLMClient())
    if kind == GraderKind.HUMAN:
        return _human_pending(grader)
    raise ValueError(f"Unknown grader kind: {kind}")


def _result(
    grader: Grader,
    score: float,
    passed: bool,
    rationale: str,
    *,
    source: str = "auto",
    uncertainty: float | None = None,
    pending: bool = False,
) -> GraderResult:
    return GraderResult(
        kind=grader.kind,
        score=max(0.0, min(1.0, score)),
        weight=grader.weight,
        passed=passed,
        rationale=rationale,
        source=source,
        uncertainty=uncertainty,
        pending=pending,
    )


def _exact_match(grader: Grader, trace: Trace) -> GraderResult:
    expected = (grader.expected or "").strip()
    got = (trace.output or "").strip()
    ok = expected == got
    return _result(grader, 1.0 if ok else 0.0, ok, "exact match" if ok else "output != expected")


def _contains(grader: Grader, trace: Trace) -> GraderResult:
    expected = (grader.expected or "").strip().lower()
    ok = expected in (trace.output or "").lower()
    return _result(grader, 1.0 if ok else 0.0, ok, f"substring {'found' if ok else 'missing'}: {expected!r}")


def _regex(grader: Grader, trace: Trace) -> GraderResult:
    pattern = grader.pattern or grader.expected or ""
    try:
        ok = bool(re.search(pattern, trace.output or ""))
    except re.error as e:
        return _result(grader, 0.0, False, f"bad regex: {e}")
    return _result(grader, 1.0 if ok else 0.0, ok, f"pattern {'matched' if ok else 'no match'}")


def _json_schema(grader: Grader, trace: Trace) -> GraderResult:
    raw = (trace.output or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return _result(grader, 0.0, False, f"output is not valid JSON: {e}")

    schema = grader.json_schema or {}
    required = schema.get("required", [])
    props = schema.get("properties", {})
    if not isinstance(data, dict):
        return _result(grader, 0.0, False, "JSON root is not an object")

    missing = [k for k in required if k not in data]
    type_errors = [k for k, spec in props.items() if k in data and not _json_type_ok(data[k], spec.get("type"))]

    total_checks = max(1, len(required) + len(type_errors))
    failed = len(missing) + len(type_errors)
    score = max(0.0, 1.0 - failed / total_checks)
    ok = not missing and not type_errors
    detail = []
    if missing:
        detail.append(f"missing: {missing}")
    if type_errors:
        detail.append(f"type errors: {type_errors}")
    return _result(grader, score, ok, "valid against schema" if ok else "; ".join(detail))


def _tool_trajectory(grader: Grader, case: EvalCase, trace: Trace) -> GraderResult:
    expected = grader.expected_tools if grader.expected_tools is not None else case.expected_tools
    score = metrics.tool_trajectory_match(expected, trace.tool_calls)
    ok = score >= 0.999
    return _result(
        grader, score, ok,
        f"expected {expected} vs actual {trace.tool_calls} (LCS ratio {score:.2f})",
    )


def _numeric_threshold(grader: Grader, case: EvalCase, trace: Trace) -> GraderResult:
    metric = grader.metric or ""
    value = metrics.read_metric(metric, trace, case)
    threshold = grader.threshold if grader.threshold is not None else 0.0
    if grader.direction == "lte":
        ok = value <= threshold
    else:
        ok = value >= threshold
    return _result(
        grader, 1.0 if ok else 0.0, ok,
        f"{metric}={value:.3f} {grader.direction} {threshold}",
    )


def _llm_judge(grader: Grader, case: EvalCase, trace: Trace, client: LLMClient) -> GraderResult:
    verdict = run_jury(
        grader.rubric,
        case,
        trace,
        client,
        jury_size=grader.jury_size,
        jury_models=grader.jury_models,
    )
    source = "offline_stub" if not client.online else ("jury" if verdict.n_jurors > 1 else "auto")
    return _result(
        grader,
        verdict.score,
        verdict.passed,
        verdict.rationale or "judged",
        source=source,
        uncertainty=verdict.uncertainty,
    )


def _human_pending(grader: Grader) -> GraderResult:
    """Placeholder result for a human grader before a reviewer has weighed in.

    Scored neutrally and flagged ``pending`` so the runner can surface it in the
    review queue; once a human verdict is imported it replaces this result.
    """
    return _result(
        grader,
        0.5,
        True,
        "[pending human review] export the queue, collect a verdict, then re-import",
        source="human",
        pending=True,
    )


def _json_type_ok(value: object, json_type: str | None) -> bool:
    if json_type is None:
        return True
    mapping: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }
    expected = mapping.get(json_type)
    if expected is None:
        return True
    if json_type == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected)
