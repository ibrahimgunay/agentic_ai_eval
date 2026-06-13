"""Execute an EvalSuite against per-case Traces and aggregate into an EvalReport.

A Trace is whatever the system-under-test produced for a case. The caller
supplies traces (by running their real agent), or the pipeline can synthesize
placeholder traces for a dry run that exercises the harness end to end.

Scoring is weighted-mean over graders within a case, mean over cases within an
eval, and a severity-style worst-of/mean blend is avoided in favor of a simple,
auditable weighted mean across evals for the overall score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..llm import LLMClient
from ..schema import (
    CaseResult,
    Eval,
    EvalReport,
    EvalResult,
    EvalSuite,
    SystemAnalysis,
    Trace,
)
from ..stats import mean_ci
from . import graders as grader_mod

# A TraceProvider maps a case id -> Trace. This is the seam where a team plugs
# in their actual agent. Anything callable with the same shape works.
TraceProvider = Callable[[str], Trace] | Mapping[str, Trace]


def materialize_traces(suite: EvalSuite, traces: TraceProvider | None = None) -> dict[str, Trace]:
    """Resolve a trace for every case id in the suite into a plain dict.

    Handy for persisting exactly what was graded (so a human review queue can
    show the real outputs) and for re-running graders without re-running the
    agent.
    """
    get_trace = _normalize_provider(traces)
    return {case.id: get_trace(case.id) for ev in suite.evals for case in ev.cases}


def run_suite(
    suite: EvalSuite,
    traces: TraceProvider | None = None,
    *,
    client: LLMClient | None = None,
    analysis: SystemAnalysis | None = None,
) -> EvalReport:
    client = client or LLMClient()
    get_trace = _normalize_provider(traces)

    results: list[EvalResult] = []
    for ev in suite.evals:
        results.append(_run_eval(ev, get_trace, client))

    overall = _weighted_overall(results)
    overall_ci = mean_ci([r.score for r in results]) if results else None
    return EvalReport(
        spec_name=suite.spec_name,
        overall_score=overall,
        passed=all(r.passed for r in results) if results else False,
        results=results,
        analysis=analysis,
        provider=client.provider_name,
        model=client.model,
        ci_low=overall_ci.low if overall_ci else None,
        ci_high=overall_ci.high if overall_ci else None,
    )


def _run_eval(ev: Eval, get_trace: Callable[[str], Trace], client: LLMClient) -> EvalResult:
    case_results: list[CaseResult] = []
    for case in ev.cases:
        trace = get_trace(case.id)
        graders = ev.graders_for(case)
        grs = [grader_mod.grade(g, case, trace, client=client) for g in graders]

        if grs:
            total_w = sum(g.weight for g in grs) or 1.0
            score = sum(g.score * g.weight for g in grs) / total_w
            passed = all(g.passed for g in grs)
        else:
            score, passed = 0.0, False

        case_results.append(
            CaseResult(case_id=case.id, score=score, passed=passed, grader_results=grs)
        )

    eval_score = sum(c.score for c in case_results) / len(case_results) if case_results else 0.0
    ci = mean_ci([c.score for c in case_results]) if case_results else None
    return EvalResult(
        eval_id=ev.id,
        target_component=ev.target_component,
        dimension=ev.dimension,
        score=eval_score,
        passed=eval_score >= ev.pass_threshold,
        pass_threshold=ev.pass_threshold,
        case_results=case_results,
        ci_low=ci.low if ci else None,
        ci_high=ci.high if ci else None,
    )


def _weighted_overall(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def _normalize_provider(traces: TraceProvider | None) -> Callable[[str], Trace]:
    if traces is None:
        return _placeholder_trace
    if isinstance(traces, Mapping):
        return lambda cid: traces.get(cid) or _placeholder_trace(cid)
    return traces  # already callable


def _placeholder_trace(case_id: str) -> Trace:
    """Dry-run trace: empty output, zero cost/steps. Exercises the harness so a
    team can validate wiring before connecting their real agent."""
    return Trace(case_id=case_id, output="", tool_calls=[], steps=1, latency_ms=0.0,
                 input_tokens=0, output_tokens=0)
