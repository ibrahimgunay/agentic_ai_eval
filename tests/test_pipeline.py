"""End-to-end pipeline tests, offline."""

from __future__ import annotations

from agentic_ai_eval import Pipeline
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.report import render_markdown
from agentic_ai_eval.schema import Trace

OFFLINE = LLMClient(api_key="")

DESC = (
    "A support agent that routes intents, retrieves help-center docs (RAG), "
    "calls tools including a side-effecting refund tool, applies a safety "
    "guardrail, and returns JSON. Must respond within 6 seconds."
)


def test_pipeline_runs_offline_end_to_end():
    pipe = Pipeline(client=OFFLINE)
    assert pipe.online is False
    art = pipe.run(DESC, name="support")
    assert art.spec.components
    assert art.suite.evals
    assert art.report is not None
    assert 0.0 <= art.report.overall_score <= 1.0
    # Every eval produced at least one case.
    assert all(ev.cases for ev in art.suite.evals)


def test_pipeline_grades_supplied_traces():
    pipe = Pipeline(client=OFFLINE)
    art = pipe.run(DESC, name="support", generate_code=False)

    # Build passing JSON-shaped traces for every case.
    traces = {
        case.id: Trace(
            case_id=case.id,
            output='{"answer": "ok", "action_taken": "none"}',
            tool_calls=[],
            steps=1,
            latency_ms=500,
            input_tokens=100,
            output_tokens=50,
        )
        for ev in art.suite.evals
        for case in ev.cases
    }
    from agentic_ai_eval.evals import run_suite

    report = run_suite(art.suite, traces, client=OFFLINE, analysis=art.analysis)
    # JSON-schema and latency/cost evals should pass with these traces.
    latency = next((r for r in report.results if r.dimension.value == "latency"), None)
    assert latency is None or latency.passed
    md = render_markdown(report)
    assert "Eval Report" in md and "Risk register" in md


def test_scaffold_produces_importable_agent(tmp_path):
    pipe = Pipeline(client=OFFLINE)
    art = pipe.run(DESC, name="support-agent")
    assert "agent.py" in art.scaffold_files
    assert "eval_harness.py" in art.scaffold_files
    # The generated agent.py must be syntactically valid Python.
    compile(art.scaffold_files["agent.py"], "agent.py", "exec")
    compile(art.scaffold_files["eval_harness.py"], "eval_harness.py", "exec")


def test_write_artifacts(tmp_path):
    pipe = Pipeline(client=OFFLINE)
    art = pipe.run(DESC, name="support")
    out = pipe.write_artifacts(art, tmp_path / "run")
    assert (out / "spec.json").exists()
    assert (out / "suite.json").exists()
    assert (out / "report.md").exists()
    assert (out / "MANIFEST.json").exists()
    assert (out / "generated" / "agent.py").exists()
