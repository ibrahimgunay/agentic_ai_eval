"""Run the full pipeline on the example support agent and print the report.

Works fully offline (deterministic). Set ANTHROPIC_API_KEY for LLM-backed
ingestion, analysis, case generation, and judging.

    python examples/run_example.py
"""

from __future__ import annotations

from pathlib import Path

from agentic_ai_eval import Pipeline
from agentic_ai_eval.report import render_markdown
from agentic_ai_eval.schema import Trace

HERE = Path(__file__).parent


def main() -> None:
    description = (HERE / "customer_support_agent.md").read_text()

    pipe = Pipeline()
    print(f"Mode: {'ONLINE (Claude)' if pipe.online else 'OFFLINE (deterministic)'}\n")

    artifacts = pipe.run(description, name="customer-support-agent")

    print(f"Parsed {len(artifacts.spec.components)} components:")
    for c in artifacts.spec.components:
        print(f"  • {c.id} ({c.type.value})")

    print(f"\nGenerated {len(artifacts.suite.evals)} evals.")
    severe = sum(1 for r in artifacts.analysis.risks if r.severity.value in ("high", "critical"))
    print(f"High/critical risks: {severe}")

    # Example of plugging in *real* traces for a couple of cases. In practice
    # build these from your agent (see generated/eval_harness.py).
    example_traces: dict[str, Trace] = {}
    for ev in artifacts.suite.evals:
        for case in ev.cases:
            example_traces[case.id] = Trace(
                case_id=case.id,
                output='{"answer": "Your order ships tomorrow.", "action_taken": "lookup_order"}',
                tool_calls=["lookup_order"],
                steps=1,
                latency_ms=1200.0,
                input_tokens=800,
                output_tokens=120,
            )

    from agentic_ai_eval.evals import run_suite

    report = run_suite(artifacts.suite, example_traces, analysis=artifacts.analysis)
    print("\n" + "=" * 70)
    print(render_markdown(report))

    out = pipe.write_artifacts(artifacts, HERE.parent / "runs" / "customer-support-agent")
    print(f"\nArtifacts written to: {out}")


if __name__ == "__main__":
    main()
