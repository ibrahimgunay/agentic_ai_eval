"""End-to-end demo of the *evaluation engineering* workflow, fully offline.

Shows the parts a model-eval team uses every day:
  1. run the pipeline on a frontier-style agent spec,
  2. persist the run to a SQL store and read it back,
  3. A/B two runs with a significance-tested regression gate,
  4. route uncertain/human cases to a review queue and merge verdicts back,
  5. report judge↔human agreement.

    python examples/research_pipeline_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentic_ai_eval import Pipeline
from agentic_ai_eval.analytics import compare_runs, score_trend
from agentic_ai_eval.human import apply_reviews, build_review_queue
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.store import EvalStore

SPEC = (Path(__file__).parent / "deep_research_agent.md").read_text()


def main() -> None:
    client = LLMClient()  # uses a provider key if present, else offline
    pipe = Pipeline(client=client)
    print(f"Mode: {'ONLINE (' + client.provider_name + ')' if pipe.online else 'OFFLINE'}\n")

    art = pipe.run(SPEC, name="deep-research", generate_code=False)
    r = art.report
    print(f"Run 1: {len(art.spec.components)} components, {len(art.suite.evals)} evals")
    print(f"  overall={r.overall_score:.2f}  95% CI [{r.ci_low:.2f}, {r.ci_high:.2f}]")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "demo.db"
        with EvalStore(db) as store:
            run_a = store.save_report(r, label="baseline")
            run_b = store.save_report(pipe.run(SPEC, name="deep-research").report, label="candidate")

            print("\nTrend:")
            for p in score_trend(store, "deep-research"):
                print(f"  {p.label:<10} {p.overall_score:.2f} {'✅' if p.passed else '❌'}")

            print("\nA/B (baseline → candidate):")
            cmp = compare_runs(store, run_a, run_b)
            print(f"  Δ overall = {cmp.overall_delta:+.3f}  (p={cmp.overall_test.p_value:.3f})")
            print(f"  significant regressions: {[d.dimension for d in cmp.regressed] or 'none'}")

    # Human-in-the-loop: queue the flagged cases, simulate verdicts, merge back.
    queue = build_review_queue(r, art.suite, art.traces)
    print(f"\nReview queue: {len(queue)} case(s) flagged for humans")
    for i, item in enumerate(queue):
        item.human_passed = i % 2 == 0
        item.human_score = 1.0 if i % 2 == 0 else 0.0
    merged = apply_reviews(r, queue)
    print(f"  after human review: overall={merged.overall_score:.2f}  pending={merged.num_pending_review}")


if __name__ == "__main__":
    main()
