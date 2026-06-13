"""SQLite store + analytics. All offline, in a temp database."""

from __future__ import annotations

from agentic_ai_eval import Pipeline
from agentic_ai_eval.analytics import compare_runs, judge_reliability, score_trend
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.store import EvalStore

OFFLINE = LLMClient(api_key="")
DESC = (
    "A support agent that routes intents, retrieves docs (RAG), calls a "
    "side-effecting refund tool, applies a guardrail, returns JSON."
)


def _report():
    return Pipeline(client=OFFLINE).run(DESC, name="support", generate_code=False).report


def test_save_and_read_round_trip(tmp_path):
    report = _report()
    with EvalStore(tmp_path / "e.db") as store:
        run_id = store.save_report(report, label="baseline")
        assert store.run(run_id)["spec_name"] == "support"
        assert len(store.eval_results(run_id)) == len(report.results)
        dims = store.dimension_scores(run_id)
        assert set(dims).issuperset(report.by_dimension().keys())
        assert "support" in store.specs()


def test_arbitrary_sql_query(tmp_path):
    report = _report()
    with EvalStore(tmp_path / "e.db") as store:
        store.save_report(report)
        rows = store.query("SELECT COUNT(*) AS n FROM eval_results")
        assert rows[0]["n"] == len(report.results)


def test_score_trend_orders_oldest_first(tmp_path):
    with EvalStore(tmp_path / "e.db") as store:
        store.save_report(_report(), label="r1")
        store.save_report(_report(), label="r2")
        trend = score_trend(store, "support")
        assert [p.label for p in trend] == ["r1", "r2"]


def test_compare_runs_is_significance_tested(tmp_path):
    with EvalStore(tmp_path / "e.db") as store:
        a = store.save_report(_report(), label="base")
        b = store.save_report(_report(), label="cand")
        rep = compare_runs(store, a, b)
        # Identical offline runs: no significant movement anywhere.
        assert not rep.regressed and not rep.improved
        assert rep.dimensions


def test_judge_reliability_summarizes_graders(tmp_path):
    with EvalStore(tmp_path / "e.db") as store:
        run_id = store.save_report(_report())
        rel = judge_reliability(store, run_id)
        assert isinstance(rel, dict)
