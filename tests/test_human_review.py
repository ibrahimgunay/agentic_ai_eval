"""Human-in-the-loop round trip + judge calibration. All offline."""

from __future__ import annotations

from agentic_ai_eval.evals import run_suite
from agentic_ai_eval.evals.judge import fit_calibration
from agentic_ai_eval.human import (
    apply_reviews,
    build_review_queue,
    judge_human_agreement,
    read_verdicts,
    write_queue,
)
from agentic_ai_eval.llm import LLMClient
from agentic_ai_eval.schema import (
    Eval,
    EvalCase,
    EvalDimension,
    EvalSuite,
    Grader,
    GraderKind,
    Trace,
)

OFFLINE = LLMClient(api_key="")


def _human_suite() -> EvalSuite:
    return EvalSuite(
        spec_name="hitl",
        evals=[
            Eval(
                id="quality",
                target_component="system",
                dimension=EvalDimension.CORRECTNESS,
                description="Human quality check.",
                graders=[Grader(kind=GraderKind.HUMAN, rubric="Is this answer good?")],
                cases=[EvalCase(id="q0", input="hi"), EvalCase(id="q1", input="bye")],
            )
        ],
    )


def test_human_grader_marks_cases_pending():
    report = run_suite(_human_suite(), client=OFFLINE)
    assert report.num_pending_review == 2
    assert report.results[0].has_pending_review


def test_review_queue_export_import_apply(tmp_path):
    suite = _human_suite()
    traces = {"q0": Trace(case_id="q0", output="great answer"),
              "q1": Trace(case_id="q1", output="bad answer")}
    report = run_suite(suite, traces, client=OFFLINE)

    items = build_review_queue(report, suite, traces)
    assert len(items) == 2
    assert all(it.reason_queued == "human_grader" for it in items)
    assert items[0].output == "great answer"

    # Reviewer fills in verdicts.
    items[0].human_score, items[0].human_passed = 1.0, True
    items[1].human_score, items[1].human_passed = 0.0, False

    path = write_queue(items, tmp_path / "q.jsonl")
    loaded = read_verdicts(path)
    assert sum(1 for v in loaded if v.reviewed) == 2

    updated = apply_reviews(report, loaded)
    assert updated.num_pending_review == 0
    # One pass, one fail -> eval mean 0.5.
    assert abs(updated.results[0].score - 0.5) < 1e-9
    assert updated.results[0].ci_low is not None


def test_csv_round_trip(tmp_path):
    suite = _human_suite()
    report = run_suite(suite, client=OFFLINE)
    items = build_review_queue(report, suite)
    for it in items:
        it.human_passed = True
    path = write_queue(items, tmp_path / "q.csv")
    loaded = read_verdicts(path)
    assert all(v.human_passed for v in loaded)


def test_fit_calibration_recovers_linear_map():
    judge = [0.0, 0.25, 0.5, 0.75, 1.0]
    human = [0.1, 0.3, 0.5, 0.7, 0.9]  # human ≈ 0.8*judge + 0.1
    cal = fit_calibration(judge, human)
    assert abs(cal.scale - 0.8) < 1e-6
    assert abs(cal.bias - 0.1) < 1e-6
    assert cal.r > 0.99
    assert abs(cal.apply(0.5) - 0.5) < 1e-6


def test_judge_human_agreement_none_without_overlap():
    report = run_suite(_human_suite(), client=OFFLINE)
    # No LLM_JUDGE graders -> nothing to compare against.
    assert judge_human_agreement(report, []) is None
