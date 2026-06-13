"""Human-in-the-loop evaluation: put a person in the grading loop where it
counts, and measure whether the automated judge has earned the right to stand in
for them.

The workflow is a three-step round trip:

  1. **Export** a review queue from a run — every case graded by a ``human``
     grader, plus any LLM-judge verdict the model itself flagged as uncertain or
     borderline. (JSONL for tooling, CSV for a spreadsheet.)
  2. A reviewer fills in ``human_score`` / ``human_passed`` / ``notes``.
  3. **Import** the verdicts: pending human results are replaced with the real
     scores, the report is re-aggregated, and — wherever an LLM judge and a human
     graded the same case — we report judge↔human **agreement** (Cohen's kappa,
     correlation) and a **calibration** fit. That number is how you decide how
     far to trust the judge unattended.

Nothing here needs the network; it is plain data in, data out.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .evals.judge import Calibration, fit_calibration
from .schema import EvalReport, EvalSuite, GraderKind, Trace
from .stats import cohens_kappa, mean_ci

# A case is sent to a human if it carries a human grader, or if the LLM judge's
# jury disagreed by at least this much (std-dev), or its score lands in this
# ambiguous band around the decision boundary.
_UNCERTAIN_STDEV = 0.2
_BORDERLINE = (0.4, 0.6)


@dataclass
class ReviewItem:
    """One unit of human work, plus everything needed to judge it in context."""

    eval_id: str
    case_id: str
    dimension: str
    rubric: str
    input: str = ""
    context: str = ""
    reference: str = ""
    output: str = ""
    tool_calls: str = ""
    auto_score: float | None = None
    auto_source: str = ""
    reason_queued: str = ""
    # Filled in by the reviewer:
    human_score: float | None = None
    human_passed: bool | None = None
    notes: str = ""

    @property
    def reviewed(self) -> bool:
        return self.human_score is not None or self.human_passed is not None


# --------------------------------------------------------------------------- #
# 1. Build / export the queue
# --------------------------------------------------------------------------- #


def build_review_queue(
    report: EvalReport,
    suite: EvalSuite,
    traces: Mapping[str, Trace] | None = None,
    *,
    include_uncertain: bool = True,
    all_cases: bool = False,
) -> list[ReviewItem]:
    """Collect the cases a human should look at from a completed run.

    By default this is the high-value subset: cases with a ``human`` grader, plus
    LLM-judge verdicts the jury disagreed on or that landed near the decision
    boundary. Pass ``all_cases=True`` to queue every case (full audit).
    """
    cases_by_id = {c.id: (ev, c) for ev in suite.evals for c in ev.cases}
    items: list[ReviewItem] = []

    for result in report.results:
        for cr in result.case_results:
            ev_case = cases_by_id.get(cr.case_id)
            if ev_case is None:
                continue
            ev, case = ev_case
            reason = _queue_reason(cr, include_uncertain) or ("full_audit" if all_cases else "")
            if not reason:
                continue
            rubric = next(
                (g.rubric for g in ev.graders_for(case) if g.rubric),
                ev.description,
            )
            trace = traces.get(cr.case_id) if traces else None
            items.append(
                ReviewItem(
                    eval_id=result.eval_id,
                    case_id=cr.case_id,
                    dimension=result.dimension.value,
                    rubric=rubric,
                    input=case.input,
                    context=case.context,
                    reference=case.reference or "",
                    output=trace.output if trace else "",
                    tool_calls=", ".join(trace.tool_calls) if trace else "",
                    auto_score=cr.score,
                    auto_source=_dominant_source(cr),
                    reason_queued=reason,
                )
            )
    return items


def _queue_reason(case_result, include_uncertain: bool) -> str:
    if any(g.pending for g in case_result.grader_results):
        return "human_grader"
    if not include_uncertain:
        return ""
    for g in case_result.grader_results:
        if g.kind == GraderKind.LLM_JUDGE:
            if g.uncertainty is not None and g.uncertainty >= _UNCERTAIN_STDEV:
                return "jury_disagreement"
            if _BORDERLINE[0] <= g.score <= _BORDERLINE[1]:
                return "borderline_score"
    return ""


def _dominant_source(case_result) -> str:
    return case_result.grader_results[0].source if case_result.grader_results else ""


_CSV_FIELDS = [
    "eval_id", "case_id", "dimension", "reason_queued", "auto_score", "auto_source",
    "rubric", "input", "context", "reference", "output", "tool_calls",
    "human_score", "human_passed", "notes",
]


def write_queue(items: list[ReviewItem], path: str | Path) -> Path:
    """Write the queue. ``.csv`` -> spreadsheet; anything else -> JSONL."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for it in items:
                writer.writerow({k: getattr(it, k) for k in _CSV_FIELDS})
    else:
        with path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it.__dict__, default=str) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 2/3. Read verdicts back and apply them
# --------------------------------------------------------------------------- #


def read_verdicts(path: str | Path) -> list[ReviewItem]:
    """Load a (partially) reviewed queue back into ReviewItems."""
    path = Path(path)
    rows: list[dict] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return [_row_to_item(r) for r in rows]


def _row_to_item(r: dict) -> ReviewItem:
    def num(v):
        if v in (None, "", "None"):
            return None
        return float(v)

    def boolean(v):
        if v in (None, "", "None"):
            return None
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes", "y", "pass")

    return ReviewItem(
        eval_id=r.get("eval_id", ""),
        case_id=r.get("case_id", ""),
        dimension=r.get("dimension", ""),
        rubric=r.get("rubric", ""),
        input=r.get("input", ""),
        context=r.get("context", ""),
        reference=r.get("reference", ""),
        output=r.get("output", ""),
        tool_calls=r.get("tool_calls", ""),
        auto_score=num(r.get("auto_score")),
        auto_source=r.get("auto_source", ""),
        reason_queued=r.get("reason_queued", ""),
        human_score=num(r.get("human_score")),
        human_passed=boolean(r.get("human_passed")),
        notes=r.get("notes", ""),
    )


def apply_reviews(report: EvalReport, verdicts: list[ReviewItem]) -> EvalReport:
    """Return a new report with human verdicts merged in and re-aggregated.

    For each reviewed case, the human verdict replaces any pending human grader
    result (and overrides the case score). Eval-, dimension-, and overall scores,
    plus confidence intervals, are recomputed.
    """
    by_case = {v.case_id: v for v in verdicts if v.reviewed}
    new = report.model_copy(deep=True)

    for result in new.results:
        for cr in result.case_results:
            v = by_case.get(cr.case_id)
            if v is None:
                continue
            score = v.human_score if v.human_score is not None else (1.0 if v.human_passed else 0.0)
            passed = v.human_passed if v.human_passed is not None else score >= 0.5
            for g in cr.grader_results:
                if g.kind == GraderKind.HUMAN:
                    g.score, g.passed, g.pending = score, bool(passed), False
                    g.source = "human"
                    g.rationale = v.notes or "human verdict"
            # Human is authoritative: override the case verdict.
            cr.score, cr.passed = score, bool(passed)

        scores = [c.score for c in result.case_results]
        if scores:
            result.score = sum(scores) / len(scores)
            result.passed = result.score >= result.pass_threshold
            ci = mean_ci(scores)
            result.ci_low, result.ci_high = ci.low, ci.high

    eval_scores = [r.score for r in new.results]
    if eval_scores:
        new.overall_score = sum(eval_scores) / len(eval_scores)
        new.passed = all(r.passed for r in new.results)
        ci = mean_ci(eval_scores)
        new.ci_low, new.ci_high = ci.low, ci.high
    return new


# --------------------------------------------------------------------------- #
# Judge ↔ human agreement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AgreementReport:
    n: int
    kappa: float                 # Cohen's kappa on pass/fail
    calibration: Calibration     # affine judge->human fit (carries correlation r)
    mean_abs_error: float

    def summary(self) -> str:
        return (
            f"n={self.n} · κ={self.kappa:.2f} · r={self.calibration.r:.2f} · "
            f"MAE={self.mean_abs_error:.3f} (judge {self.calibration.scale:.2f}·x"
            f"{self.calibration.bias:+.2f})"
        )


def judge_human_agreement(
    report: EvalReport, verdicts: list[ReviewItem]
) -> AgreementReport | None:
    """Compare the LLM judge against humans on every case both graded.

    Returns None if there is no overlap to measure.
    """
    auto: dict[str, float] = {}
    auto_pass: dict[str, bool] = {}
    for result in report.results:
        for cr in result.case_results:
            judge_g = next((g for g in cr.grader_results if g.kind == GraderKind.LLM_JUDGE), None)
            if judge_g is not None:
                auto[cr.case_id] = judge_g.score
                auto_pass[cr.case_id] = judge_g.passed

    pairs_score: list[tuple[float, float]] = []
    pairs_pass: list[tuple[int, int]] = []
    for v in verdicts:
        if not v.reviewed or v.case_id not in auto:
            continue
        h_score = v.human_score if v.human_score is not None else (1.0 if v.human_passed else 0.0)
        pairs_score.append((auto[v.case_id], h_score))
        h_pass = v.human_passed if v.human_passed is not None else h_score >= 0.5
        pairs_pass.append((int(auto_pass[v.case_id]), int(h_pass)))

    if not pairs_score:
        return None

    judge_scores = [a for a, _ in pairs_score]
    human_scores = [h for _, h in pairs_score]
    mae = sum(abs(a - h) for a, h in pairs_score) / len(pairs_score)
    kappa = cohens_kappa([a for a, _ in pairs_pass], [h for _, h in pairs_pass])
    cal = fit_calibration(judge_scores, human_scores)
    return AgreementReport(n=len(pairs_score), kappa=kappa, calibration=cal, mean_abs_error=mae)
