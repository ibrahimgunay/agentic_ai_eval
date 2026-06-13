"""Analysis over the results store: trends, regressions, and dimension drill-down.

This is the layer a model-evaluation team actually lives in — not "what did this
one run score" but "is the line going up, and is the latest dip real or noise?"
Every comparison is backed by a significance test from :mod:`stats`, so a CI
regression gate fires on signal, not on sampling jitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import ComparisonResult, two_proportion_ztest
from .store import EvalStore


@dataclass
class TrendPoint:
    run_id: str
    created_at: str
    label: str | None
    model: str | None
    overall_score: float
    passed: bool


def score_trend(store: EvalStore, spec_name: str, *, limit: int = 50) -> list[TrendPoint]:
    """Overall-score history for a spec, oldest → newest."""
    rows = store.runs(spec_name, limit=limit)
    points = [
        TrendPoint(
            run_id=r["run_id"],
            created_at=r["created_at"],
            label=r["label"],
            model=r["model"],
            overall_score=r["overall_score"] or 0.0,
            passed=bool(r["passed"]),
        )
        for r in rows
    ]
    return list(reversed(points))


@dataclass
class DimensionDelta:
    dimension: str
    baseline: float
    candidate: float
    delta: float
    comparison: ComparisonResult


@dataclass
class RegressionReport:
    spec_name: str
    baseline_run: str
    candidate_run: str
    overall_delta: float
    overall_test: ComparisonResult
    dimensions: list[DimensionDelta] = field(default_factory=list)

    @property
    def regressed(self) -> list[DimensionDelta]:
        """Dimensions that dropped by a statistically significant margin."""
        return [d for d in self.dimensions if d.delta < 0 and d.comparison.significant]

    @property
    def improved(self) -> list[DimensionDelta]:
        return [d for d in self.dimensions if d.delta > 0 and d.comparison.significant]


def compare_runs(store: EvalStore, baseline_run: str, candidate_run: str) -> RegressionReport:
    """A/B two runs with significance testing on overall and per-dimension pass rates.

    Pass rates use case-level counts (the right unit for a proportion test), so
    the verdict accounts for how many cases actually backed each number.
    """
    base = store.run(baseline_run)
    cand = store.run(candidate_run)
    spec = (cand or base or {}).get("spec_name", "")

    overall_test = _passrate_test(store, baseline_run, candidate_run)

    base_dims = _dimension_passrates(store, baseline_run)
    cand_dims = _dimension_passrates(store, candidate_run)
    deltas: list[DimensionDelta] = []
    for dim in sorted(set(base_dims) | set(cand_dims)):
        b_passed, b_total = base_dims.get(dim, (0, 0))
        c_passed, c_total = cand_dims.get(dim, (0, 0))
        b_rate = b_passed / b_total if b_total else 0.0
        c_rate = c_passed / c_total if c_total else 0.0
        test = two_proportion_ztest(b_passed, b_total, c_passed, c_total)
        deltas.append(DimensionDelta(dim, b_rate, c_rate, c_rate - b_rate, test))

    b_overall = (base or {}).get("overall_score") or 0.0
    c_overall = (cand or {}).get("overall_score") or 0.0
    return RegressionReport(
        spec_name=spec,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        overall_delta=c_overall - b_overall,
        overall_test=overall_test,
        dimensions=deltas,
    )


def judge_reliability(store: EvalStore, run_id: str) -> dict:
    """Summarize the automated judge's self-reported reliability for a run.

    Surfaces how many judge verdicts were high-uncertainty (jury disagreement)
    or still pending human review — the inputs to "should we trust this run".
    """
    rows = store.query(
        "SELECT kind, AVG(uncertainty) AS mean_unc, "
        "SUM(CASE WHEN uncertainty >= 0.2 THEN 1 ELSE 0 END) AS high_unc, "
        "SUM(pending) AS pending, COUNT(*) AS n "
        "FROM grader_results WHERE run_id = ? GROUP BY kind",
        (run_id,),
    )
    return {r["kind"]: r for r in rows}


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _passrate_test(store: EvalStore, run_a: str, run_b: str) -> ComparisonResult:
    a = store.query(
        "SELECT SUM(passed) AS p, COUNT(*) AS n FROM case_results WHERE run_id = ?", (run_a,)
    )[0]
    b = store.query(
        "SELECT SUM(passed) AS p, COUNT(*) AS n FROM case_results WHERE run_id = ?", (run_b,)
    )[0]
    return two_proportion_ztest(a["p"] or 0, a["n"] or 0, b["p"] or 0, b["n"] or 0)


def _dimension_passrates(store: EvalStore, run_id: str) -> dict[str, tuple[int, int]]:
    rows = store.query(
        "SELECT er.dimension AS dim, SUM(cr.passed) AS p, COUNT(*) AS n "
        "FROM eval_results er JOIN case_results cr "
        "  ON er.run_id = cr.run_id AND er.eval_id = cr.eval_id "
        "WHERE er.run_id = ? GROUP BY er.dimension",
        (run_id,),
    )
    return {r["dim"]: (r["p"] or 0, r["n"] or 0) for r in rows}
