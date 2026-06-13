"""Research-grade LLM-as-judge: a calibrated jury, not a single oracle.

A single judge call at temperature 0 is a point estimate from one biased model.
Frontier eval teams harden it three ways, all implemented here:

  * **Self-consistency.** Sample the judge ``jury_size`` times and aggregate
    (median score, majority pass). The spread across samples is a free
    *uncertainty* estimate — a wide spread means the rubric is ambiguous or the
    case is genuinely borderline and deserves a human.
  * **Cross-model jury.** Ensemble judges across different models/providers so
    no single model's idiosyncrasies dominate. Family-level disagreement is one
    of the strongest signals that a verdict is unreliable.
  * **Calibration to humans.** Fit a monotone map from judge scores to human
    labels on a gold set, then apply it. This is what lets you *quantify* how
    much to trust the automated judge instead of asserting it.

All of this degrades cleanly: offline, the judge returns a neutral, clearly
labelled stub so a suite still runs end to end.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..schema import EvalCase, Trace

_JUDGE_SYSTEM = """You are a rigorous, calibrated evaluation judge. Score the \
system's output against the rubric on a 0..1 scale, where 1.0 fully satisfies \
the rubric and 0.0 fails it. Be strict and specific; reward correctness and \
faithfulness to any provided context, penalize unsupported claims. Return a \
score, a pass/fail decision, and a one- to two-sentence rationale."""


class _JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0..1 quality score against the rubric.")
    passed: bool
    rationale: str = ""


@dataclass
class JuryVerdict:
    """Aggregated verdict across one or more jurors."""

    score: float
    passed: bool
    rationale: str
    uncertainty: float | None = None       # std-dev of juror scores
    n_jurors: int = 1
    juror_scores: list[float] = field(default_factory=list)


def judge(
    rubric: str,
    case: EvalCase,
    trace: Trace,
    client: LLMClient,
    *,
    jury_size: int = 1,
    jury_models: list[str] | None = None,
) -> JuryVerdict:
    """Score one (case, trace) against a rubric, with an optional jury."""
    if not client.online:
        return JuryVerdict(
            score=0.5,
            passed=True,
            rationale="[offline] llm_judge stub — set a provider API key for real grading",
            uncertainty=None,
            n_jurors=0,
        )

    prompt = _judge_prompt(rubric, case, trace)

    # Build the juror roster: each (model, temperature) pair is one juror.
    jurors: list[tuple[str, float]] = []
    models = jury_models or [client.judge_model]
    for model in models:
        if jury_size <= 1:
            jurors.append((model, 0.0))  # deterministic single vote
        else:
            # Spread temperatures so samples are genuinely diverse, not identical.
            for i in range(jury_size):
                temp = 0.2 + 0.6 * (i / max(1, jury_size - 1))
                jurors.append((model, round(temp, 3)))

    verdicts: list[_JudgeVerdict] = []
    for model, temp in jurors:
        v = client.parse(
            schema=_JudgeVerdict,
            system=_JUDGE_SYSTEM,
            user=prompt,
            model=model,
            max_tokens=2000,
            temperature=temp,
        )
        verdicts.append(v)

    return _aggregate(verdicts)


def _aggregate(verdicts: list[_JudgeVerdict]) -> JuryVerdict:
    scores = [max(0.0, min(1.0, v.score)) for v in verdicts]
    if not scores:
        return JuryVerdict(0.5, True, "no verdicts", None, 0)
    score = statistics.median(scores)
    passed = sum(1 for v in verdicts if v.passed) > len(verdicts) / 2
    uncertainty = statistics.pstdev(scores) if len(scores) > 1 else None
    # Prefer the rationale of the juror closest to the aggregate score.
    rep = min(verdicts, key=lambda v: abs(v.score - score))
    rationale = rep.rationale or "judged"
    if len(verdicts) > 1:
        rationale = f"[jury of {len(verdicts)}, σ={uncertainty:.2f}] {rationale}"
    return JuryVerdict(
        score=score,
        passed=passed,
        rationale=rationale,
        uncertainty=uncertainty,
        n_jurors=len(verdicts),
        juror_scores=scores,
    )


def _judge_prompt(rubric: str, case: EvalCase, trace: Trace) -> str:
    parts = [f"RUBRIC:\n{rubric or 'Is the output correct and helpful for the input?'}", ""]
    parts.append(f"INPUT:\n{case.input}")
    if case.context:
        parts.append(f"\nPROVIDED CONTEXT (output must be grounded in this):\n{case.context}")
    if case.reference:
        parts.append(f"\nREFERENCE / GOLD ANSWER:\n{case.reference}")
    parts.append(f"\nSYSTEM OUTPUT:\n{trace.output}")
    if trace.tool_calls:
        parts.append(f"\nTOOL CALLS MADE: {trace.tool_calls}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Calibration of judge -> human
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Calibration:
    """Affine map ``human ≈ scale * judge + bias`` fit by least squares.

    Clamped to [0, 1] on apply. ``r`` is the correlation on the fit set — treat a
    low ``r`` as "do not trust this judge unattended".
    """

    scale: float
    bias: float
    r: float
    n: int

    def apply(self, judge_score: float) -> float:
        return max(0.0, min(1.0, self.scale * judge_score + self.bias))


def fit_calibration(judge_scores: list[float], human_scores: list[float]) -> Calibration:
    """Ordinary least squares fit of human labels on judge scores."""
    from ..stats import pearson_correlation

    n = len(judge_scores)
    if n < 2 or len(human_scores) != n:
        return Calibration(scale=1.0, bias=0.0, r=0.0, n=n)
    mx = sum(judge_scores) / n
    my = sum(human_scores) / n
    var = sum((x - mx) ** 2 for x in judge_scores)
    if var == 0:
        return Calibration(scale=1.0, bias=my - mx, r=0.0, n=n)
    cov = sum((x - mx) * (y - my) for x, y in zip(judge_scores, human_scores, strict=True))
    scale = cov / var
    bias = my - scale * mx
    r = pearson_correlation(judge_scores, human_scores)
    return Calibration(scale=scale, bias=bias, r=r, n=n)
