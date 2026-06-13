"""Statistical rigor for eval scores — the part that separates a dashboard from
a measurement.

A single pass-rate number is a point estimate over a finite, noisy sample. To
make ship/no-ship and regression decisions defensibly you need:

  * **Confidence intervals** on every aggregate score (how much could this move
    if we resampled the eval set?).
  * **Significance tests** when comparing two runs (is model B *really* better,
    or is it sampling noise?).
  * **Inter-rater agreement** between an LLM judge and human reviewers (can we
    trust the judge to stand in for a human?).

Everything here is dependency-free (pure stdlib ``math`` + ``random``) and
deterministic given a seed, so it runs anywhere the rest of the pipeline runs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """A point estimate with a (lower, upper) confidence interval."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #


def mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for the mean of continuous scores in [0, 1].

    The bootstrap makes no normality assumption, which matters for the bimodal,
    clipped score distributions typical of rubric grading.
    """
    if not values:
        return Interval(0.0, 0.0, 0.0, confidence)
    point = sum(values) / len(values)
    if len(values) == 1:
        return Interval(point, point, point, confidence)

    rng = random.Random(seed)
    n = len(values)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    alpha = 1.0 - confidence
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return Interval(point, lo, hi, confidence)


def wilson_interval(successes: int, n: int, *, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial pass rate.

    Far better than the textbook normal approximation for the small n and
    near-0/near-1 rates that dominate safety evals.
    """
    if n == 0:
        return Interval(0.0, 0.0, 1.0, confidence)
    z = _z_for_confidence(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half), confidence)


# --------------------------------------------------------------------------- #
# Significance testing (run-to-run comparison / regression gating)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComparisonResult:
    delta: float            # rate_b - rate_a
    p_value: float
    significant: bool
    z: float


def two_proportion_ztest(
    succ_a: int, n_a: int, succ_b: int, n_b: int, *, alpha: float = 0.05
) -> ComparisonResult:
    """Two-sided two-proportion z-test: did the pass rate move for real?

    Use this to gate CI on regressions (B = candidate, A = baseline) without
    crying wolf over sampling noise.
    """
    if n_a == 0 or n_b == 0:
        return ComparisonResult(0.0, 1.0, False, 0.0)
    p_a, p_b = succ_a / n_a, succ_b / n_b
    pool = (succ_a + succ_b) / (n_a + n_b)
    se = math.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return ComparisonResult(p_b - p_a, 1.0, False, 0.0)
    z = (p_b - p_a) / se
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return ComparisonResult(p_b - p_a, p_value, p_value < alpha, z)


# --------------------------------------------------------------------------- #
# Inter-rater agreement (judge vs. human, human vs. human)
# --------------------------------------------------------------------------- #


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's kappa for two raters over paired categorical labels.

    Returns chance-corrected agreement in (-1, 1]; ~1.0 = excellent, ~0 = no
    better than chance. The right metric for "can the LLM judge replace a human?"
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    count_a = {c: labels_a.count(c) / n for c in categories}
    count_b = {c: labels_b.count(c) / n for c in categories}
    expected = sum(count_a[c] * count_b[c] for c in categories)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Pearson r between two continuous score series (e.g. judge vs. human)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _z_for_confidence(confidence: float) -> float:
    table = {0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    if confidence in table:
        return table[confidence]
    # Inverse-CDF via bisection for arbitrary confidence levels.
    target = 1 - (1 - confidence) / 2
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
