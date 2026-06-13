"""Statistical primitives. Deterministic given seeds."""

from __future__ import annotations

from agentic_ai_eval.stats import (
    cohens_kappa,
    mean_ci,
    pearson_correlation,
    two_proportion_ztest,
    wilson_interval,
)


def test_mean_ci_brackets_point_estimate():
    ci = mean_ci([0.2, 0.4, 0.6, 0.8, 1.0])
    assert abs(ci.point - 0.6) < 1e-9
    assert ci.low <= ci.point <= ci.high
    assert ci.low >= 0.0 and ci.high <= 1.0


def test_mean_ci_degenerate_cases():
    assert mean_ci([]).point == 0.0
    one = mean_ci([0.7])
    assert one.low == one.high == 0.7


def test_wilson_interval_is_tighter_with_more_data():
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)
    assert 0.0 <= narrow.low <= 0.5 <= narrow.high <= 1.0


def test_wilson_handles_zero_n():
    iv = wilson_interval(0, 0)
    assert iv.low == 0.0 and iv.high == 1.0


def test_two_proportion_ztest_detects_real_difference():
    res = two_proportion_ztest(50, 100, 90, 100)
    assert res.delta == 0.4
    assert res.significant and res.p_value < 0.05


def test_two_proportion_ztest_ignores_noise():
    res = two_proportion_ztest(50, 100, 52, 100)
    assert not res.significant


def test_cohens_kappa_perfect_and_chance():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # Total disagreement gives strongly negative kappa.
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_pearson_correlation():
    assert abs(pearson_correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0
