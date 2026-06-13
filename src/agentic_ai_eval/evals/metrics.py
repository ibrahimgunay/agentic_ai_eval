"""Deterministic metrics computed over a Trace.

These are cheap, reproducible signals — no model in the loop — used both as
standalone numeric-threshold graders (latency/cost/steps) and as inputs to
aggregate scorecards.
"""

from __future__ import annotations

from ..schema import EvalCase, Trace

# Rough blended price ($ per 1M tokens) for back-of-envelope cost estimates.
# Defaults sit in the range of current frontier models; override per deployment
# via the `input_price` / `output_price` arguments to `estimated_cost_usd`.
_INPUT_PRICE_PER_MTOK = 5.0
_OUTPUT_PRICE_PER_MTOK = 25.0


def estimated_cost_usd(
    trace: Trace,
    input_price: float = _INPUT_PRICE_PER_MTOK,
    output_price: float = _OUTPUT_PRICE_PER_MTOK,
) -> float:
    it = trace.input_tokens or 0
    ot = trace.output_tokens or 0
    return (it / 1_000_000) * input_price + (ot / 1_000_000) * output_price


def tool_trajectory_match(expected: list[str], actual: list[str]) -> float:
    """Order-sensitive similarity of two tool-call sequences in [0, 1].

    Uses a longest-common-subsequence ratio so that extra/missing/ reordered
    calls are penalized proportionally rather than all-or-nothing.
    """
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    lcs = _lcs_len(expected, actual)
    return lcs / max(len(expected), len(actual))


def step_efficiency(trace: Trace, optimal_steps: int) -> float:
    """1.0 if at/under optimal, decaying as steps exceed the optimal count."""
    if trace.steps <= 0 or optimal_steps <= 0:
        return 1.0
    if trace.steps <= optimal_steps:
        return 1.0
    return max(0.0, optimal_steps / trace.steps)


def read_metric(name: str, trace: Trace, case: EvalCase | None = None) -> float:
    """Resolve a named metric from a trace for NUMERIC_THRESHOLD graders."""
    if name == "latency_ms":
        return float(trace.latency_ms or 0.0)
    if name == "steps":
        return float(trace.steps)
    if name == "cost_usd":
        return estimated_cost_usd(trace)
    if name == "input_tokens":
        return float(trace.input_tokens or 0)
    if name == "output_tokens":
        return float(trace.output_tokens or 0)
    if name == "tool_trajectory" and case is not None:
        return tool_trajectory_match(case.expected_tools, trace.tool_calls)
    return 0.0


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]
