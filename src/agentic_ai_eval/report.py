"""Render an EvalReport (and SystemAnalysis) as Markdown or JSON.

The Markdown report is the human-facing deliverable: an overall verdict, a
per-dimension scorecard, a per-eval table, and the risk register. JSON is the
machine-facing artifact for dashboards and CI diffing.
"""

from __future__ import annotations

from .schema import EvalReport, Severity, SystemAnalysis

_SEV_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
_SEV_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🟢",
}


def render_json(report: EvalReport) -> str:
    return report.model_dump_json(indent=2)


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    verdict = "✅ PASS" if report.passed else "❌ FAIL"
    lines.append(f"# Eval Report — {report.spec_name}")
    lines.append("")
    ci = ""
    if report.ci_low is not None and report.ci_high is not None:
        ci = f" _(95% CI [{report.ci_low:.2f}, {report.ci_high:.2f}])_"
    lines.append(f"**Verdict:** {verdict}  |  **Overall score:** {report.overall_score:.2f}{ci}")
    grader = f"{report.provider or 'offline'} · {report.model or '—'}"
    lines.append(f"_Generated {report.created_at.isoformat()} · grader: {grader}_")
    if report.num_pending_review:
        lines.append("")
        lines.append(
            f"> ⏳ **{report.num_pending_review} case(s) awaiting human review.** "
            "Export the queue with `agentic-eval review export`."
        )
    lines.append("")

    # Per-dimension scorecard
    dims = report.by_dimension()
    if dims:
        lines.append("## Scorecard by dimension")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|---|---|")
        for dim, score in sorted(dims.items()):
            lines.append(f"| {dim} | {_bar(score)} {score:.2f} |")
        lines.append("")

    # Per-eval table
    lines.append("## Evaluations")
    lines.append("")
    lines.append("| Eval | Target | Dimension | Score | Threshold | Cases | Result |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(report.results, key=lambda x: (x.passed, x.score)):
        status = "✅" if r.passed else "❌"
        lines.append(
            f"| `{r.eval_id}` | {r.target_component} | {r.dimension.value} | "
            f"{r.score:.2f} | {r.pass_threshold:.2f} | {r.num_passed}/{r.num_cases} | {status} |"
        )
    lines.append("")

    # Failing-case detail (the actionable part)
    failing = [r for r in report.results if not r.passed]
    if failing:
        lines.append("## Failing evals — detail")
        lines.append("")
        for r in failing:
            lines.append(f"### ❌ `{r.eval_id}` ({r.dimension.value}, target: {r.target_component})")
            for c in r.case_results:
                if c.passed:
                    continue
                lines.append(f"- **{c.case_id}** — score {c.score:.2f}")
                for g in c.grader_results:
                    mark = "✓" if g.passed else "✗"
                    lines.append(f"    - {mark} `{g.kind.value}` ({g.score:.2f}): {g.rationale}")
            lines.append("")

    if report.analysis:
        lines.append(render_analysis_markdown(report.analysis))

    return "\n".join(lines)


def render_analysis_markdown(analysis: SystemAnalysis) -> str:
    lines: list[str] = ["## Risk register", ""]
    if analysis.risks:
        lines.append("| Severity | Risk | Mitigation |")
        lines.append("|---|---|---|")
        for risk in sorted(analysis.risks, key=lambda r: _SEV_ORDER.get(r.severity, 9)):
            emoji = _SEV_EMOJI.get(risk.severity, "")
            lines.append(f"| {emoji} {risk.severity.value} | {risk.title} | {risk.mitigation or '—'} |")
        lines.append("")
    else:
        lines.append("_No risks identified._\n")

    if analysis.failure_modes:
        lines.append("## Failure modes")
        lines.append("")
        for fm in sorted(analysis.failure_modes, key=lambda f: _SEV_ORDER.get(f.severity, 9)):
            emoji = _SEV_EMOJI.get(fm.severity, "")
            lines.append(f"- {emoji} **{fm.title}** (`{fm.component_id}`): {fm.description}")
            if fm.detection:
                lines.append(f"    - _detection:_ {fm.detection}")
        lines.append("")

    return "\n".join(lines)


def _bar(score: float, width: int = 10) -> str:
    filled = round(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)
