"""Command-line interface for agentic-ai-eval.

Core pipeline:
    agentic-eval analyze   <input>   # ingest + analyze, print risk register
    agentic-eval evals     <input>   # generate the eval suite, write suite.json
    agentic-eval run       <input>   # full pipeline (dry run) -> ./runs/<name>
    agentic-eval scaffold  <input>   # generate agent + harness code

Human-in-the-loop:
    agentic-eval review export <run_dir>   # write a reviewer queue (JSONL/CSV)
    agentic-eval review import <queue>     # merge human verdicts, show agreement

Data / analytics:
    agentic-eval db ingest <run_dir>       # load a run into the SQLite store
    agentic-eval db runs [--spec NAME]     # list runs
    agentic-eval db trend <spec>           # overall-score history
    agentic-eval db compare <a> <b>        # significance-tested A/B
    agentic-eval db query "<sql>"          # arbitrary read-only SQL
    agentic-eval serve                     # REST API (needs [server] extra)

`<input>` is a file path (.md/.txt/.mmd/.json) or an inline description string.
Provider is auto-detected from ANTHROPIC_API_KEY / OPENAI_API_KEY /
GOOGLE_API_KEY, or pin it with --provider / AGENTIC_EVAL_PROVIDER.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .analyze import analyze
from .evals import generate_suite
from .ingest import ingest
from .llm import LLMClient
from .pipeline import Pipeline
from .report import render_analysis_markdown
from .scaffold import scaffold
from .schema import EvalReport, EvalSuite, Trace

app = typer.Typer(add_completion=False, help="Evaluation pipeline for agentic AI features.")
review_app = typer.Typer(add_completion=False, help="Human-in-the-loop review queue.")
db_app = typer.Typer(add_completion=False, help="Query the results store.")
app.add_typer(review_app, name="review")
app.add_typer(db_app, name="db")
console = Console()


def _read_input(value: str) -> tuple[str, str | None]:
    """Return (text, name). If `value` is a readable file, load it; else treat
    it as an inline description."""
    p = Path(value)
    if p.exists() and p.is_file():
        return p.read_text(), p.stem
    return value, None


def _client(offline: bool, provider: str | None) -> LLMClient:
    if offline:
        return LLMClient(api_key="")
    return LLMClient(provider=provider)


def _mode_banner(client: LLMClient) -> None:
    if client.online:
        console.print(f"[dim]mode: ONLINE ({client.provider_name} · {client.model})[/dim]")
    else:
        console.print("[dim]mode: OFFLINE (deterministic)[/dim]")


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #


@app.command("analyze")
def analyze_cmd(
    input: str = typer.Argument(..., help="File path or inline description."),
    name: str | None = typer.Option(None, help="Override the system name."),
    provider: str | None = typer.Option(None, help="anthropic | openai | gemini."),
    offline: bool = typer.Option(False, help="Force offline mode (no API calls)."),
) -> None:
    """Ingest + analyze: print components and the risk register."""
    client = _client(offline, provider)
    _mode_banner(client)
    text, inferred = _read_input(input)
    spec = ingest(text, name=name or inferred, client=client)
    an = analyze(spec, client=client)

    console.print(f"[bold]{spec.name}[/bold] — {len(spec.components)} components")
    for c in spec.components:
        console.print(f"  • [cyan]{c.id}[/cyan] ({c.type.value}) → {c.outputs or '∅'}")
    console.print()
    console.print(render_analysis_markdown(an))


@app.command()
def evals(
    input: str = typer.Argument(...),
    name: str | None = typer.Option(None),
    out: Path = typer.Option(Path("suite.json"), help="Where to write the suite."),
    cases: int = typer.Option(5, help="Cases per eval (online generation)."),
    provider: str | None = typer.Option(None, help="anthropic | openai | gemini."),
    offline: bool = typer.Option(False),
) -> None:
    """Generate the eval suite and write it to disk."""
    client = _client(offline, provider)
    _mode_banner(client)
    text, inferred = _read_input(input)
    spec = ingest(text, name=name or inferred, client=client)
    an = analyze(spec, client=client)
    suite = generate_suite(spec, an, client=client, cases_per_eval=cases)
    out.write_text(suite.model_dump_json(indent=2))
    console.print(f"Wrote [green]{len(suite.evals)}[/green] evals → [bold]{out}[/bold]")
    for ev in suite.evals:
        console.print(f"  • [cyan]{ev.id}[/cyan] [{ev.dimension.value}] ({len(ev.cases)} cases)")


@app.command()
def run(
    input: str = typer.Argument(...),
    name: str | None = typer.Option(None),
    out: Path = typer.Option(Path("runs"), help="Output directory root."),
    cases: int = typer.Option(5),
    no_code: bool = typer.Option(False, help="Skip code scaffolding."),
    db: Path | None = typer.Option(None, help="Also ingest the run into this SQLite db."),
    provider: str | None = typer.Option(None, help="anthropic | openai | gemini."),
    offline: bool = typer.Option(False),
) -> None:
    """Run the full pipeline (dry-run traces) and write all artifacts."""
    client = _client(offline, provider)
    _mode_banner(client)
    text, inferred = _read_input(input)
    pipe = Pipeline(client=client)
    art = pipe.run(text, name=name or inferred, cases_per_eval=cases, generate_code=not no_code)
    out_dir = pipe.write_artifacts(art, out / art.spec.name)

    r = art.report
    assert r is not None
    verdict = "PASS" if r.passed else "FAIL"
    color = "green" if r.passed else "red"
    ci = f" (95% CI [{r.ci_low:.2f}, {r.ci_high:.2f}])" if r.ci_low is not None else ""
    console.print(
        f"\n[{color}]{verdict}[/{color}] overall={r.overall_score:.2f}{ci} "
        f"across {len(r.results)} evals"
    )
    if r.num_pending_review:
        console.print(f"[yellow]⏳ {r.num_pending_review} case(s) awaiting human review[/yellow]")
    if db is not None:
        from .store import EvalStore

        with EvalStore(db) as s:
            rid = s.save_report(r, label=name or inferred)
        console.print(f"Ingested into [bold]{db}[/bold] as run [cyan]{rid}[/cyan]")
    console.print(f"Artifacts → [bold]{out_dir}[/bold]")
    console.print("[dim](dry-run traces are empty — connect your agent via the generated eval_harness.py)[/dim]")


@app.command("scaffold")
def scaffold_cmd(
    input: str = typer.Argument(...),
    name: str | None = typer.Option(None),
    out: Path = typer.Option(Path("generated"), help="Output directory."),
    provider: str | None = typer.Option(None, help="anthropic | openai | gemini."),
    offline: bool = typer.Option(False),
) -> None:
    """Generate an agent skeleton + eval harness."""
    client = _client(offline, provider)
    _mode_banner(client)
    text, inferred = _read_input(input)
    spec = ingest(text, name=name or inferred, client=client)
    files = scaffold(spec, client=client)
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        (out / rel).write_text(content)
    console.print(f"Wrote {len(files)} files → [bold]{out}[/bold]: {', '.join(files)}")


# --------------------------------------------------------------------------- #
# Human-in-the-loop review
# --------------------------------------------------------------------------- #


def _load_run(run_dir: Path) -> tuple[EvalReport, EvalSuite, dict[str, Trace]]:
    report = EvalReport.model_validate_json((run_dir / "report.json").read_text())
    suite = EvalSuite.model_validate_json((run_dir / "suite.json").read_text())
    traces: dict[str, Trace] = {}
    tp = run_dir / "traces.json"
    if tp.exists():
        traces = {k: Trace.model_validate(v) for k, v in json.loads(tp.read_text()).items()}
    return report, suite, traces


@review_app.command("export")
def review_export(
    run_dir: Path = typer.Argument(..., help="A ./runs/<name> directory."),
    out: Path = typer.Option(Path("review_queue.jsonl"), help="Queue file (.jsonl or .csv)."),
    all_cases: bool = typer.Option(False, "--all", help="Queue every case, not just flagged ones."),
) -> None:
    """Export cases needing human review (human graders + uncertain judgments)."""
    from .human import build_review_queue, write_queue

    report, suite, traces = _load_run(run_dir)
    items = build_review_queue(report, suite, traces, all_cases=all_cases)
    write_queue(items, out)
    console.print(f"Wrote [green]{len(items)}[/green] review item(s) → [bold]{out}[/bold]")
    if items:
        console.print("[dim]Fill in human_score / human_passed / notes, then `review import`.[/dim]")


@review_app.command("import")
def review_import(
    queue: Path = typer.Argument(..., help="A reviewed queue file (.jsonl or .csv)."),
    run_dir: Path = typer.Option(..., "--run", help="The ./runs/<name> dir to update."),
    db: Path | None = typer.Option(None, help="Optionally re-ingest the updated run here."),
) -> None:
    """Merge human verdicts into a run, re-aggregate, and report judge agreement."""
    from .human import apply_reviews, judge_human_agreement, read_verdicts
    from .report import render_json, render_markdown

    report, _suite, _traces = _load_run(run_dir)
    verdicts = read_verdicts(queue)
    reviewed = [v for v in verdicts if v.reviewed]
    updated = apply_reviews(report, verdicts)

    (run_dir / "report.json").write_text(render_json(updated))
    (run_dir / "report.md").write_text(render_markdown(updated))

    verdict = "PASS" if updated.passed else "FAIL"
    console.print(
        f"Merged [green]{len(reviewed)}[/green] human verdict(s) → "
        f"overall={updated.overall_score:.2f} ({verdict})"
    )
    agreement = judge_human_agreement(report, verdicts)
    if agreement:
        console.print(f"[bold]Judge ↔ human agreement:[/bold] {agreement.summary()}")
        if agreement.kappa < 0.4:
            console.print("[yellow]⚠ low agreement — do not trust this judge unattended.[/yellow]")
    if db is not None:
        from .store import EvalStore

        with EvalStore(db) as s:
            s.save_report(updated, label=run_dir.name)
        console.print(f"Re-ingested into [bold]{db}[/bold]")


# --------------------------------------------------------------------------- #
# Data store / analytics
# --------------------------------------------------------------------------- #


@db_app.command("ingest")
def db_ingest(
    run_dir: Path = typer.Argument(..., help="A ./runs/<name> directory."),
    db: Path = typer.Option(Path("eval_results.db"), help="SQLite database file."),
    label: str | None = typer.Option(None, help="Human-friendly run label."),
) -> None:
    """Load a run's report.json into the results store."""
    from .store import EvalStore

    report = EvalReport.model_validate_json((run_dir / "report.json").read_text())
    with EvalStore(db) as s:
        rid = s.save_report(report, label=label or run_dir.name)
    console.print(f"Ingested run [cyan]{rid}[/cyan] → [bold]{db}[/bold]")


@db_app.command("runs")
def db_runs(
    db: Path = typer.Option(Path("eval_results.db")),
    spec: str | None = typer.Option(None, help="Filter by spec name."),
    limit: int = typer.Option(20),
) -> None:
    """List recorded runs."""
    from .store import EvalStore

    with EvalStore(db) as s:
        rows = s.runs(spec, limit=limit)
    table = Table("run_id", "spec", "model", "score", "passed", "created_at")
    for r in rows:
        table.add_row(
            r["run_id"], r["spec_name"], str(r["model"]),
            f"{r['overall_score']:.2f}" if r["overall_score"] is not None else "—",
            "✅" if r["passed"] else "❌", r["created_at"][:19],
        )
    console.print(table)


@db_app.command("trend")
def db_trend(
    spec: str = typer.Argument(...),
    db: Path = typer.Option(Path("eval_results.db")),
    limit: int = typer.Option(30),
) -> None:
    """Show the overall-score history for a spec."""
    from .analytics import score_trend
    from .store import EvalStore

    with EvalStore(db) as s:
        points = score_trend(s, spec, limit=limit)
    if not points:
        console.print(f"[yellow]No runs for spec '{spec}'.[/yellow]")
        return
    for p in points:
        bar = "█" * round(p.overall_score * 20)
        mark = "✅" if p.passed else "❌"
        console.print(f"{p.created_at[:19]} {mark} {p.overall_score:.2f} {bar} [dim]{p.label or p.run_id}[/dim]")


@db_app.command("compare")
def db_compare(
    baseline: str = typer.Argument(..., help="Baseline run_id."),
    candidate: str = typer.Argument(..., help="Candidate run_id."),
    db: Path = typer.Option(Path("eval_results.db")),
) -> None:
    """A/B two runs with significance testing (regression gate)."""
    from .analytics import compare_runs
    from .store import EvalStore

    with EvalStore(db) as s:
        rep = compare_runs(s, baseline, candidate)
    sig = "significant" if rep.overall_test.significant else "not significant"
    console.print(
        f"[bold]{rep.spec_name}[/bold]: Δ overall = {rep.overall_delta:+.3f} "
        f"(p={rep.overall_test.p_value:.3f}, {sig})"
    )
    table = Table("dimension", "baseline", "candidate", "Δ", "p-value", "verdict")
    for d in rep.dimensions:
        if d.delta < 0 and d.comparison.significant:
            verdict = "[red]REGRESSED[/red]"
        elif d.delta > 0 and d.comparison.significant:
            verdict = "[green]improved[/green]"
        else:
            verdict = "[dim]≈ noise[/dim]"
        table.add_row(
            d.dimension, f"{d.baseline:.2f}", f"{d.candidate:.2f}",
            f"{d.delta:+.2f}", f"{d.comparison.p_value:.3f}", verdict,
        )
    console.print(table)
    if rep.regressed:
        raise typer.Exit(code=1)  # CI gate: fail on significant regression


@db_app.command("query")
def db_query(
    sql: str = typer.Argument(..., help="Read-only SQL against the store."),
    db: Path = typer.Option(Path("eval_results.db")),
) -> None:
    """Run arbitrary SQL and print rows as JSON."""
    from .store import EvalStore

    with EvalStore(db) as s:
        rows = s.query(sql)
    console.print_json(json.dumps(rows, default=str))


@app.command("serve")
def serve_cmd(
    db: Path = typer.Option(Path("eval_results.db")),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the REST API over the results store (needs the [server] extra)."""
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        console.print("[red]The API needs extras:[/red] pip install 'agentic-ai-eval[server]'")
        raise typer.Exit(code=1) from None
    uvicorn.run(create_app(str(db)), host=host, port=port)


if __name__ == "__main__":
    app()
