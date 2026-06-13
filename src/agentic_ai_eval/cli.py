"""Command-line interface for agentic-ai-eval.

    agentic-eval analyze   <input>   # ingest + analyze, print risk register
    agentic-eval evals     <input>   # generate the eval suite, write suite.json
    agentic-eval run       <input>   # full pipeline (dry run) -> ./runs/<name>
    agentic-eval scaffold  <input>   # generate agent + harness code

`<input>` is a file path (.md/.txt/.mmd/.json) or an inline description string.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .analyze import analyze
from .evals import generate_suite
from .ingest import ingest
from .llm import LLMClient
from .pipeline import Pipeline
from .report import render_analysis_markdown
from .scaffold import scaffold

app = typer.Typer(add_completion=False, help="Evaluation pipeline for agentic AI features.")
console = Console()


def _read_input(value: str) -> tuple[str, str | None]:
    """Return (text, name). If `value` is a readable file, load it; else treat
    it as an inline description."""
    p = Path(value)
    if p.exists() and p.is_file():
        return p.read_text(), p.stem
    return value, None


def _client(online: bool) -> LLMClient:
    if not online:
        # Force offline by passing an empty key.
        return LLMClient(api_key="")
    return LLMClient()


def _mode_banner(client: LLMClient) -> None:
    mode = "ONLINE (Claude)" if client.online else "OFFLINE (deterministic)"
    console.print(f"[dim]mode: {mode}[/dim]")


@app.command("analyze")
def analyze_cmd(
    input: str = typer.Argument(..., help="File path or inline description."),
    name: str | None = typer.Option(None, help="Override the system name."),
    offline: bool = typer.Option(False, help="Force offline mode (no API calls)."),
) -> None:
    """Ingest + analyze: print components and the risk register."""
    client = _client(online=not offline)
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
    offline: bool = typer.Option(False),
) -> None:
    """Generate the eval suite and write it to disk."""
    client = _client(online=not offline)
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
    offline: bool = typer.Option(False),
) -> None:
    """Run the full pipeline (dry-run traces) and write all artifacts."""
    client = _client(online=not offline)
    _mode_banner(client)
    text, inferred = _read_input(input)
    pipe = Pipeline(client=client)
    art = pipe.run(text, name=name or inferred, cases_per_eval=cases, generate_code=not no_code)
    out_dir = pipe.write_artifacts(art, out / art.spec.name)

    r = art.report
    assert r is not None
    verdict = "PASS" if r.passed else "FAIL"
    color = "green" if r.passed else "red"
    console.print(
        f"\n[{color}]{verdict}[/{color}] overall={r.overall_score:.2f} "
        f"across {len(r.results)} evals"
    )
    console.print(f"Artifacts → [bold]{out_dir}[/bold]")
    console.print("[dim](dry-run traces are empty — connect your agent via the generated eval_harness.py)[/dim]")


@app.command("scaffold")
def scaffold_cmd(
    input: str = typer.Argument(...),
    name: str | None = typer.Option(None),
    out: Path = typer.Option(Path("generated"), help="Output directory."),
    offline: bool = typer.Option(False),
) -> None:
    """Generate an agent skeleton + eval harness."""
    client = _client(online=not offline)
    _mode_banner(client)
    text, inferred = _read_input(input)
    spec = ingest(text, name=name or inferred, client=client)
    files = scaffold(spec, client=client)
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        (out / rel).write_text(content)
    console.print(f"Wrote {len(files)} files → [bold]{out}[/bold]: {', '.join(files)}")


if __name__ == "__main__":
    app()
