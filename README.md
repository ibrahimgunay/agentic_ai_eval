<h1 align="center">agentic-ai-eval</h1>

<p align="center">
  <strong>A research-grade, provider-agnostic evaluation pipeline for agentic AI features.</strong><br>
  Describe an agent — in prose, a sketch, or a Mermaid diagram — and it derives the
  system, finds the failure modes, generates a targeted eval suite, grades it with a
  <em>calibrated jury of judges and humans</em>, and trends every run over SQL.
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="providers" src="https://img.shields.io/badge/LLM-Anthropic%20%7C%20OpenAI%20%7C%20Gemini-7e56c2">
  <img alt="offline" src="https://img.shields.io/badge/offline-deterministic-success">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-lightgrey">
</p>

---

> Agentic features are **systems**, not single model calls. A real feature is a
> graph of routers, planners, tool-use, retrieval, memory, sub-agents, and
> guardrails. Evaluating it means evaluating each component **and the seams
> between them** — with enough statistical rigor to make a ship/no-ship call you
> can defend. This pipeline operationalizes exactly that workflow.

## Why it's different

Most "eval scripts" give you one number from one judge on a handful of cases.
This gives you the workflow a frontier-lab eval team actually runs:

| | Typical eval script | **agentic-ai-eval** |
|---|---|---|
| **Model** | Hardcoded to one vendor | **Any of Anthropic / OpenAI / Gemini**, auto-detected; fully offline with no key |
| **Coverage** | A few prompts | Component-level + end-to-end evals derived from a **failure-mode knowledge base** |
| **Judge** | One LLM call | **Jury of judges** (self-consistency + cross-model) with a measured **uncertainty** per verdict |
| **Humans** | None | **Human-in-the-loop** review queue + judge↔human **agreement (κ)** and **calibration** |
| **Rigor** | A point estimate | **Bootstrap confidence intervals**, **two-proportion significance tests** for regressions |
| **Data** | A printout | Normalized **SQLite store**, trend/A-B **analytics**, and an optional **REST API** |
| **Output** | — | A runnable **agent scaffold + CI harness** generated from the spec |

## The pipeline

```
  description / sketch / mermaid diagram
        │
        ▼  ingest        →  SystemSpec        (typed component graph + data flow)
        ▼  analyze       →  SystemAnalysis    (failure modes + prioritized risk register)
        ▼  generate      →  EvalSuite         (component + end-to-end evals, with graders)
        ▼  run           →  EvalReport        (jury + human grading, CIs, scorecard)
        ▼  persist       →  SQLite store      (trends, A/B regression gates, REST API)
        ▼  scaffold      →  agent.py + eval_harness.py + README
```

1. **Understand the system.** Parse prose or a Mermaid flowchart into a typed
   `SystemSpec`: components (`router`, `planner`, `tool_use`, `retrieval`,
   `memory`, `subagent`, `guardrail`, `generation`, `output_formatter`,
   `human_in_loop`), data flow, constraints, and assumptions.
2. **Analyze failure modes.** A curated knowledge base of per-component failure
   modes (misrouting, RAG hallucination, unsafe side-effecting tool calls,
   cross-session memory leakage, over-/under-refusal, …) yields a prioritized
   risk register with mitigations and the eval dimensions that matter.
3. **Generate evals.** Targeted evals per component and end-to-end, each with
   appropriate graders: jury LLM-as-judge with rubrics, tool-trajectory
   matching, JSON-schema validation, exact/contains/regex, numeric thresholds
   (latency / cost / steps), and **human** graders.
4. **Run & grade.** Execute the suite against your agent's **traces**. Each
   judged verdict carries a confidence interval; uncertain or borderline cases
   are routed to humans automatically.
5. **Persist & analyze.** Every run lands in a SQL store you can trend over
   time, A/B with significance testing (a CI regression gate), and serve via REST.
6. **Scaffold code.** Emit a runnable agent skeleton wired to a CI eval harness —
   go from a diagram to failing tests you can make pass.

## Provider-agnostic by design

Set **one** key and the same pipeline runs unchanged:

```bash
export ANTHROPIC_API_KEY=...     # → claude-opus-4-8
# or
export OPENAI_API_KEY=...        # → gpt-4.1
# or
export GOOGLE_API_KEY=...        # → gemini-2.5-pro
```

The provider is auto-detected (or pin it with `--provider` / `AGENTIC_EVAL_PROVIDER`).
With **no key at all**, every LLM stage degrades to a **deterministic offline
mode** — heuristic parsing, knowledge-base analysis, template cases, neutral
labelled judge — so the whole pipeline runs in CI and tests without spend.
The core package depends on **no vendor SDK**; install only the provider you use.

## Install

```bash
git clone https://github.com/ibrahimgunay/agentic_ai_eval.git
cd agentic_ai_eval

pip install -e .                 # offline-capable core, zero vendor SDKs
pip install -e ".[openai]"       # add a provider: openai | anthropic | gemini
pip install -e ".[all,dev]"      # everything + REST API + test tooling

cp .env.example .env             # add a provider key (optional)
```

## Quick start

### Python

```python
from agentic_ai_eval import Pipeline

art = Pipeline().run(
    "A customer-support agent that routes intents, retrieves from docs (RAG), "
    "can issue refunds via a tool, and applies a safety guardrail.",
    name="support-agent",
)
r = art.report
print(f"{r.overall_score:.2f}  95% CI [{r.ci_low:.2f}, {r.ci_high:.2f}]  via {r.provider}")
print(r.by_dimension())
```

### CLI

```bash
# Inspect the system + risk register
agentic-eval analyze examples/customer_support_agent.md

# Generate the eval suite
agentic-eval evals examples/customer_support_agent.md --out suite.json

# Full pipeline (dry run) -> ./runs/<name>/ + ingest into a SQL store
agentic-eval run examples/deep_research_agent.md --db eval.db

# A diagram works as input too
agentic-eval run examples/diagram.mmd

# Generate agent + harness code
agentic-eval scaffold examples/coding_agent.md --out generated/
```

Add `--offline` to any command to force deterministic mode, or
`--provider openai|anthropic|gemini` to pin one.

## Connecting your real agent

The eval runner takes a **trace provider** — a `case_id -> Trace` mapping or
callable. That's the single seam where you plug in your agent:

```python
from agentic_ai_eval.evals import run_suite
from agentic_ai_eval.schema import Trace

def my_traces(case_id: str) -> Trace:
    out = my_agent.run(case_input_for(case_id))   # your code
    return Trace(case_id=case_id, output=out.text, tool_calls=out.tools,
                 steps=out.steps, latency_ms=out.latency,
                 input_tokens=out.in_tok, output_tokens=out.out_tok)

report = run_suite(suite, my_traces)
```

The generated `eval_harness.py` shows the same wiring against the scaffolded
`agent.py`.

## Research-grade grading

A single judge call is a biased point estimate. The judge here is hardened the
way frontier eval teams do it (see `evals/judge.py`):

```python
from agentic_ai_eval.schema import Grader, GraderKind

# A 5-vote self-consistency jury, ensembled across two model families.
Grader(
    kind=GraderKind.LLM_JUDGE,
    rubric="Is every factual claim supported by the provided context?",
    jury_size=5,
    jury_models=["claude-opus-4-8", "gpt-4.1"],
)
```

* **Self-consistency** — multiple sampled verdicts aggregated by median; their
  spread is a free **uncertainty** estimate surfaced on every `GraderResult`.
* **Cross-model jury** — ensemble across models so no single model's
  idiosyncrasies dominate.
* **Calibration to humans** — fit a monotone judge→human map on a gold set, with
  Cohen's **κ** and correlation, so you can *quantify* how far to trust the judge.

Every aggregate score ships with a **bootstrap 95% CI**, and run-to-run
comparisons use a **two-proportion z-test** so regression gates fire on signal,
not noise.

## Human-in-the-loop

Put a person in the loop exactly where it pays off — and measure whether the
automated judge has earned the right to stand in for them:

```bash
# 1. Export the cases that need a human (human graders + uncertain verdicts)
agentic-eval review export runs/deep_research_agent --out queue.csv

# 2. A reviewer fills in human_score / human_passed / notes (CSV or JSONL)

# 3. Merge verdicts back, re-aggregate, and print judge↔human agreement
agentic-eval review import queue.csv --run runs/deep_research_agent
#   → Judge ↔ human agreement: n=24 · κ=0.71 · r=0.83 · MAE=0.11
```

Cases are auto-routed to humans when a `human` grader is attached, when the jury
**disagreed**, or when a score lands in the **borderline** band.

## Data extraction & analytics

Every run is persisted to a normalized **SQLite** database — queryable by any BI
tool, notebook, or the bundled REST API.

```bash
agentic-eval db ingest runs/support              # load a run
agentic-eval db runs --spec support              # list runs
agentic-eval db trend support                    # overall-score history
agentic-eval db compare <baseline> <candidate>   # significance-tested A/B (exits 1 on regression)
agentic-eval db query "SELECT dimension, AVG(score) FROM eval_results GROUP BY 1"
```

```python
from agentic_ai_eval import EvalStore
from agentic_ai_eval.analytics import compare_runs

with EvalStore("eval.db") as store:
    rep = compare_runs(store, baseline_run, candidate_run)
    print("regressed dimensions:", [d.dimension for d in rep.regressed])
```

Serve it over HTTP for dashboards and CI bots (`pip install '.[server]'`):

```bash
agentic-eval serve --db eval.db
# GET /runs  /runs/{id}/evals  /specs/{spec}/trend  /compare?baseline=..&candidate=..
```

## Example use cases

Specs modeled on real frontier agentic features ship in [`examples/`](examples/):

| Spec | Models the pattern behind |
|---|---|
| [`deep_research_agent.md`](examples/deep_research_agent.md) | OpenAI Deep Research · Anthropic Research · Gemini Deep Research |
| [`coding_agent.md`](examples/coding_agent.md) | Claude Code · OpenAI Codex · SWE-bench agents |
| [`computer_use_agent.md`](examples/computer_use_agent.md) | Anthropic Computer Use · OpenAI Operator · Google Mariner |
| [`customer_support_agent.md`](examples/customer_support_agent.md) | RAG + tool-use + guardrail support agent |

## Architecture

| Module | Responsibility |
|---|---|
| `schema.py` | Typed contract for every stage (Pydantic). |
| `providers/` | Vendor-neutral LLM backends (Anthropic / OpenAI / Gemini / offline). |
| `llm.py` | `LLMClient` facade: structured outputs + offline fallback. |
| `ingest.py` | Description / Mermaid / JSON → `SystemSpec`. |
| `analyze.py` | Failure-mode KB + risk register → `SystemAnalysis`. |
| `evals/generate.py` | `SystemSpec` + analysis → `EvalSuite`. |
| `evals/judge.py` | Calibrated jury of judges + judge→human calibration. |
| `evals/graders.py` | Grader implementations (jury judge, human, deterministic). |
| `evals/metrics.py` | Trajectory / latency / cost / step metrics. |
| `evals/runner.py` | Execute suite against traces → `EvalReport` (+ CIs). |
| `stats.py` | Bootstrap CIs, Wilson intervals, z-tests, Cohen's κ. |
| `human.py` | Review-queue export/import + judge↔human agreement. |
| `store.py` | Normalized SQLite results store. |
| `analytics.py` | Trends + significance-tested run comparison. |
| `server.py` | Optional FastAPI REST surface over the store. |
| `scaffold.py` | Generate agent skeleton + CI harness. |
| `report.py` | Markdown / JSON reporting. |
| `pipeline.py` | Orchestration + artifact persistence. |
| `cli.py` | `agentic-eval` command-line interface. |

## Tests

```bash
pytest        # fully offline, deterministic — no keys, no network, no spend
```

## License

Apache-2.0
