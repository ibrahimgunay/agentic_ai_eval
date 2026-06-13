# agentic-ai-eval

**A world-class evaluation pipeline for agentic AI features.** Give it a
description, sketch, or diagram of an agentic AI pipeline, and it will
understand the system, generate a targeted eval suite, run it, and scaffold the
code — the workflow a research/eval team at a frontier lab would stand up
before shipping an agent.

> Agentic features are *systems*, not single model calls. A real feature is a
> graph of routers, planners, tool-use, retrieval, memory, sub-agents, and
> guardrails. Evaluating it means evaluating each component **and the seams
> between them**. This pipeline operationalizes that.

---

## What it does

```
  description / sketch / mermaid diagram
        │
        ▼  ingest        →  SystemSpec        (typed component graph + data flow)
        ▼  analyze       →  SystemAnalysis    (failure modes + prioritized risk register)
        ▼  generate      →  EvalSuite         (component + end-to-end evals, with graders)
        ▼  run           →  EvalReport        (scorecard by dimension, failing-case detail)
        ▼  scaffold      →  agent.py + eval_harness.py + README
```

1. **Understand the system.** Parse prose or a Mermaid flowchart into a typed
   `SystemSpec`: components (`router`, `planner`, `tool_use`, `retrieval`,
   `memory`, `subagent`, `guardrail`, `generation`, `output_formatter`,
   `human_in_loop`), their data flow, constraints, and assumptions.
2. **Analyze failure modes.** A curated knowledge base of per-component failure
   modes (misrouting, RAG hallucination, unsafe side-effecting tool calls,
   cross-session memory leakage, over-/under-refusal, …) produces a prioritized
   risk register with mitigations and the eval dimensions that matter.
3. **Generate evals.** Targeted evals per component and end-to-end, each with
   appropriate graders: LLM-as-judge with rubrics, tool-trajectory matching,
   JSON-schema validation, exact/contains/regex, and numeric thresholds
   (latency / cost / steps).
4. **Run.** Execute the suite against your agent's **traces** (or a dry run),
   aggregate into a scorecard with per-dimension scores and failing-case detail.
5. **Scaffold code.** Emit a runnable agent skeleton wired to a CI eval harness
   — go from a diagram to failing tests you can make pass.

## Online + offline

Backed by **Claude (`claude-opus-4-8`)** with adaptive thinking and structured
outputs for ingestion, analysis, case generation, and judging. With no
`ANTHROPIC_API_KEY` set, every LLM stage degrades to a **deterministic offline
mode** (heuristic parsing, knowledge-base analysis, template cases, neutral
labeled judge) so the whole pipeline runs in CI and tests without spend.

## Install

```bash
git clone https://github.com/ibrahimgunay/agentic_ai_eval.git
cd agentic_ai_eval
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY (optional)
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
print(art.report.overall_score)
print(art.report.by_dimension())
```

### CLI

```bash
# Inspect the system + risk register
agentic-eval analyze examples/customer_support_agent.md

# Generate the eval suite
agentic-eval evals examples/customer_support_agent.md --out suite.json

# Full pipeline (dry run) -> ./runs/<name>/{spec,analysis,suite,report.md,generated/}
agentic-eval run examples/customer_support_agent.md

# Diagram input works too
agentic-eval run examples/diagram.mmd

# Generate agent + harness code
agentic-eval scaffold examples/customer_support_agent.md --out generated/
```

Add `--offline` to any command to force deterministic mode.

## Connecting your real agent

The eval runner takes a **trace provider** — a `case_id -> Trace` mapping or
callable. That's the single seam where you plug in your agent:

```python
from agentic_ai_eval.evals import run_suite
from agentic_ai_eval.schema import Trace

def my_traces(case_id: str) -> Trace:
    out = my_agent.run(case_input_for(case_id))   # your code
    return Trace(case_id=case_id, output=out.text, tool_calls=out.tools,
                 steps=out.steps, latency_ms=out.latency, input_tokens=out.in_tok,
                 output_tokens=out.out_tok)

report = run_suite(suite, my_traces)
```

The generated `eval_harness.py` shows the same wiring against the scaffolded
`agent.py`.

## Architecture

| Module | Responsibility |
|---|---|
| `schema.py` | Typed contract for every stage (Pydantic). |
| `llm.py` | Anthropic client wrapper; structured outputs + offline fallback. |
| `ingest.py` | Description / Mermaid / JSON → `SystemSpec`. |
| `analyze.py` | Failure-mode KB + risk register → `SystemAnalysis`. |
| `evals/generate.py` | `SystemSpec` + analysis → `EvalSuite`. |
| `evals/graders.py` | Grader implementations (judge + deterministic). |
| `evals/metrics.py` | Trajectory/latency/cost/step metrics. |
| `evals/runner.py` | Execute suite against traces → `EvalReport`. |
| `scaffold.py` | Generate agent skeleton + CI harness. |
| `report.py` | Markdown / JSON reporting. |
| `pipeline.py` | Orchestration + artifact persistence. |
| `cli.py` | `agentic-eval` command-line interface. |

## Tests

```bash
pytest        # fully offline, deterministic
```

## License

Apache-2.0
