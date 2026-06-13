"""agentic-ai-eval — a research-grade, provider-agnostic evaluation pipeline for
agentic AI features.

Give it a description, sketch, or diagram of an agentic AI pipeline and it will:
  1. understand the system (decompose into components + data flow),
  2. analyze failure modes and produce a prioritized risk register,
  3. generate a targeted eval suite (component + end-to-end, with graders),
  4. run the suite against your agent's traces — with a calibrated jury of
     judges, human-in-the-loop review, and bootstrap confidence intervals,
  5. persist every run to a SQL store you can trend, A/B, and serve over REST,
  6. scaffold an agent skeleton + CI eval harness.

Works with Anthropic (Claude), OpenAI (GPT), or Google (Gemini) — auto-detected
from the API key in your environment — and degrades to a fully deterministic
offline mode with no key at all.

Quick start:

    from agentic_ai_eval import Pipeline

    art = Pipeline().run("A customer-support agent that routes, retrieves "
                         "from docs (RAG), and can issue refunds via a tool.")
    print(art.report.overall_score, art.report.ci_low, art.report.ci_high)
"""

from .analyze import analyze
from .evals import generate_suite, materialize_traces, run_suite
from .human import apply_reviews, build_review_queue, judge_human_agreement
from .ingest import ingest
from .llm import LLMClient
from .pipeline import Pipeline, PipelineArtifacts
from .scaffold import scaffold
from .schema import (
    Component,
    ComponentType,
    Eval,
    EvalCase,
    EvalDimension,
    EvalReport,
    EvalSuite,
    Grader,
    GraderKind,
    SystemAnalysis,
    SystemSpec,
    Trace,
)
from .store import EvalStore

__version__ = "0.2.0"

__all__ = [
    "Pipeline",
    "PipelineArtifacts",
    "LLMClient",
    "ingest",
    "analyze",
    "generate_suite",
    "run_suite",
    "materialize_traces",
    "scaffold",
    "build_review_queue",
    "apply_reviews",
    "judge_human_agreement",
    "EvalStore",
    "SystemSpec",
    "SystemAnalysis",
    "Component",
    "ComponentType",
    "Eval",
    "EvalCase",
    "EvalSuite",
    "EvalDimension",
    "EvalReport",
    "Grader",
    "GraderKind",
    "Trace",
]
