"""agentic-ai-eval — a world-class evaluation pipeline for agentic AI features.

Give it a description, sketch, or diagram of an agentic AI pipeline and it will:
  1. understand the system (decompose into components + data flow),
  2. analyze failure modes and produce a prioritized risk register,
  3. generate a targeted eval suite (component + end-to-end, with graders),
  4. run the suite against your agent's traces (or a dry run), and
  5. scaffold an agent skeleton + CI eval harness.

Quick start:

    from agentic_ai_eval import Pipeline

    art = Pipeline().run("A customer-support agent that routes, retrieves "
                         "from docs (RAG), and can issue refunds via a tool.")
    print(art.report.overall_score)
"""

from .analyze import analyze
from .evals import generate_suite, run_suite
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

__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "PipelineArtifacts",
    "LLMClient",
    "ingest",
    "analyze",
    "generate_suite",
    "run_suite",
    "scaffold",
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
