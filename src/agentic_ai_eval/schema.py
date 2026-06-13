"""Typed representation of an agentic AI system and its evaluation artifacts.

These Pydantic models are the contract that flows through the whole pipeline:

    raw description / diagram
        -> SystemSpec            (ingest)
        -> SystemAnalysis        (analyze: components, failure modes, risk)
        -> EvalPlan / EvalSuite  (generate evals)
        -> EvalReport            (run + aggregate)

Keeping every stage strongly typed means the LLM-backed stages can use
structured outputs, the offline stages stay testable, and a generated agent
or CI harness can be derived deterministically from the same objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComponentType(str, Enum):
    """The canonical building blocks of an agentic feature.

    A real agentic pipeline is rarely one model call — it is a graph of these.
    Evaluating the feature means evaluating each of these *and* their seams.
    """

    PLANNER = "planner"            # decomposes the task / decides next step
    ROUTER = "router"             # classifies intent, dispatches to a path
    TOOL_USE = "tool_use"          # function/tool calling
    RETRIEVAL = "retrieval"        # RAG / search / knowledge lookup
    MEMORY = "memory"             # short- or long-term state across turns
    SUBAGENT = "subagent"         # delegated agent in a multi-agent setup
    GUARDRAIL = "guardrail"        # safety / policy / validation layer
    GENERATION = "generation"      # the model call that produces user-facing text
    OUTPUT_FORMATTER = "output_formatter"  # structured-output / post-processing
    HUMAN_IN_LOOP = "human_in_loop"  # approval / confirmation gate
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvalDimension(str, Enum):
    """What an individual evaluation is actually measuring."""

    CORRECTNESS = "correctness"        # did it produce the right answer/action
    FAITHFULNESS = "faithfulness"      # grounded in retrieved/provided context
    TOOL_SELECTION = "tool_selection"  # right tool, right args, right order
    PLANNING = "planning"              # coherent, efficient decomposition
    ROBUSTNESS = "robustness"          # handles noisy/adversarial/edge inputs
    SAFETY = "safety"                  # refusals, policy adherence, harm avoidance
    LATENCY = "latency"                # steps / wall-clock / token budget
    COST = "cost"                      # token + tool-call economics
    FORMAT = "format"                  # schema / structured-output validity
    INSTRUCTION_FOLLOWING = "instruction_following"


class GraderKind(str, Enum):
    LLM_JUDGE = "llm_judge"        # rubric-scored by a model (optionally a jury)
    HUMAN = "human"               # scored by a human reviewer (HITL queue)
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    TOOL_TRAJECTORY = "tool_trajectory"  # compares expected vs actual tool calls
    NUMERIC_THRESHOLD = "numeric_threshold"


# --------------------------------------------------------------------------- #
# System specification (output of ingest)
# --------------------------------------------------------------------------- #


class Tool(BaseModel):
    name: str
    description: str = ""
    side_effects: bool = Field(
        default=False,
        description="True if calling the tool mutates external state (send, write, pay).",
    )
    requires_auth: bool = False


class Component(BaseModel):
    """A single node in the agentic pipeline graph."""

    id: str = Field(description="Stable slug, e.g. 'planner', 'rag_retriever'.")
    type: ComponentType
    name: str
    description: str = ""
    model: str | None = Field(default=None, description="Model id if this node calls an LLM.")
    tools: list[Tool] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list, description="Component ids feeding this node.")
    outputs: list[str] = Field(default_factory=list, description="Component ids this node feeds.")


class SystemSpec(BaseModel):
    """A complete, structured picture of the agentic feature under evaluation."""

    name: str
    summary: str = ""
    goal: str = Field(default="", description="What the feature is supposed to accomplish.")
    components: list[Component] = Field(default_factory=list)
    entrypoint: str | None = Field(default=None, description="Component id that receives user input.")
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard requirements: latency budgets, policies, SLAs, formats.",
    )
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    def component(self, component_id: str) -> Component | None:
        return next((c for c in self.components if c.id == component_id), None)

    def components_of(self, ctype: ComponentType) -> list[Component]:
        return [c for c in self.components if c.type == ctype]


# --------------------------------------------------------------------------- #
# Analysis (output of analyze)
# --------------------------------------------------------------------------- #


class FailureMode(BaseModel):
    component_id: str
    title: str
    description: str
    severity: Severity = Severity.MEDIUM
    detection: str = Field(default="", description="How an eval would catch this.")


class RiskItem(BaseModel):
    title: str
    severity: Severity
    rationale: str
    mitigation: str = ""
    dimensions: list[EvalDimension] = Field(default_factory=list)


class SystemAnalysis(BaseModel):
    spec_name: str
    failure_modes: list[FailureMode] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    recommended_dimensions: list[EvalDimension] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- #
# Eval definitions (output of generate)
# --------------------------------------------------------------------------- #


class Grader(BaseModel):
    kind: GraderKind
    # Free-form config interpreted by the grader implementation in evals/graders.py
    rubric: str = ""                      # LLM_JUDGE / HUMAN
    expected: str | None = None           # EXACT_MATCH / CONTAINS / REGEX
    pattern: str | None = None            # REGEX
    json_schema: dict | None = None       # JSON_SCHEMA
    expected_tools: list[str] | None = None  # TOOL_TRAJECTORY (ordered tool names)
    metric: str | None = None             # NUMERIC_THRESHOLD: which metric to read
    threshold: float | None = None
    direction: str = "gte"                # "gte" | "lte" for NUMERIC_THRESHOLD
    weight: float = 1.0
    # LLM_JUDGE jury controls. jury_size>1 samples multiple independent verdicts
    # (self-consistency); jury_models ensembles across models to cut single-model
    # bias. Both surface an uncertainty estimate on the GraderResult.
    jury_size: int = 1
    jury_models: list[str] | None = None


class EvalCase(BaseModel):
    """One concrete test: an input, optional gold reference, and graders."""

    id: str
    input: str
    context: str = Field(default="", description="Docs/state provided to the system, for grounding.")
    reference: str | None = Field(default=None, description="Gold answer, if known.")
    expected_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    graders: list[Grader] = Field(default_factory=list)


class Eval(BaseModel):
    """A named evaluation targeting one component (or end-to-end) and one dimension."""

    id: str
    target_component: str = Field(description="Component id, or 'system' for end-to-end.")
    dimension: EvalDimension
    description: str = ""
    graders: list[Grader] = Field(
        default_factory=list,
        description="Default graders applied to every case that doesn't define its own.",
    )
    cases: list[EvalCase] = Field(default_factory=list)
    pass_threshold: float = Field(default=0.7, description="Aggregate score required to pass.")

    def graders_for(self, case: EvalCase) -> list[Grader]:
        """Case-level graders win; otherwise fall back to the eval's defaults."""
        return case.graders if case.graders else self.graders


class EvalSuite(BaseModel):
    spec_name: str
    evals: list[Eval] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Run artifacts (output of the runner)
# --------------------------------------------------------------------------- #


class Trace(BaseModel):
    """What the system-under-test produced for a single case.

    Supplied by the caller (their agent), or synthesized in dry-run mode.
    """

    case_id: str
    output: str = ""
    tool_calls: list[str] = Field(default_factory=list, description="Ordered tool names invoked.")
    steps: int = 0
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class GraderResult(BaseModel):
    kind: GraderKind
    score: float = Field(ge=0.0, le=1.0)
    weight: float = 1.0
    passed: bool
    rationale: str = ""
    source: str = Field(default="auto", description="auto | offline_stub | human | jury.")
    uncertainty: float | None = Field(
        default=None, description="Std-dev across jurors/reviewers, if measured.")
    pending: bool = Field(
        default=False, description="True if awaiting a human verdict (HITL).")


class CaseResult(BaseModel):
    case_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    grader_results: list[GraderResult] = Field(default_factory=list)


class EvalResult(BaseModel):
    eval_id: str
    target_component: str
    dimension: EvalDimension
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    pass_threshold: float
    case_results: list[CaseResult] = Field(default_factory=list)
    # 95% bootstrap CI on the mean score, populated by the runner.
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def num_cases(self) -> int:
        return len(self.case_results)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.case_results if c.passed)

    @property
    def has_pending_review(self) -> bool:
        return any(
            g.pending for c in self.case_results for g in c.grader_results
        )


class EvalReport(BaseModel):
    spec_name: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    results: list[EvalResult] = Field(default_factory=list)
    analysis: SystemAnalysis | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    # Reproducibility metadata: who graded this, and how.
    provider: str | None = None
    model: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    def by_dimension(self) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        for r in self.results:
            agg.setdefault(r.dimension.value, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in agg.items()}

    @property
    def num_pending_review(self) -> int:
        return sum(
            1
            for r in self.results
            for c in r.case_results
            for g in c.grader_results
            if g.pending
        )
