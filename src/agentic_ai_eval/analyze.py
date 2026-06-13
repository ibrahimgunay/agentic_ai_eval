"""Understand the system: derive failure modes, a risk register, and the eval
dimensions that matter for this particular pipeline.

This is the "what could go wrong, and how would we catch it" stage. It is
backed by a curated knowledge base of failure modes per component type (the
kind of institutional knowledge an eval team accumulates), and — when online —
augmented by the model reasoning over the specific spec.
"""

from __future__ import annotations

from .llm import LLMClient
from .schema import (
    ComponentType,
    EvalDimension,
    FailureMode,
    RiskItem,
    Severity,
    SystemAnalysis,
    SystemSpec,
)

# Per-component-type failure-mode knowledge base.
# Each entry: (title, description template, severity, detection, dimensions)
_FAILURE_KB: dict[ComponentType, list[tuple[str, str, Severity, str, list[EvalDimension]]]] = {
    ComponentType.ROUTER: [
        (
            "Misrouting",
            "Routes the request to the wrong path/intent, derailing the whole trajectory.",
            Severity.HIGH,
            "Intent-classification eval with a labeled set, including near-miss pairs.",
            [EvalDimension.CORRECTNESS, EvalDimension.ROBUSTNESS],
        ),
        (
            "Fallthrough on novel intents",
            "No path matches; silently picks a default instead of escalating.",
            Severity.MEDIUM,
            "Out-of-distribution intents that should trigger a fallback/handoff.",
            [EvalDimension.ROBUSTNESS, EvalDimension.SAFETY],
        ),
    ],
    ComponentType.PLANNER: [
        (
            "Incoherent decomposition",
            "Produces a plan with missing, redundant, or contradictory steps.",
            Severity.HIGH,
            "Plan-quality judge over multi-step tasks; check step count vs. optimal.",
            [EvalDimension.PLANNING, EvalDimension.CORRECTNESS],
        ),
        (
            "Looping / non-termination",
            "Repeats steps without progress and burns the step/token budget.",
            Severity.HIGH,
            "Cap steps; assert termination and a step-efficiency threshold.",
            [EvalDimension.LATENCY, EvalDimension.COST],
        ),
    ],
    ComponentType.TOOL_USE: [
        (
            "Wrong tool / wrong arguments",
            "Selects an inappropriate tool or passes malformed arguments.",
            Severity.HIGH,
            "Tool-trajectory eval comparing expected vs. actual ordered calls + args.",
            [EvalDimension.TOOL_SELECTION, EvalDimension.CORRECTNESS],
        ),
        (
            "Unsafe side-effecting calls",
            "Invokes a mutating/destructive tool without need or confirmation.",
            Severity.CRITICAL,
            "Adversarial cases that must NOT trigger side-effecting tools.",
            [EvalDimension.SAFETY, EvalDimension.TOOL_SELECTION],
        ),
        (
            "No error recovery",
            "Fails to handle a tool error and proceeds on a bad result.",
            Severity.MEDIUM,
            "Inject tool errors; assert retry/abort/escalate behavior.",
            [EvalDimension.ROBUSTNESS],
        ),
    ],
    ComponentType.RETRIEVAL: [
        (
            "Low recall / irrelevant context",
            "Retrieves nothing useful, starving generation of grounding.",
            Severity.HIGH,
            "Retrieval relevance eval (recall@k) on a labeled corpus.",
            [EvalDimension.CORRECTNESS, EvalDimension.FAITHFULNESS],
        ),
        (
            "Hallucination despite context",
            "Generates claims not supported by the retrieved passages.",
            Severity.HIGH,
            "Faithfulness judge: every claim must be attributable to context.",
            [EvalDimension.FAITHFULNESS],
        ),
    ],
    ComponentType.MEMORY: [
        (
            "Stale or wrong recall",
            "Surfaces outdated or mismatched state across turns.",
            Severity.MEDIUM,
            "Multi-turn cases that depend on earlier-turn facts.",
            [EvalDimension.CORRECTNESS, EvalDimension.ROBUSTNESS],
        ),
        (
            "Cross-session leakage",
            "Leaks one user's state into another session.",
            Severity.CRITICAL,
            "Isolation cases asserting no cross-tenant memory bleed.",
            [EvalDimension.SAFETY],
        ),
    ],
    ComponentType.SUBAGENT: [
        (
            "Delegation drift",
            "Sub-agent diverges from the delegated objective.",
            Severity.MEDIUM,
            "Per-thread objective-adherence judge.",
            [EvalDimension.INSTRUCTION_FOLLOWING, EvalDimension.PLANNING],
        ),
    ],
    ComponentType.GUARDRAIL: [
        (
            "Over-refusal",
            "Blocks benign requests, hurting helpfulness.",
            Severity.MEDIUM,
            "Benign-but-sensitive cases that must be answered.",
            [EvalDimension.SAFETY, EvalDimension.INSTRUCTION_FOLLOWING],
        ),
        (
            "Under-refusal / jailbreak",
            "Lets disallowed content or actions through.",
            Severity.CRITICAL,
            "Red-team / jailbreak suite that must be refused.",
            [EvalDimension.SAFETY, EvalDimension.ROBUSTNESS],
        ),
    ],
    ComponentType.GENERATION: [
        (
            "Incorrect or unhelpful answer",
            "Final response is wrong, vague, or off-task.",
            Severity.HIGH,
            "Correctness judge against references; helpfulness rubric.",
            [EvalDimension.CORRECTNESS, EvalDimension.INSTRUCTION_FOLLOWING],
        ),
    ],
    ComponentType.OUTPUT_FORMATTER: [
        (
            "Invalid structured output",
            "Output violates the required schema, breaking downstream consumers.",
            Severity.HIGH,
            "JSON-schema validation eval over diverse inputs.",
            [EvalDimension.FORMAT],
        ),
    ],
    ComponentType.HUMAN_IN_LOOP: [
        (
            "Bypassed approval gate",
            "Takes a gated action without the required confirmation.",
            Severity.CRITICAL,
            "Assert side-effecting actions pause for approval.",
            [EvalDimension.SAFETY],
        ),
    ],
}


_ANALYZE_SYSTEM = """You are a VP of Engineering for AI evaluation at a frontier \
lab. Given a SystemSpec for an agentic feature and a draft analysis, produce a \
sharper SystemAnalysis: the most important failure modes (mapped to component \
ids), a prioritized risk register with mitigations, and the evaluation \
dimensions that matter most for THIS system. Be concrete and prioritize by \
blast radius. Preserve valid items from the draft; add what's missing."""


def analyze(spec: SystemSpec, *, client: LLMClient | None = None) -> SystemAnalysis:
    client = client or LLMClient()
    draft = _baseline_analysis(spec)

    if client.online:
        refined = client.parse(
            schema=SystemAnalysis,
            system=_ANALYZE_SYSTEM,
            user=(
                f"SystemSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Draft analysis:\n{draft.model_dump_json(indent=2)}"
            ),
        )
        if refined.failure_modes or refined.risks:
            refined.spec_name = spec.name
            return refined

    return draft


def _baseline_analysis(spec: SystemSpec) -> SystemAnalysis:
    failure_modes: list[FailureMode] = []
    dimensions: set[EvalDimension] = set()

    for comp in spec.components:
        for title, desc, sev, detection, dims in _FAILURE_KB.get(comp.type, []):
            failure_modes.append(
                FailureMode(
                    component_id=comp.id,
                    title=title,
                    description=desc,
                    severity=sev,
                    detection=detection,
                )
            )
            dimensions.update(dims)

        # Side-effecting tools always raise a safety risk regardless of type KB.
        if any(t.side_effects for t in comp.tools):
            dimensions.add(EvalDimension.SAFETY)

    # End-to-end dimensions every agentic feature should carry.
    dimensions.update({EvalDimension.CORRECTNESS, EvalDimension.LATENCY, EvalDimension.COST})

    risks = _risk_register(spec, failure_modes)

    return SystemAnalysis(
        spec_name=spec.name,
        failure_modes=failure_modes,
        risks=risks,
        recommended_dimensions=sorted(dimensions, key=lambda d: d.value),
        notes=(
            f"Analyzed {len(spec.components)} components; "
            f"{sum(1 for f in failure_modes if f.severity in (Severity.HIGH, Severity.CRITICAL))} "
            "high/critical failure modes identified."
        ),
    )


def _risk_register(spec: SystemSpec, failure_modes: list[FailureMode]) -> list[RiskItem]:
    risks: list[RiskItem] = []

    critical = [f for f in failure_modes if f.severity == Severity.CRITICAL]
    if critical:
        risks.append(
            RiskItem(
                title="Critical safety/side-effect exposure",
                severity=Severity.CRITICAL,
                rationale="; ".join(f"{f.component_id}: {f.title}" for f in critical),
                mitigation="Gate side-effecting tools behind confirmation; add a red-team suite.",
                dimensions=[EvalDimension.SAFETY],
            )
        )

    has_retrieval = bool(spec.components_of(ComponentType.RETRIEVAL))
    has_generation = bool(spec.components_of(ComponentType.GENERATION))
    if has_retrieval and has_generation:
        risks.append(
            RiskItem(
                title="Grounding gap (RAG hallucination)",
                severity=Severity.HIGH,
                rationale="Retrieval + generation seam is the top source of confident, wrong answers.",
                mitigation="Faithfulness judge requiring claim-level attribution to retrieved context.",
                dimensions=[EvalDimension.FAITHFULNESS, EvalDimension.CORRECTNESS],
            )
        )

    multi_step = spec.components_of(ComponentType.PLANNER) or spec.components_of(ComponentType.SUBAGENT)
    if multi_step:
        risks.append(
            RiskItem(
                title="Trajectory cost/latency blowup",
                severity=Severity.MEDIUM,
                rationale="Multi-step planning/delegation can loop and exhaust budgets.",
                mitigation="Enforce step caps and a token/step-efficiency threshold in CI.",
                dimensions=[EvalDimension.LATENCY, EvalDimension.COST],
            )
        )

    if not spec.constraints:
        risks.append(
            RiskItem(
                title="Unspecified success criteria",
                severity=Severity.MEDIUM,
                rationale="No explicit constraints (latency/policy/format) were found in the spec.",
                mitigation="Define checkable acceptance criteria before shipping.",
                dimensions=[EvalDimension.CORRECTNESS],
            )
        )

    return risks
