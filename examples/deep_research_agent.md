# Deep Research Agent

A long-horizon research agent (in the spirit of OpenAI Deep Research, Anthropic's
Research feature, and Google's Gemini Deep Research) that turns an open-ended
question into a cited, multi-section report.

The agent first **plans** the investigation: it decomposes the question into
sub-questions and an ordered research strategy. A **router** decides, per
sub-question, whether to use **web search**, an internal **retrieval** index, or
existing memory. For breadth it spawns parallel **sub-agents**, one per
sub-question, each running its own search → read → extract loop with **tools**
(`web_search`, `open_url`, `code_interpreter` for quick analysis). It maintains
long-horizon **memory** of findings and visited sources across many steps so it
does not loop or re-fetch. A **guardrail** enforces sourcing policy: no claim
ships without an attributable citation, and the agent must refuse to fabricate
sources. Finally a **generation** step synthesizes the findings and an
**output formatter** emits a structured report with inline citations.

Constraints:
- Every factual claim in the final report must cite a retrieved source (no
  unsupported assertions).
- The agent must terminate within a 40-step / $2.00 budget per query.
- Parallel sub-agents must not duplicate each other's work or exceed the global
  step budget in aggregate.
- The agent must never present a hallucinated or unreachable URL as a citation.
