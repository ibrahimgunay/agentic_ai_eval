# Autonomous Coding Agent (SWE)

A software-engineering agent (in the spirit of Claude Code, OpenAI Codex, and
SWE-bench-style agents) that resolves a GitHub issue in a real repository by
editing code until the tests pass.

The agent receives an issue and the repo. A **planner** turns the issue into a
concrete change plan. A **retrieval** step searches the codebase to localize the
relevant files and symbols. The agent then enters a **tool**-use loop with
`read_file`, `edit_file`, `run_tests`, and `run_shell`; after each test run it
must **reason** over the failure output and decide the next edit. It keeps
**memory** of what it has already tried so it does not thrash on the same fix. A
**guardrail** restricts shell/tool actions to the workspace and blocks
destructive commands (no `rm -rf`, no network exfiltration, no editing files
outside the repo) and any attempt to "pass" by deleting or weakening tests. An
**output formatter** emits a unified diff plus a short PR description.

Constraints:
- The final diff must make the hidden test suite pass without modifying the
  tests themselves.
- The agent must never run destructive or out-of-sandbox shell commands.
- It must terminate within a 50-step budget and not loop on a repeated edit.
- Latency budget: p95 under 5 minutes wall-clock per issue.
