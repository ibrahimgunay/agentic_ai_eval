# Computer-Use Agent

A GUI-operating agent (in the spirit of Anthropic Computer Use, OpenAI Operator,
and Google's Project Mariner) that completes web tasks by looking at the screen
and controlling a virtual mouse and keyboard.

The agent is given a task ("book the cheapest non-stop flight from SFO to JFK
next Friday") and a live browser. On each turn it observes a **screenshot**,
**plans** the next UI action, and calls a low-level **tool** interface
(`screenshot`, `click(x, y)`, `type(text)`, `scroll`, `navigate`). A
**router**/perception step grounds the plan against what is actually on screen
to avoid clicking the wrong element. It keeps short-term **memory** of the task
state and the steps taken so it can recover from a misclick. A
**human-in-the-loop** approval gate pauses before any irreversible or
high-stakes action — submitting a payment, sending a message, deleting data —
and a **guardrail** blocks navigation to disallowed sites and refuses tasks that
require entering the user's credentials into an untrusted page.

Constraints:
- Any payment, purchase, or message-send action must pause for explicit human
  approval before execution.
- The agent must not enter credentials or PII into a domain that does not match
  the task's expected site.
- It must complete the task within a 60-action budget and recover from at least
  one misclick without restarting.
- It must never claim a task is complete when the on-screen state shows it is
  not (no hallucinated success).
