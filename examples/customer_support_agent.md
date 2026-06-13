# Customer Support Agent

A customer-support agent for an e-commerce company.

The agent receives a user message and first **routes** it by intent
(order-status, refund, general question). It maintains short-term **memory** of
the conversation so follow-ups make sense. For knowledge questions it
**retrieves** relevant passages from the help-center docs (RAG) and must ground
its answer in them. It can call **tools**: `lookup_order`, `issue_refund`
(side-effecting — moves money), and `escalate_to_human`. A **guardrail** layer
enforces policy: refunds above $200 must be escalated, not auto-issued, and the
agent must never reveal another customer's data. Finally it **formats** the
response as JSON `{ "answer": str, "action_taken": str }`.

Constraints:
- p95 end-to-end latency must be under 6 seconds.
- The agent must never issue a refund over $200 without human approval.
- Every factual claim about policy must be grounded in retrieved docs.
