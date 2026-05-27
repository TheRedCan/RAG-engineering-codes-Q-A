"""Evaluation harness: hand-curated Q&A test cases plus deterministic
structural metrics over the canonical ``Answer`` schema.

We deliberately avoid LLM-as-judge metrics (faithfulness scored by a
second LLM call). They roughly double per-query cost and introduce
noise from the judge model's own quirks. Structural checks against
hand-written expectations catch the real regressions:

- Language: answer in the right script.
- Refusal: claims empty exactly when we expect a refusal.
- Citations: required source documents appear in the citations.
- Content: required keywords appear in claim text; forbidden ones don't.
- Latency: per-query wall-clock tracked for trend, not pass/fail.

Entry point: ``python -m eval.run``.
"""
