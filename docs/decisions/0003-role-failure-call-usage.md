# ADR 0003: Keep call usage on failed roles

- Status: accepted
- Date: 2026-10-27
- Owner: Sai Chetan Muppalla (providers / Judge); Shailen Sutradhar consumes them in policy and orchestration
- Contract version: v1, additive. Existing constructors and JSON without the new fields still validate.

## Context

`OpenAICompatibleProvider` summed tokens across repair attempts, then discarded that sum when it raised `RoleOutputInvalid`. Governance recorded a role failure and a cost of zero. Failed three-call attempts looked free, which biases RQ4.

## Decision

1. Typed errors may carry a `CallUsage` (calls, tokens, latency, estimated cost).
2. `RoleFailure` stores the same numbers, defaulting to zero when a failure never reached a model.
3. `PolicyEngine` adds those numbers to `DecisionRecord.cost` along with successful role metadata.
4. Model prices come from a versioned table (`COST_RATE_VERSION`). Unknown models are not billed at the `gpt-4o-mini` rate.

## Consequences

- A repair that ends in R1 still shows its LLM calls in the decision cost.
- Regenerated `contracts/v1/RoleFailure.json` includes the new optional fields.
- Experiments that compare cost must use the same `COST_RATE_VERSION` or explicit `SEKF_LLM_*_COST_PER_MTOK` overrides.

## Acknowledgment

Accepted 2026-10-27. The fields are additive (default zero) and already consumed by policy and the API on main. No further change to these fields is proposed. Shailen (policy and API), Krishna (persisted audit), and Mahalakshmi (role-failure traces) are the affected owners; the shape is the one already merged, and a follow-up only happens if one of them objects to that shape.
