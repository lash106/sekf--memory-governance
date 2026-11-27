# ADR 0008: Optional policy-v2 and customer brief

- Status: proposed; on hold until advisor approves the customer-service direction
- Date: 2026-11-18
- Owners affected: Shailen (policy, API), Sai Chetan (originating branch)

## Context

Sprint 3 keeps the research version general (see ADR 0007 for projections and offline metrics).
A customer-service policy with risk tiers, compliance recording, and a 30-day episodic brief
TTL is useful only if the advisor approves that product direction.

## Decision

1. `PolicyEngine` / `policy-v1` remains the default. `SEKF_POLICY=v2` selects `PolicyEngineV2`.
2. Risk tier, channel prior, required-evidence text, `needs_verification`, and `R_COMPLIANCE`
   are written into `DecisionRecord.explanation` and `rule_ids_fired`. Contract objects are
   not extended.
3. `R_COMPLIANCE` does not change `final_action`.
4. Claims are never deleted. A 30-day episodic TTL only drops old `CANDIDATE` rows from
   `GET /v1/customers/{subject_key}/brief`.

## Consequences

- Held behind advisor approval. Not part of the Sprint-3 merge of ADR 0007.
- Markdown projection and offline metrics stay in ADR 0007 / PR #25.
