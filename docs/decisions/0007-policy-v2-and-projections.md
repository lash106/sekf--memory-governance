# ADR 0007: policy-v2, Markdown projection, and separate retrieval recall

- Status: proposed for review by Shailen (policy, API, evaluation) and Krishna (memory)
- Date: 2026-11-18
- Owners affected: Shailen, Krishna, Sai Chetan (Judge calibration helper)

## Context

SPEC.md section 9.4 is one global threshold table. A customer-service claim about a preference and a claim about a payment should not share it. The repository also had no offline place to report section 14 metrics with a paired bootstrap interval, and no readable projection of a committed decision.

Pydantic in this repository is already v2 (`model_validate`, `ConfigDict`). There is no v1 model to freeze and no migration in this change.

## Decision

1. `PolicyEngine` / `policy-v1` is unchanged and remains the default. `SEKF_POLICY=v2` selects `PolicyEngineV2`. Unknown values raise `ConfigurationError`.
2. Risk tier, channel prior, required-evidence text, `needs_verification`, and `R_COMPLIANCE` are written into `DecisionRecord.explanation` and `rule_ids_fired`. The contract objects in `src/sekf/contracts/` are not extended.
3. `R_COMPLIANCE` does not change `final_action`. Epistemic truth and the service-owner check stay distinct.
4. Claims are never deleted. A 30-day episodic TTL only drops old `CANDIDATE` rows from `GET /v1/customers/{subject_key}/brief`.
5. `search_related_claims` stays an exact subject+predicate match. Hybrid BM25 over the same subject is a separate function used for Retrieval Recall@K.
6. `markdown_root` / `SEKF_MARKDOWN_DIR` appends L0–L3 Markdown after a successful commit. The SQLite row is the source of truth. L0/L1/L2 are written first (each line carries `decision:{id}`); the L3 audit entry is the completion marker and is written last. A projection failure after a SQL commit raises `ProjectionFailed` (HTTP 503, retryable); retrying the same decision finishes any missing Markdown layers. The in-memory double rolls the bundle back and raises `CommitRejected`.
7. `POST /v1/experiments/run` scores `datasets/dev` with deterministic roles and the fake Judge. `single_judge` is not run without a real provider. The endpoint does not substitute another model.

## Consequences

- Section 14 numbers from this runner are an offline dev score, not a held-out test result.
- A second annotator is still required before any `test` split is frozen.
- LoRA fine-tuning is not part of this change. `sekf.judge.calibration.fit_temperature` only rescales cached confidences.
