# ADR 0007: Markdown projection, retrieval recall, and offline metrics

- Status: proposed for review by Shailen (evaluation, API) and Krishna (memory)
- Date: 2026-11-18
- Owners affected: Shailen, Krishna, Sai Chetan (Judge calibration helper)

## Context

The repository had no offline place to report SPEC.md section 14 metrics with a paired
bootstrap interval, and no readable projection of a committed decision.
`search_related_claims` stays an exact subject+predicate match; Retrieval Recall@K needs a
separate ranking helper over the same subject.

Pydantic in this repository is already v2 (`model_validate`, `ConfigDict`). There is no v1
model to freeze and no migration in this change.

Optional customer-service policy-v2 (`PolicyEngineV2`, `R_COMPLIANCE`, customer brief, 30-day
episodic TTL) is held on a separate branch until the advisor approves that direction. It is
not part of this ADR.

## Decision

1. `search_related_claims` stays an exact subject+predicate match. Hybrid BM25 over the same
   subject is a separate function used for Retrieval Recall@K.
2. `markdown_root` / `SEKF_MARKDOWN_DIR` appends L0–L3 Markdown after a successful commit. The
   SQLite row is the source of truth. L0/L1/L2 are written first (each line carries
   `decision:{id}`); the L3 audit entry is the completion marker and is written last. A
   projection failure after a SQL commit raises `ProjectionFailed` (HTTP 503, retryable);
   retrying the same decision finishes any missing Markdown layers. The in-memory double
   rolls the bundle back and raises `CommitRejected`.
3. `POST /v1/experiments/run` scores `datasets/dev` with the packaged baselines in
   `src/sekf/baselines/`, deterministic roles, and the fake Judge. Metrics reported:
   macro-F1 (with paired bootstrap CI), promotion precision/recall, revision accuracy,
   false-promotion rate, unnecessary-quarantine rate. `single_judge` is not run without a
   real provider. The endpoint does not substitute another model.
4. Claims are never deleted.

## Consequences

- Section 14 numbers from this runner are an offline dev score, not a held-out test result.
- A second annotator is still required before any `test` split is frozen.
- LoRA fine-tuning is not part of this change. `sekf.judge.calibration.fit_temperature` only
  rescales cached confidences.
