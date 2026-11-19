# ADR 0006: policy-v2, Markdown projection, and separate retrieval recall

- Status: proposed for review by Shailen (policy, API, evaluation) and Krishna (memory)
- Date: 2026-11-18
- Owners affected: Shailen, Krishna, Sai Chetan (Judge calibration helper)

## Context

SPEC.md section 9.4 is one global threshold table. A customer-service claim about a preference and a claim about a payment should not share it. The repository also had no offline place to report section 14 metrics with a paired bootstrap interval, and no readable projection of a committed decision.

Pydantic in this repository is already v2 (`model_validate`, `ConfigDict`). There is no v1 model to freeze and no migration in this change.

## Decision

1. `PolicyEngine` / `policy-v1` is unchanged and remains the default. `SEKF_POLICY=v2` selects `PolicyEngineV2`. Unknown values raise `ConfigurationError`.
2. `DecisionRecord` adds four optional fields, also present in the generated `AuditEntry`, `AuditTrace`, `CommitBundle`, and `DecideResponse` schemas: `risk_tier`, `needs_verification`, `compliance_outcome`, and `next_evidence_required`. They default, so an older payload still validates, and unknown fields stay rejected. policy-v2 fills them. policy-v1 leaves `risk_tier`, `compliance_outcome`, and `next_evidence_required` null and `needs_verification` false. The same facts remain in `explanation` and `rule_ids_fired`.
3. `R_COMPLIANCE` does not change `final_action`. Epistemic truth and the service-owner check stay distinct.
4. Claims are never deleted. A 30-day episodic TTL only drops old `CANDIDATE` rows from `GET /v1/customers/{subject_key}/brief`.
5. `search_related_claims` stays an exact subject+predicate match. Hybrid BM25 over the same subject is a separate function used for Retrieval Recall@K.
6. `markdown_root` / `SEKF_MARKDOWN_DIR` appends L0–L3 Markdown after a successful commit. The SQLite row is the source of truth. A projection failure after a SQL commit raises `ProjectionFailed` and a retry replays the decision. The in-memory double rolls the bundle back and raises `CommitRejected`.
7. `POST /v1/experiments/run` scores `datasets/dev` with deterministic roles and the fake Judge. `single_judge` is not run without a real provider. The endpoint does not substitute another model.
8. One live capture writes `experiments/runs/<id>/predictions.jsonl`. `scripts/run_offline_experiment.py`, `scripts/sweep_policy_thresholds.py`, and `scripts/analyze_judge_evidence.py` replay that file. HTTP 429 is retried only in that capture path.
9. While `datasets/test/` does not exist, threshold selection is leave-one-author-out dev CV across the author folders in `datasets/dev`. The reported number is the mean of the held-out folds. A full-split score is labeled descriptive and is not the selected result. A fitted Judge temperature is applied inside `PolicyEngineV2` before R2 and R5; temperature 1.0 leaves the raw confidence unchanged and keeps the version string `policy-v2`.

## Consequences

- Quote cached dev scores as leave-one-author-out dev CV. They are not a held-out test result.
- A second annotator is still required before any `test` split is frozen.
- LoRA fine-tuning is not part of this change. `sekf.judge.calibration.fit_temperature` only rescales cached confidences.
- `R_COMPLIANCE` still does not change `final_action`. `compliance_outcome` stores the clause.
