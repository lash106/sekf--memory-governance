# ADR 0001: Atomic commit bundle and reproducibility metadata

- Status: proposed (becomes accepted when Krishna, Mahalakshmi, and Sai Chetan approve the skeleton PR)
- Date: 2026-10-23
- Affected owners: Krishna (memory), Mahalakshmi (Historian/Synthesizer), Sai Chetan (Judge/providers), Shailen (policy/API)
- Contract version: v1, kept. These changes are NOT backward compatible with the first skeleton draft: several `CallMetadata` fields became required, `AuditTrace.decisions` became `AuditTrace.bundles`, `DecisionRecord` gained a required `request_hash`, and the memory write method changed. Nothing has been merged or released against v1 yet, so there are no consumers to break and we treat this as a pre-release change. After the first merge to `main`, breaking changes require a version bump.

## Context

A code review of the Sprint 0 skeleton found two design gaps.

1. Commit mode stored the candidate and evidence *before* the roles and policy ran, then recorded the decision separately. A crash in between left a candidate with no decision or audit record, which contradicts "only the policy authorizes mutations" and "writes are transactional".
2. `CallMetadata` did not record temperature, seed, prompt hash, or provider, and `llm_calls` was computed by counting role outputs. SPEC.md sections 11 and 15 require this metadata, and cost-normalized comparison (RQ2, RQ4) needs real call counts, including repair attempts.

3. `decision_id` ignored thresholds and evidence, so a changed input could be mistaken for an idempotent replay; the first fix then hashed role outputs too, so a retry against a real LLM (new UUIDs, latency, wording) could never be a replay.
4. The commit lost the actual role failures and the run configuration; the in-memory repository dropped the role outputs; bundle consistency checks were thin; and `LLMResult` could not report real call counts.

## Decision

1. **One atomic write.** `MemoryRepository` has side-effect-free reads and a single write, `apply_decision(bundle: CommitBundle)`. The bundle holds the episode (optional), candidate, evidence, role outputs, decision, typed `role_errors`, and the `run_config`. `AuditTrace` returns one entry per committed decision: the bundle plus the `MemoryDelta` it actually produced, so exact state changes and attached evidence stay in the permanent history. The repository validates the bundle first, then persists everything in one transaction, idempotent by `decision.decision_id`. An inconsistent bundle raises `CommitRejected` and writes nothing. The separate `store_episode`, `store_candidate`, and `store_evidence` methods are removed from the protocol.
2. **Nothing is written before the policy decides.** The orchestrator computes everything read-only, then makes the one commit call (commit mode only). Role failures still produce an `ESCALATE_FOR_REVIEW` decision whose bundle is stored, so the review queue and audit trail stay complete. A crash before the commit leaves memory untouched and the request can be retried.
3. **Call metadata.** `CallMetadata` gains `provider`, `temperature`, `seed`, `prompt_hash`, and `llm_calls` (real provider calls including repairs; fake or deterministic roles report 0). All except `seed` are required, so omissions fail validation. `LLMResult` gains `provider` and `llm_calls` (tokens, latency, and cost are summed across repair attempts), which roles copy into their metadata. `CostSummary.llm_calls` sums them.
4. **`decision_id`** is a UUIDv5 of the policy version and a SHA-256 of the *stable request* (details below). Role outputs are excluded. A retry maps to the same ID; the first committed decision for an ID wins; replaying an ID with different candidate or evidence raises `CommitRejected`. A different `idempotency_key` forces a deliberate re-run.
   - The fingerprint (`DecisionRecord.request_hash`) covers the full candidate, every evidence record in full (including reliability, source, timestamps), thresholds, the whole run configuration (including `timeout_s`, so a retry with a longer timeout can succeed instead of replaying a timeout escalation), and the policy version. It deliberately EXCLUDES the memory snapshot: a committed decision changes memory (REVISE supersedes a claim), and the retry must still be recognized afterwards. Repositories compare `request_hash` to tell a true replay from a reused ID. To re-evaluate the same candidate against newer memory, send a new `idempotency_key`.
   - The snapshot the roles saw is stored separately as `CommitBundle.memory_before`, so a decision can be reconstructed and audited.
   - In commit mode the orchestrator computes the ID before running any role and answers a retry from the stored bundle, so a retry spends no LLM calls and the client always sees the decision that was actually committed.
   - The identity is recomputable from the stored bundle (`compute_request_hash`, `compute_decision_id` in `sekf.contracts`; `idempotency_key` is stored in the bundle). The consistency check recomputes both and rejects a bundle whose `request_hash` or `decision_id` does not match, so a repository never has to trust them.
   - Trade-off: an escalated-by-failure decision that was committed will be replayed on retry until the caller supplies a new `idempotency_key`.
5. **R7** has no source-reliability condition, matching SPEC.md section 9.4. Attaching evidence to an active claim is non-destructive and the reliability score is still recorded.
6. **Duplicate evidence IDs** are rejected in `DecideRequest` and `CommitBundle`, whether or not the contents differ.
7. **Stale snapshots are rejected, including phantoms.** Memory may change between reading it and committing. `apply_decision` must compare the COMPLETE current related-claim set (same subject and predicate) with `memory_before`: a claim that was changed, removed, or newly added makes the decision stale. It then raises `CommitRejected` and writes nothing; the caller retries with a fresh read. Checking only the claims already in the snapshot is not enough (two requests that both read an empty memory could both promote).
8. **Stored audit history is immutable to callers.** Repositories store and return copies (or frozen models) so a returned bundle or trace cannot alter history.

## Requirements for the real memory store (Krishna, Sprint 1)

The in-memory repository is a placeholder that performs no claim state transitions, so it cannot expose these. The SQLAlchemy implementation must:

- apply the claim state transitions of the decision (PROMOTE, REVISE, ATTACH_EVIDENCE, QUARANTINE, REJECT) inside the same transaction as the bundle write;
- do the stale-snapshot check (item 7, including newly added claims) inside that transaction, so two concurrent decisions cannot both supersede or both promote. The strongest guard is a partial unique index allowing at most one `ACTIVE` claim per (`subject_key`, `predicate_key`), plus row versioning on the claims being superseded;
- persist the actual `MemoryDelta` with the bundle (resulting statuses, attached evidence) so the audit trace keeps it;
- make `get_bundle(decision_id)` return the original bundle after the transitions have changed memory, since replays are answered from it;
- treat `decision_id` as unique and compare `request_hash` on conflict;
- ensure a retry after a committed REVISE is a replay, not a second decision (integration test).

## Consequences

- Krishna implements `apply_decision(bundle)` transactionally with SQLAlchemy; the in-memory repository is a placeholder.
- Real roles (Sai Chetan, Mahalakshmi) must fill `provider`, `prompt_hash`, `temperature`, `seed`, and `llm_calls` from the provider result and run configuration; validation fails otherwise.
- `CommitBundle` validates its own consistency (shared `check_bundle_consistency`); a real repository additionally checks references against stored state (for example that related claim IDs exist).
- Per-run manifests (SPEC.md section 15) can now be rebuilt from stored role outputs.
