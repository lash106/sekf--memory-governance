# ADR 0004: Memory store schema (claims, versions, relationships)

- Status: proposed (to be accepted by Shailen, as API/integration owner, and Mahalakshmi as reviewer)
- Date: 2026-10-27
- Owner: Krishna Panjiyar (memory)
- Affected owners: Shailen (orchestrator wiring), Mahalakshmi (Historian reads related claims)
- Contract version: v1, unchanged. No Pydantic model or enum changes; this ADR only fixes how
  the existing contracts are persisted.
- SPEC.md section 18: Sprint 0 "draft the DB schema and an ADR for the claim/version/relationship
  model" and Sprint 1 "SQLAlchemy models and Alembic migrations; repository methods;
  `search_related_claims`; transactional, idempotent `apply_decision`".

## Context

ADR 0001 defines the single write, `apply_decision(bundle)`, and lists what the real store must
guarantee. The skeleton `InMemoryRepository` records bundles but never changes claim state. We
need a persistent store (SQLite by default, `SEKF_DB_URL`) that applies the action semantics of
SPEC.md section 9.1 and never loses history.

## Decision

### Tables (`src/sekf/memory/schema.py`, migration `0001_initial`)

| Table | Purpose |
|---|---|
| `claims` | One row per governed claim. Mirrors `ExistingClaim`, plus `row_version` (optimistic lock) and `created_by_decision_id`. Statuses are the canonical `ClaimStatus` values. |
| `claim_evidence` | Link table claim -> evidence, with the `decision_id` that attached it. `ExistingClaim.evidence_ids` is read from here in attach order. |
| `claim_relationships` | Typed edges (`RelationshipType`): `SUPERSEDES` (REVISE), `DUPLICATES` (ATTACH_EVIDENCE, candidate -> target), `CONTRADICTS` (QUARANTINE when the Historian reports a value, source or temporal-update conflict or overlapping validity: candidate -> each related ACTIVE claim it named). `from_id` may be a candidate that never became a claim, so it has no foreign key. |
| `evidence` | Immutable evidence records, keyed by `evidence_id`. Re-sending the same ID with different content is rejected. |
| `episodes`, `candidates` | Raw inputs, kept for every decision (including `KEEP_EPISODIC`). |
| `decisions` | One row per committed decision: `decision_id` (PK), `request_hash`, the full `CommitBundle` JSON, the produced `MemoryDelta` JSON, and an insertion sequence for audit order. Never updated or deleted. |
| `review_queue` | `ESCALATE_FOR_REVIEW` entries (decision, claim, reason). |
| `claim_keys` | One row per (`subject_key`, `predicate_key`) with a `revision` counter, used as the concurrency guard (below). |

### Action semantics

The candidate's `claim_id` is the ID of the claim it becomes.

| Action | Effect |
|---|---|
| `PROMOTE` | Candidate claim `ACTIVE`, version 1, evidence linked. |
| `REVISE` | Target (`previous_claim_ids[0]`, must be `ACTIVE`) becomes `SUPERSEDED` (row kept). Candidate becomes `ACTIVE` with `version = target.version + 1`, `supersedes_claim_id = target`, and a `SUPERSEDES` edge. |
| `ATTACH_EVIDENCE` | Target (`resulting_claim_id`, must be `ACTIVE`) gets the bundle's evidence linked; no new claim, no new version. `attached_evidence_ids` lists only newly linked evidence. |
| `QUARANTINE` | Candidate claim `QUARANTINED`; ACTIVE claims unchanged; `CONTRADICTS` edges recorded. |
| `REJECT` | Candidate claim `REJECTED` (kept for audit). |
| `KEEP_EPISODIC` | Episode/candidate/evidence stored; no claim row. |
| `ESCALATE_FOR_REVIEW` | Candidate claim stored as `CANDIDATE` (or left as it is if it already exists) and added to `review_queue`. |

A candidate whose own claim already exists may move `CANDIDATE`/`QUARANTINED` -> any action's
status (re-evaluation with a new `idempotency_key`). A claim that is already `ACTIVE`,
`SUPERSEDED` or `REJECTED` cannot be re-created or re-transitioned by a mutating action; that
raises `CommitRejected`. Nothing is ever deleted.

The decision's `previous_claim_ids` / `resulting_claim_id` must match the action (same shape as
`sekf.baselines.decision_links`); otherwise the bundle is rejected. This keeps the stored delta
and the decision record from disagreeing.

### Concurrency and atomicity

Everything for one bundle happens in one database transaction. Inside it the repository:

1. bumps `claim_keys.revision` for the candidate's (`subject_key`, `predicate_key`), taking the
   row lock (SQLite: the transaction starts with `BEGIN IMMEDIATE`, so writers are serialized);
2. checks idempotency (`decision_id` exists -> replay if `request_hash` matches, else reject);
3. compares the COMPLETE current related-claim set with `memory_before` (ADR 0001 item 7,
   phantoms included);
4. applies the state transitions, stores the bundle and the delta.

Any failure rolls the whole transaction back.

We deliberately do **not** use a partial unique index "at most one ACTIVE claim per key", which
ADR 0001 suggested. Claims with non-overlapping validity windows (dataset category
`non_overlapping_temporal_validity`) can legitimately both be `ACTIVE`, and the per-key lock row
already prevents two concurrent decisions from both promoting or both superseding.

### Migrations

Alembic scripts live in `src/sekf/memory/migrations/`. `run_migrations(url)` upgrades to head
programmatically, so no root `alembic.ini` is needed. A test checks that the migrated schema
matches the SQLAlchemy models.

## Consequences

- `SqlMemoryRepository` implements `MemoryRepository`; `InMemoryRepository` stays as the
  lightweight test double.
- **Not yet connected to the application.** `src/sekf/api/app.py` still builds an
  `InMemoryRepository`, so `make run` keeps memory only until restart. Follow-up for Shailen
  (API owner): use `build_memory_repository()` in `build_default_governance`, and map
  `CommitRejected` from `/v1/decide` to HTTP 409 instead of 500.
- One configuration variable: `SEKF_DB_URL` (see `.env.example`). The application and the
  migrations use the same URL because the repository runs the migrations itself; there is no
  separate `alembic.ini` or `SEKF_DATABASE_URL`. The parent directory of a SQLite file is
  created on open.
- Changing the schema needs a new Alembic revision; never edit `0001_initial` after merge.
