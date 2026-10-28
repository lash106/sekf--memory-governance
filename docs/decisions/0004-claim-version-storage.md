# ADR 0004: Claim versions and persistent memory schema

- Status: proposed
- Date: 2026-10-27
- Owner: Krishna Panjiyar
- Affected owners: Krishna (memory), Shailen (shared integration)
- Related tasks: SPEC.md section 18, Sprint 0 and Sprint 1

## Context

The Sprint 0 memory spike needs a durable representation for claim versions, relationships,
evidence, decisions, and the complete audit bundle. Sprint 1 must apply a policy decision and
record its audit data atomically, while preserving old claim versions.

## Decision

SQLite is the prototype persistence backend, mapped with SQLAlchemy 2. Claims use UUID primary
keys and retain their status, monotonically increasing version, superseded-claim link, and full
canonical JSON representation. A `SUPERSEDES` relationship is also stored explicitly. Evidence
records and claim/evidence links are separate rows; raw episodes are retained when supplied.
Each decision row stores the stable request hash, full `CommitBundle`, and resulting
`MemoryDelta`, making the audit record independent of later state changes.

`apply_decision` validates the request and its complete related-claim snapshot while holding a
SQLite `BEGIN IMMEDIATE` write transaction. It then applies the action's claim changes, writes
evidence and relationships, and stores the bundle and delta before committing. A unique
decision ID makes retries idempotent; a partial unique index permits at most one `ACTIVE` claim
per subject/predicate pair. Claim rows are never deleted. Revision marks the old claim
`SUPERSEDED` and inserts a new active version linked to it.

The repository supports SQLite only in this sprint. `InMemoryRepository` remains available for
offline unit tests and test doubles. Alembic owns schema evolution; the initial revision is
`0001_memory_schema`.

## Consequences

- The SQL repository provides persistent reads, state transitions, and audit traces.
- SQLite serializes writers; stale snapshots, including newly inserted related claims, are
  rejected before applying a decision.
- Migrations are forward-only because downgrading the initial schema would delete claims and
  audit history.
- Other database backends, migration of an existing in-memory process, and claim search beyond
  exact subject/predicate matching remain out of scope.

## Review

This ADR is proposed for review by the affected owners. No shared contract or enum changes are
included.
