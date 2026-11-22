# ADR 0006: Storage and retrieval baselines, and the memory-delta comparison

- Status: accepted
- Date: 2026-11-21
- Owner: Krishna Panjiyar (memory, storage baselines)
- Affected owners: Shailen (experiment runner), Sai Chetan (other baselines use the same links)
- Contract version: v1, unchanged.
- SPEC.md section 18: Sprint 2 "`direct_storage`, `latest_write_wins`, `static_rag`; reusable
  initial-memory fixtures; expected-versus-actual memory-delta comparison; 15+ state-transition
  cases".

## Context

The pilot compares the governed system with systems that skip governance. Until now the three
storage and retrieval baselines existed only as one-line rules inside
`experiments/run_dev_pilot.py`, with no real store behind them. A baseline that never writes
cannot show what ungoverned memory does to the store, which is the claim the project makes.

## Decision

Each baseline is a package under `src/sekf/baselines/` with the same `decide(DecideRequest)`
shape as the other systems. They make no LLM calls and share `storage_common.StorageBaseline`:
choose an action with a fixed rule, build a `DecisionRecord` with the shared link rules
(`decision_links`), and in `commit` mode write a `CommitBundle` through the same
`apply_decision` as the governed system. `dry_run` never writes. A retry returns the stored
decision (idempotent by `decision_id`, as in the orchestrator).

| Baseline | Rule | What it gets wrong, by design |
|---|---|---|
| `direct_storage` | always `PROMOTE` | poisoned, refuted, stale and duplicate claims all become ACTIVE |
| `latest_write_wins` | no ACTIVE claim: `PROMOTE`; candidate observed after the newest ACTIVE claim was updated: `REVISE` that claim; otherwise `KEEP_EPISODIC` | newer but unreliable claims overwrite good ones; past-period claims replace current ones |
| `static_rag` | always `KEEP_EPISODIC`; ranks related claims and evidence by word overlap (Jaccard, deterministic) and records the hits | cannot store a new fact or correct a stale one |

These rules are the ones the pilot script already used, so existing pilot numbers stay
comparable. Policy versions are `direct-storage-v1`, `latest-write-wins-v1`, `static-rag-v1`,
so a shared store never confuses their decisions.

Memory-delta comparison (`sekf.evaluation.memory_delta`): the gold
`expected_memory_delta` and the `MemoryDelta` returned by a commit are put in the same five
buckets, compared as sets, and summarized as an exact-match rate with per-bucket precision and
recall. A claim left as CANDIDATE by `ESCALATE_FOR_REVIEW` is reported separately and not
compared, because the annotation guide defines escalation as "governed memory unchanged".

Fixtures (`sekf.memory.fixtures`): load a labeled case and build a fresh in-memory SQLite store
seeded with its `initial_memory`. Only `SqlMemoryRepository` changes claim state, so
`InMemoryRepository` cannot be used for delta checks.

## Consequences

- `scripts/run_state_transition_eval.py` scores all three baselines plus an offline council
  stand-in on any split. The council row uses `FakeJudge` and is a wiring check, not a result.
- `tests/integration/test_state_transitions.py` commits every case's gold action through the
  store and requires the delta to equal the gold delta. Those checks run on every file under
  `datasets/dev/` (not only `datasets/dev/krishna/`), so a bad gold delta from any owner fails
  this suite. This tests the store against the labels without any model.
- Known limits: `latest_write_wins` compares only against the most recently updated ACTIVE
  claim and supersedes one claim; `static_rag` retrieval is lexical, not embedding-based, so it
  is a floor for retrieval baselines, not a tuned one. `direct_storage` allows several ACTIVE
  claims for one key on purpose.
