# AGENTS.md

Instructions for any AI coding agent working in this repository (SEKF: Self-Evolving Knowledge Framework). Keep this file short; the full design is in `SPEC.md`.

## What this project is

A research prototype that decides whether a candidate claim should enter an AI agent's long-term memory. Pipeline: Memory lookup -> Historian (time/provenance/conflict) + Judge (evidence support) -> Synthesizer (advisory) -> deterministic Policy -> transactional Memory update. We compare it against baselines on a labeled dataset. Core idea: a newer claim is not automatically a true claim.

## Before you code

1. Read `SPEC.md` sections 0, 4 (worked example), 6 (ownership, branching), 9 (contracts and policy), and the section for your module.
2. Ask the user which module and which `SPEC.md` section 18 task you are working on. Do not modify another owner's production module. Tests, fixtures, prompts, and documentation may be updated when the task requires it.
3. The Pydantic models in `src/sekf/contracts/` define the implemented interface. If the implementation disagrees with `SPEC.md`, do not silently choose one: report the mismatch to the user and update the appropriate artifact through the normal review process.

## Module ownership (do not edit others' modules)

| Path | Owner |
|---|---|
| `src/sekf/memory/`, `src/sekf/baselines/direct_storage`, `src/sekf/baselines/latest_write_wins`, `src/sekf/baselines/static_rag` | Krishna |
| `src/sekf/historian/`, `src/sekf/synthesizer/`, `datasets/ANNOTATION_GUIDE.md`, temporal cases | Mahalakshmi |
| `src/sekf/judge/`, `src/sekf/providers/`, `src/sekf/baselines/single_judge`, `src/sekf/baselines/generic_ensemble` | Sai Chetan |
| `src/sekf/policy/`, `src/sekf/api/`, `src/sekf/evaluation/`, `.github/`, `Dockerfile` | Shailen |
| `src/sekf/contracts/`, `contracts/v1/`, `Makefile`, `pyproject.toml` | Shared: small PRs, all affected owners approve |

If a task needs a change in another owner's module, stop and tell the user. Use the other module's Protocol or a test double instead.

## Commands

```bash
make setup       # create env, install deps
make test        # pytest (offline, fake provider)
make lint        # ruff format check + lint
make typecheck   # mypy/pyright
make contracts   # regenerate contracts/v1/*.json from Pydantic
make run         # start the API locally
```

All targets exist (Python 3.12 required; `make setup` creates `.venv`). If a target is missing or fails, say so rather than inventing a substitute.

Always use the project venv (`.venv`): run tools through `make`, or `source .venv/bin/activate` first. Never `pip install` into the global Python. Add dependencies in `pyproject.toml` (a shared file: small PR), then run `make setup && make lock` and commit the updated `constraints.txt`.

## Canonical names (use exactly; do not invent alternatives)

- Final actions: `PROMOTE`, `ATTACH_EVIDENCE`, `KEEP_EPISODIC`, `QUARANTINE`, `REVISE`, `REJECT`, `ESCALATE_FOR_REVIEW`
- Claim statuses: `CANDIDATE`, `ACTIVE`, `QUARANTINED`, `SUPERSEDED`, `REJECTED`
- Judge labels: `SUPPORTED`, `REFUTED`, `INSUFFICIENT_EVIDENCE`, `MIXED_EVIDENCE`, `INVALID_INPUT`
- "Conflict" is not an action: a conflicted claim is `QUARANTINE`d, with the conflict recorded in the Historian's `conflict_type`.
- Full enum lists live in `SPEC.md` section 9.1 and `src/sekf/contracts/`.

## Rules

- Tests are offline. Every LLM-dependent component must work with the fake provider; never require network or API keys in tests.
- All LLM calls go through `StructuredLLMProvider`. Validate output with Pydantic; at most two repair attempts; raise typed errors on failure.
- Deterministic code (policy, state transitions, metrics, aggregation) must not call an LLM.
- Only the policy engine authorizes memory mutations; the Synthesizer is advisory and may only make outcomes more conservative. Memory writes are transactional and idempotent by `decision_id`.
- Never delete claims or audit records. Revision marks the old claim `SUPERSEDED`.
- Judge must never receive gold labels or the final action. Never tune prompts or thresholds on the `test` split.
- Never silently fall back to a different model, prompt, policy, or evidence set. Record model, prompt version, temperature, seed, tokens, latency, and cost.
- Do not change contracts or enums without an ADR in `docs/decisions/` and approval of every affected owner.
- No secrets in code, fixtures, logs, or commits. Use `.env` (git-ignored) and update `.env.example`.
- Add or update tests with every behavior change. Public functions get docstrings.

## Git workflow

- Never commit to `main`. Branch from latest `main` as `<name>/<short-task>`; keep branches to 1-3 days.
- Rebase on `origin/main` daily. Open a draft PR early; one approval from a different teammate plus green CI to merge; squash-merge; name the `SPEC.md` section 18 task in the PR description.
- Keep PRs small and focused on one task. Do not reformat unrelated files.
- Do not commit, push, or open a PR unless the user asks.

## Definition of done

Acceptance criteria met; tests pass; `make lint` and `make typecheck` pass; docs/fixtures updated; no secrets; ready for a teammate to review.
