# Part-one work that still needs another owner

Owner: Sai Chetan Muppalla. These items sit on the SPEC.md section 18 plan through Sprint 5.
They are not implemented in the Judge, provider, or Sai Chetan baseline folders because each
one needs another module or a second person.

Status checked against `origin/main` after PR #21 (2026-11-21).

| Blocked item | Who has to move first | Current status |
|---|---|---|
| Grow the shared set toward 120–200 cases and freeze `test` | A second annotator, then Shailen for the split | Still blocked. Dev has grown (Mahalakshmi #21); no frozen `test/` yet. |
| Dataset loader, experiment manifest, resumable runner | Shailen (`src/sekf/evaluation/`) | **On main** via #16 (`manifest`, `jsonl_store`, agreement checker). Wire the pilot to those next. |
| Required ablations and paired bootstrap intervals | Shailen + Mahalakshmi | Historian ablations **on main** (#18). Bootstrap on saved pilot rows still open. |
| First official held-out run | Shailen, after the frozen `test` split | Not started. Judge prompts are frozen in this package so that run does not tune them. |
| Storage and state error analysis | Krishna | **On main** via #17 (`memory_delta`, state-transition eval). Wire deeper storage error tables next. |
| Provenance, time, and conflict error analysis | Mahalakshmi | **On main** (#18–#21). Group-C gold relabels are on the review-checks branch, not yet on main. |
| Policy, veto, and cost error analysis | Shailen | Still open. |
| `direct_storage`, `latest_write_wins`, `static_rag` packages | Krishna | **On main** via #17. `experiments/run_dev_pilot.py` calls the packaged baselines in dry_run. |
| Docker image, pinned clean-environment boot, tag `v0.1.0` | Shailen, with the team | Docker/REPRODUCE **on main** via #16. Tag `v0.1.0` still shared. |
| Demo rehearsal of promote, revise, and quarantine | All four members | Needs one running server with all roles. |
| Live `single_judge` row in the pilot | Whoever holds the API budget | Copy `huggingface.env.example` → `huggingface.env` (git-ignored) and set a key. Never commit the key. |
| Second model, extra seeds, local model | Semester 2 | SPEC.md section 18 stops part one at the December demonstration. |
| policy-v2 / memory projection / DecisionRecord risk fields | Shailen + Krishna (ADR) | Open as `sai-chetan/governance-metrics-v2` with **ADR 0007** (Krishna keeps `0006-storage-baselines`). Needs Shailen + Krishna review before merge. |

## How to re-run the saved offline pilot

```bash
# offline (fake provider): uses .env.example defaults
python experiments/run_dev_pilot.py

# live single_judge / council Judge (optional):
# 1. Revoke any old HF token that was ever committed.
# 2. cp huggingface.env.example huggingface.env
# 3. paste a fresh token into huggingface.env (never commit it)
# 4. python experiments/run_dev_pilot.py
```

Saved numbers: `experiments/dev_pilot_2026-10-29/`.
