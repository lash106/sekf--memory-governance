# Development pilot, 29 October 2026

Git commit recorded at run time: `68d7fccd6b64ec1cd09acf6522705e2ccc14e491`.
Split: `dev`. Cases loaded: 63. Mode: `dry_run`. Temperature: 0. Seed: 0.
Env files loaded: .env.example, .env, huggingface.env.
Live API used: no.
Macro-F1 averages F1 over the seven final actions that appear in the gold labels or the predictions. Promotion precision and recall use PROMOTE. Revision accuracy is the fraction of gold REVISE cases that returned REVISE. Contradiction F1 uses Judge label REFUTED as the positive class and is reported only when the system returns a Judge label.
These numbers are a development pilot. They are not a held-out test result, and prompts were not changed to improve them.

| System | n | Final-action macro-F1 | Promotion precision | Promotion recall | Revision accuracy | Contradiction F1 | Median latency (ms) | LLM calls | Tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_storage | 63 | 0.0489 | 0.2063 | 1.0000 | 0.0000 | n/a | 0 | 0 | 0 | 0.0000 |
| latest_write_wins | 63 | 0.1300 | 0.3636 | 0.6154 | 1.0000 | n/a | 0 | 0 | 0 | 0.0000 |
| static_rag | 63 | 0.0425 | n/a | 0.0000 | 0.0000 | n/a | 0 | 0 | 0 | 0.0000 |
| single_judge | not run |  |  |  |  |  |  |  |  |  |
| generic_ensemble | 63 | 0.1286 | 0.2619 | 0.8462 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0000 |
| proposed_council | 63 | 0.4698 | 0.7333 | 0.8462 | 0.9167 | 0.0000 | 0 | 0 | 0 | 0.0000 |

## What each row means

- direct_storage: Krishna `DirectStorageBaseline` (dry_run). Every parsed candidate is PROMOTE.
- latest_write_wins: Krishna `LatestWriteWinsBaseline` (dry_run). REVISE when the candidate is newer than the current ACTIVE claim; KEEP_EPISODIC when not newer; otherwise PROMOTE.
- static_rag: Krishna `StaticRagBaseline` (dry_run). Retrieves by lexical overlap but never writes; action is KEEP_EPISODIC.
- single_judge: not run. `.env.example` leaves SEKF_LLM_API_KEY empty and SEKF_PROVIDER=fake. Copy huggingface.env.example to huggingface.env and set a key to run this row.
- generic_ensemble: three FakeJudge votes. No network. A live three-call panel was not run, to keep the live budget on the council and single_judge.
- proposed_council: DeterministicHistorian, FakeJudge, DeterministicSynthesizer, policy-v1, dry_run. No network.
