# Development pilot, 29 October 2026

Git commit recorded at run time: `56689bf4fa987658885c64591f548db551befe82`.
Split: `dev`. Cases loaded: 35. Mode: `dry_run`. Temperature: 0. Seed: 0.
Judge for the council and the ensemble: `FakeJudge` (SUPPORTED when evidence text contains the object text, otherwise INSUFFICIENT_EVIDENCE). It does not emit REFUTED, so contradiction F1 on the REFUTED class is 0 whenever that gold label is present.
Macro-F1 averages F1 over the seven final actions that appear in the gold labels or the predictions. Promotion precision and recall use PROMOTE. Revision accuracy is the fraction of gold REVISE cases that returned REVISE. Contradiction F1 uses Judge label REFUTED as the positive class and is reported only when the system returns a Judge label.
These numbers are a development pilot. They are not a held-out test result, and prompts were not changed to improve them.

| System | n | Final-action macro-F1 | Promotion precision | Promotion recall | Revision accuracy | Contradiction F1 | Median latency (ms) | LLM calls | Tokens | Cost (USD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_storage | 35 | 0.0418 | 0.1714 | 1.0000 | 0.0000 | n/a | 0 | 0 | 0 | 0.0000 |
| latest_write_wins | 35 | 0.1442 | 0.3846 | 0.8333 | 1.0000 | n/a | 0 | 0 | 0 | 0.0000 |
| static_rag |  |  |  |  |  |  |  |  |  |  |
| single_judge |  |  |  |  |  |  |  |  |  |  |
| generic_ensemble | 35 | 0.1464 | 0.2500 | 1.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0000 |
| proposed_council | 35 | 0.5129 | 0.6667 | 1.0000 | 0.8571 | 0.0000 | 0 | 0 | 0 | 0.0000 |

## What each row means

- direct_storage: Specified rule in this script. Package is not in the repository. Every parsed candidate is PROMOTE.
- latest_write_wins: Specified rule in this script. Package is not in the repository. REVISE when a related ACTIVE claim is older than the candidate; KEEP_EPISODIC when a related ACTIVE claim is not older; otherwise PROMOTE.
- static_rag: Not run. The package is not in the repository and this pilot makes no model call.
- single_judge: Not run. No SEKF_LLM_API_KEY is configured, so the live one-model baseline was not called.
- generic_ensemble: Ran. Three FakeJudge votes. No network.
- proposed council: Ran. DeterministicHistorian, FakeJudge, DeterministicSynthesizer, policy-v1, dry_run. No network.
