# Part-one freeze for Judge, providers, and Sai Chetan's baselines

Owner: Sai Chetan Muppalla. SPEC.md Sprints 3–5 for the modules he owns. Semester 2 work is out of scope.

## Frozen prompts

`sekf.judge.freeze.assert_prompts_frozen` checks these sha256 values, which are the hashes `load_prompt` stores on each call:

| File | Version id | sha256 |
|---|---|---|
| `prompts/judge/v1.txt` | `judge/v1` | `da2bfe3727e347098f0c57745028995eeb2ec8eda71e5f74914df5fd24b7dac3` |
| `prompts/baselines/single_judge/v1.txt` | `baselines/single_judge/v1` | `df160414eb3d1e3de4c08c38e07b3828c295f4a7b979abd88ed6ce400bd3e9cb` |

The earlier hashes were for the prompts before commit `30fb350` added the instruction-boundary lines. This branch pins the text that is in the tree now. The prompt files themselves are unchanged by this update.

`generic_ensemble` has no separate prompt. It calls the Judge it is given. A prompt edit that should become the official version needs a new file, a new hash in `src/sekf/judge/freeze.py`, and a test change in the same commit. The code does not switch files on its own.

Thresholds stay in the policy module. This freeze does not change them.

## Known limitations

- `FakeJudge` is the offline stand-in. It marks `SUPPORTED` only when evidence text contains the claim object, and it never emits `REFUTED`. Pilot contradiction F1 of 0 for the fake council is that behavior, not a tuned result.
- Live calls need `SEKF_LLM_API_KEY`. A missing key raises `ProviderUnavailable`. There is no fallback model.
- Client-side rate limiting and dev-response caching are not enabled (ADR 0002). HTTP 429 becomes `ProviderUnavailable`.
- Models other than `gpt-4o-mini` record `estimated_cost_usd=0` unless both cost environment overrides are set for the model that was actually called.
- Repair stops at two attempts. Unknown evidence ids, duplicate citations, and an id used as both support and refutation fail in the Judge after the provider returns. They do not start another repair.
- `single_judge` refuses `commit` because it does not write memory. `generic_ensemble` maps a quorum `SUPPORTED` vote to `PROMOTE` and does not revise or attach evidence.
- Calibration is not a full distribution over the five Judge labels, because the assessment stores one confidence.

## Sprint 5 check

`tests/unit/test_judge_error_analysis.py` calls `assert_prompts_frozen`. The rest of the owned surface is covered by the provider, Judge, `single_judge`, `generic_ensemble`, matched-settings, and dataset-case tests. Those tests stay on `FakeProvider`.
