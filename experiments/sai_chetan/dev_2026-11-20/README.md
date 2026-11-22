# Judge comparison on dev (Sprint 3–4 refresh)

Owner: Sai Chetan Muppalla. Split: `dev`. Cases loaded: 51 (includes Mahalakshmi #21
temporal cases on current `main`). Mode: judge-only; `dry_run` for ensemble rows.
Temperature: 0. Seed: 0. Re-run: `python experiments/sai_chetan/run_dev_judge.py`.

Recorded run mode:

- `SEKF_MEMORY=memory`. This run does not check claim status after a commit.
- `SEKF_PROVIDER=fake` for the CI Judge row. The fixture row does not call a provider.
- `SEKF_HISTORIAN` and `SEKF_SYNTHESIZER` were not invoked.
- `single_judge` was not run. This process has no live API key.
- Policy was not applied. The ensemble row uses `generic-ensemble-v1`, which maps a Judge
  label to an action and does not revise.
- `fixture-judge-v2` `prompt_hash` is sha256 of that version string, not of a prompt file.
- Cost is 0. Neither row calls a priced model.

These numbers are a development comparison. They are not a held-out test result. A
temperature fit on these same rows is stored in `summary.json` and is not a
selected threshold.

The CI Judge is still `FakeJudge`. `FixtureJudge` is a separate lexical scorer that can
also return `REFUTED`, `MIXED_EVIDENCE`, and `INVALID_INPUT`. It sees the candidate and
the evidence only. v2 adds date/range parsing, room-token overlap, reliability-aware
aggregation (weak evidence does not overturn strong support), and softer social-noise
handling when a matching clock is asserted.

| Scorer | n | Label accuracy | Contradiction F1 | Brier | ECE | Clean cases | Wrong labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fake judge | 51 | 0.5882 | 0.0000 | 0.3394 | 0.3118 | 30 | 21 |
| fixture judge | 51 | 1.0000 | 1.0000 | 0.0046 | 0.0647 | 51 | 0 |

Gold labels in this split: SUPPORTED 31, INSUFFICIENT_EVIDENCE 9, MIXED_EVIDENCE 6,
REFUTED 4, INVALID_INPUT 1. Contradiction F1 uses REFUTED as the positive class. The fake
Judge's contradiction F1 is 0 because it never emits REFUTED. All 21 fake misses are
overconfident (confidence 0.90).

| Ensemble member | Final-action macro-F1 | Promotion precision | Promotion recall | Revision accuracy | Identical 3-member panels |
| --- | --- | --- | --- | --- | --- |
| fake judge | 0.1179 | 0.2353 | 0.8000 | 0.0000 | 51 / 51 |
| fixture judge | 0.5935 | 0.3226 | 1.0000 | 0.0000 | 51 / 51 |

Revision accuracy is 0 on both rows because this baseline maps SUPPORTED to PROMOTE and
does not emit REVISE. All 51 panels were unanimous. The fixture ignores seed, so this is
the temperature-0 degeneration described in SPEC.md section 12.3.

## Fixture disagreements

None on this 51-case `dev` run after `fixture-judge-v2`. Remaining ensemble gaps are
policy/baseline limits (no REVISE), not Judge label misses.

## Files

- `fake_observations.jsonl` / `fixture_observations.jsonl` for `scripts/analyze_judge_evidence.py`.
- `error_analysis.json` is the fixture report from the same run as `summary.json`.
- `summary.json` is the full comparison, including confusion matrices.
