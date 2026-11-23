# Judge comparison on dev (Sprint 3–4 refresh)

Owner: Sai Chetan Muppalla. Split: `dev`. Cases loaded: 63 (current `main`, including
Krishna #17 state-transition cases). Mode: judge-only; `dry_run` for ensemble rows.
Temperature: 0. Seed: 0. Re-run: `python experiments/sai_chetan/run_dev_judge.py`.

Recorded run mode:

- `SEKF_MEMORY=memory`. This run does not check claim status after a commit.
- `SEKF_PROVIDER=fake` for the CI Judge row. The fixture row does not call a provider.
- `SEKF_HISTORIAN` and `SEKF_SYNTHESIZER` were not invoked.
- `single_judge` was not run. This process has no live API key.
- Policy was not applied. The ensemble row uses `generic-ensemble-v1`, which maps a Judge
  label to an action and does not revise.
- `fixture-judge-v3` `prompt_hash` is sha256 of that version string, not of a prompt file.
- Cost is 0. Neither row calls a priced model.

These numbers are a development comparison. They are not a held-out test result. A
temperature fit on these same rows is stored in `summary.json` and is not a
selected threshold.

The CI Judge is still `FakeJudge`. `FixtureJudge` is a separate lexical scorer that can
also return `REFUTED`, `MIXED_EVIDENCE`, and `INVALID_INPUT`. It sees the candidate and
the evidence only. v3 keeps v2 date/range/room/reliability behavior and adds day-first
dates (`15 September`) so registrar-style evidence is not missed.

| Scorer | n | Label accuracy | Contradiction F1 | Brier | ECE | Clean cases | Wrong labels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fake judge | 63 | 0.6190 | 0.0000 | 0.3148 | 0.2810 | 39 | 24 |
| fixture judge | 63 | 1.0000 | 1.0000 | 0.0046 | 0.0644 | 63 | 0 |

Gold labels and the full confusion matrices are in `summary.json`. Contradiction F1 uses
REFUTED as the positive class. The fake Judge's contradiction F1 is 0 because it never
emits REFUTED.

| Ensemble member | Final-action macro-F1 | Promotion precision | Promotion recall | Revision accuracy | Identical 3-member panels |
| --- | --- | --- | --- | --- | --- |
| fake judge | 0.1286 | 0.2619 | 0.8462 | 0.0000 | 63 / 63 |
| fixture judge | 0.5966 | 0.3421 | 1.0000 | 0.0000 | 63 / 63 |

Revision accuracy is 0 on both rows because this baseline maps SUPPORTED to PROMOTE and
does not emit REVISE. All panels were unanimous. The fixture ignores seed, so this is
the temperature-0 degeneration described in SPEC.md section 12.3.

## Fixture disagreements

None on this 63-case `dev` run after `fixture-judge-v3`. Remaining ensemble gaps are
policy/baseline limits (no REVISE), not Judge label misses.

## Files

- `fake_observations.jsonl` / `fixture_observations.jsonl` for `scripts/analyze_judge_evidence.py`.
- `error_analysis.json` is the fixture report from the same run as `summary.json`.
- `summary.json` is the full comparison, including confusion matrices.
