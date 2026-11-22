# Sprint 3–4: Sai Chetan Judge / evidence patch

Owner: Sai Chetan Muppalla. Branch: `sai-chetan/sprint-3-4-judge`.

## Sprint 3 (error analysis)

- Judge evidence error analysis (`src/sekf/judge/error_analysis.py`) and offline scorer.
- Calibration (Brier / ECE) plus a descriptive temperature fit on the same `dev` rows (not a selected threshold).
- `FixtureJudge`: lexical offline scorer that can emit `REFUTED` / `MIXED_EVIDENCE` / `INVALID_INPUT` so those gold labels can be scored without a live key. CI still uses `FakeJudge`.
- Saved comparison: `experiments/sai_chetan/dev_2026-11-20/` (re-run with `python experiments/sai_chetan/run_dev_judge.py`).
- Prompt freeze guard: `assert_prompts_frozen` before any held-out run.

## Sprint 4 (hardening)

- `fixture-judge-v2`: date/range parsing, room-token overlap, reliability-aware aggregation,
  softer social-noise handling. Dev fixture score: accuracy 1.0, contradiction F1 1.0,
  Brier 0.0046, ECE 0.0647 (51/51 clean). See `experiments/sai_chetan/dev_2026-11-20/`.
- Parallel-work constraints recorded in `docs/parallel-work.md` (Judge request = candidate + evidence only).
- Live pilot path stays opt-in via gitignored `huggingface.env` (empty key in `huggingface.env.example`).
- Module limitations: FakeJudge never emits REFUTED; FixtureJudge is lexical, not a model; cost is 0 for both offline scorers.

## Out of scope here

- policy-v2, memory projection, contract field changes (need Shailen/Krishna ADR).
- Krishna storage baselines (PR #17).
- Frozen `test/` split and official held-out run (need second annotator + Shailen runner).
