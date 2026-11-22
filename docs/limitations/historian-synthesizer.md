# Known limitations: Historian and Synthesizer

Owner: Mahalakshmi. Sprint 4 task (SPEC.md section 18: "each owner freezes their module and
documents known limitations"). Applies to the frozen modules at `v0.1.0`.

Frozen prompts: `prompts/historian/v1.txt` and `prompts/synthesizer/v1.txt`, pinned by hash in
`tests/unit/test_frozen_prompts.py`. Rule-based versions: `deterministic-historian-v1` and
`deterministic-synthesizer-v1`.

## Historian, rule-based (`src/sekf/historian/deterministic.py`)

| # | Limitation | Effect | Mitigation |
|---|---|---|---|
| H1 | Never outputs `SOURCE_CONFLICT`: it reads dates and reliability scores, not what the evidence says. | Two equally reliable sources that disagree look like a `TEMPORAL_UPDATE`, which points toward `REVISE` instead of `QUARANTINE`. 3 of 36 `dev` cases. | The LLM layer adds `SOURCE_CONFLICT` (see L1). Without it, rely on the Judge's `MIXED_EVIDENCE` label. |
| H2 | Same value is checked by exact text after lowercasing and trimming spaces. | "18:00" and "6 PM" count as different values, so a restatement looks like a conflict. | Normalize values at extraction time; the LLM layer can only swap `DUPLICATE` and `PARAPHRASE`, not fix this. |
| H3 | Related claims must have exactly the same `subject_key` and `predicate_key`. | A different key spelling ("lab" vs "laboratory") gives `NO_RELATED_CLAIM`, so a contradiction can be promoted as new. | Depends on consistent extraction keys (outside the Historian). |
| H4 | Provenance score is the **best** source's reliability. It ignores how many sources agree and ignores the candidate's own `source_id`. | One strong source among weak ones gives HIGH. | Documented choice (`provenance_score`, v0 rule). |
| H5 | A candidate with no validity dates is ordered by `observed_at` against the claim's `updated_at`. | `updated_at` can move when memory changes for other reasons, which could flip "newer" to "older". | Fixed the worse failure (group A in the error analysis); this residual risk remains. |
| H6 | More than one overlapping active claim gives `UNKNOWN` at confidence 0.50. | Always escalates for review; never picks one. | Intended: ambiguity goes to a human. |
| H7 | Reliability bands are fixed: 0.8+ HIGH, 0.6 to 0.8 MEDIUM, below 0.6 LOW. Observation times within 1 minute count as the same time. | Results near the cut-offs are sensitive to these values. | Do not change after the first `test` run; tune on `dev` only. |

## Historian, LLM layer (`src/sekf/historian/llm.py`)

| # | Limitation | Effect |
|---|---|---|
| L1 | The model may only *add* `SOURCE_CONFLICT` or swap `DUPLICATE`/`PARAPHRASE`. A `SOURCE_CONFLICT` is accepted only with two or more cited evidence records from different sources within 0.30 reliability of each other. | It cannot fix a wrong time label from the rules (H5). A real conflict where the model cites badly is not accepted. |
| L2 | Any other disagreement with the rules lowers confidence to 0.50. | The case escalates for review: safe, but it costs a human look. |
| L3 | Costs model calls, and results at temperature 0 are only as repeatable as the provider. | Report calls, tokens, and seeds with every run (SPEC.md section 12.3). |

## Synthesizer (`src/sekf/synthesizer/deterministic.py`, `llm.py`)

| # | Limitation | Effect |
|---|---|---|
| S1 | Advisory only: it can make an outcome more cautious (veto to `QUARANTINE`), never less. | It cannot rescue a wrong `QUARANTINE` or `REJECT`. Intended (SPEC.md section 10.2). |
| S2 | The base action is a fixed mapping from the Judge label and the Historian conflict type. | Any Historian label error flows straight into the recommendation (for example H1: `REVISE` instead of `QUARANTINE`). |
| S3 | The S4 check vetoes any change to memory when provenance is LOW or UNKNOWN. | With the "without provenance fields" ablation, hidden reliability is 0.5 (LOW), so every change is vetoed. Report this when reading that ablation. |
| S4 | Cross-role checks are rules (S1 to S4, M1, M2). Whether the Judge's and Historian's rationales describe the same fact needs the LLM layer. | The rule-based Synthesizer cannot catch two roles talking about different facts. |
| S5 | The LLM layer ignores a model that wants to change memory where the rules would not. | It records the model's view as an open question; it never acts on it. |

## Not covered

- Claims with more than one value at once (lists, ranges).
- Time zones beyond what timestamps carry: all comparisons use aware UTC datetimes.
- Adversarial content in evidence is not filtered by the Historian; the Judge and policy handle
  it (see the prompt-injection cases).
