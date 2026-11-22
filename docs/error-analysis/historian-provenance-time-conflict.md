# Error analysis: provenance, time, and conflict (Historian)

Owner: Mahalakshmi. Sprint 3 task (SPEC.md section 18). Reviewer: Krishna.

**Scope.** The deterministic Historian (`src/sekf/historian/deterministic.py`) on the 36 labeled
`dev` cases. Role labels only (`temporal_relation`, `conflict_type`, `provenance_quality`);
final actions are scored by the experiment runner. These are `dev` results, used for tuning, not
the frozen `test` split. With 36 cases, every percentage below has a wide margin of error.

**Reproduce.**

```bash
python scripts/score_historian_dev.py --json historian_dev.json
python scripts/score_historian_dev.py --ablate temporal
python scripts/score_historian_dev.py --ablate provenance
```

## 1. Summary

| Measure | Result |
|---|---|
| Temporal-relation accuracy | 91.7% (33/36) |
| Conflict-type accuracy | 86.1% (31/36) |
| Both labels right | 83.3% (30/36) |
| Historian exceptions | 0 |

The 6 wrong cases fall into three groups:

| Group | Cases | Kind | Fix |
|---|---|---|---|
| A. Missing time fallback | 2 | Historian defect | Code fix (Sprint 4) |
| B. Sources disagreeing are not detected | 3 | Known v1 limitation | Team decision on the rule |
| C. Gold label is inconsistent with the guide or another case | 3 (overlaps A and B) | Label issue | Relabel after a second annotator agrees |

SPEC.md section 12 error types: all 6 are **role errors** (Historian) or **label issues**. There
are no retrieval errors: the related claim was found in every case.

## 2. The 6 cases

### A. Missing time fallback (Historian defect)

**`krishna/krishna-unreliable-newer-01`**: gold `NEWER_UPDATE` / `TEMPORAL_UPDATE`, predicted
`OVERLAPPING_VALIDITY` / `VALUE_CONFLICT`.
**`krishna/krishna-poisoning-01`**: gold `NEWER_UPDATE` / `VALUE_CONFLICT`, predicted
`OVERLAPPING_VALIDITY` / `VALUE_CONFLICT`.

*What happens.* The claim in memory has a validity start (Jan 1), but the candidate has **no**
validity dates. The Historian only orders claims by observation time when *neither* side has
validity dates. Here one side does, so it skips that step. It cannot tell that the candidate
starts later, and falls through to "overlapping".

*Example.* Memory: "gym opens at 06:00", valid from Jan 1. Candidate: "opens at 09:00", observed
Oct 8, no dates. A person reads this as newer information. The Historian reads it as two claims
for the same period.

*Fix.* When the candidate has no validity window, fall back to observation time
(`observed_at` vs the claim's `updated_at`), as the unbounded branch already does. Both cases
then get `NEWER_UPDATE`.

### B. Sources that disagree are not detected (known limitation)

**`source-similar-01`**: gold `SOURCE_CONFLICT`, predicted `TEMPORAL_UPDATE`.
**`shailen/shailen-source-similar-01`**: gold `SOURCE_CONFLICT`, predicted `TEMPORAL_UPDATE`.
**`mixed-01`**: gold `VALUE_CONFLICT` (open, see C), predicted `TEMPORAL_UPDATE`.

*What happens.* Two evidence records of similar reliability say different things ("Dept site A:
closes 18:00" vs "Dept site B: closes 17:00"). The deterministic Historian looks at dates and
reliability scores, never at what the evidence says, so it never outputs `SOURCE_CONFLICT`.
This is the documented v1 limitation in `deterministic.py`.

*Why it matters.* The guide's action for this pattern is `QUARANTINE`. Labeling it
`TEMPORAL_UPDATE` points the council toward `REVISE`, which would overwrite a fact while its own
sources disagree.

*Options for the team.*
1. A simple rule: two or more evidence records within 0.1 reliability of each other, where one
   mentions the candidate's value and another mentions the existing claim's value. This catches
   all three dev cases ("18:00"/"17:00", "$150"/"$120"), but matching value strings is brittle.
2. Leave it to the LLM Historian layer, and document the deterministic Historian's gap.

### C. Gold-label issues (need a second annotator)

**`supported-paraphrase-02`**: gold `PARAPHRASE`, predicted `DUPLICATE`.
The candidate's `claim_text` is word-for-word the claim in memory. Only the *evidence* is
reworded ("six PM"). The guide defines `DUPLICATE` as same value, same wording. **The Historian
looks right; propose relabeling to `DUPLICATE`.**

**`shailen/shailen-source-similar-01`**, temporal label: gold `OVERLAPPING_VALIDITY`.
`source-similar-01` has the same shape (a candidate valid from a later date than an open-ended
claim) and is labeled `NEWER_UPDATE`. One of the two should change. **Propose `NEWER_UPDATE`**
so both cases follow the same rule. The conflict label stays
`SOURCE_CONFLICT`.

**`krishna/krishna-poisoning-01`**, conflict label: gold `VALUE_CONFLICT`.
The guide says a newer, different value is `TEMPORAL_UPDATE`, and that "a weak or unreliable
source does not change this label; it changes the action." `krishna-unreliable-newer-01`
follows that rule and this case does not. **Propose `TEMPORAL_UPDATE`.** The final action
`REJECT` is unchanged: the official policy refutes the claim.

**`mixed-01`**: already open in the guide's changelog (two sources of equal standing labeled
`VALUE_CONFLICT`; by definition it may be `SOURCE_CONFLICT`). The category
(`overlapping_temporal_conflict`) also does not match its temporal label (`NEWER_UPDATE`).

*If* the team accepts fix A and the three relabels above, the projected score is 33/36 (91.7%)
for both labels, and every remaining error is a `SOURCE_CONFLICT` case (group B). This is a
projection, not a measured run.

## 3. Ablations (SPEC.md section 12.2)

`AblatedHistorian` (`src/sekf/historian/ablation.py`) hides fields from the Historian only;
the rest of the council still sees the full request.

| Historian sees | Both labels right |
|---|---|
| Everything | 83.3% (30/36) |
| No temporal fields | 44.4% (16/36) |
| No provenance fields | 83.3% (30/36) |

**Without temporal fields**, every case with a related claim becomes `SAME_PERIOD`. The
categories that depend on time drop to 0: reliable newer correction (5/5 → 0/5), unreliable
newer contradiction (3/4 → 0/4), non-overlapping validity (2/2 → 0/2), overlapping conflict
(2/3 → 0/3). Categories with no related claim are unaffected. **Time information is what makes
the Historian's temporal and conflict labels work.**

**Without provenance fields**, the temporal and conflict labels do not change, because they are
computed from dates and values only. The effect is on `provenance_quality`: 27 of 36 cases
change (23 HIGH → LOW, 4 MEDIUM → LOW). Because the policy uses provenance to choose between
`PROMOTE` and `QUARANTINE`, the real cost of this ablation shows in **final actions**, which
needs the experiment runner (pending).

*Caveat.* Hidden reliability is set to a neutral 0.5, which falls in the LOW band (below 0.6).
So "no provenance" behaves like "treat every source as weak". A different neutral value would
give a different result; report this choice with the numbers.

## 4. Follow-ups

| # | Task | Owner | Sprint |
|---|---|---|---|
| 1 | Time fallback when the candidate has no validity window (group A) | Mahalakshmi | 4 |
| 2 | Decide the `SOURCE_CONFLICT` rule (group B) | Team | 4 |
| 3 | Second-annotator check on the 3 relabels + `mixed-01` (group C) | Krishna, Shailen, Sai Chetan | 3 |
| 4 | Score final actions with and without provenance fields | Mahalakshmi, after the runner merges | 3 |
| 5 | List group B in the Historian's known-limitations doc | Mahalakshmi | 4 |
