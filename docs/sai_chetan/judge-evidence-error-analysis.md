# Judge and evidence error analysis

Owner: Sai Chetan Muppalla. SPEC.md Sprint 3, task S3.3, and the Judge half of section 14.

## What this scores

`sekf.judge.error_analysis` compares a gold Judge label with one prediction. It is deterministic and offline. A case may have several findings:

| Finding | Meaning |
|---|---|
| `WRONG_LABEL` | Predicted label is not the gold label. |
| `OVERCONFIDENT_WRONG_LABEL` | Wrong label and confidence is at least 0.80. |
| `UNDERCONFIDENT_DIRECTIONAL_LABEL` | `SUPPORTED` or `REFUTED` matches gold, but confidence is below 0.80. |
| `MISSING_SUPPORTING_CITATION` | Predicted `SUPPORTED` or `MIXED_EVIDENCE` cites no supporting id. |
| `MISSING_REFUTING_CITATION` | Predicted `REFUTED` or `MIXED_EVIDENCE` cites no refuting id. |
| `UNKNOWN_EVIDENCE_ID` | A cited id was not in the evidence given to the Judge. |
| `DUPLICATE_CITATION` | The same id appears twice in one citation list. |
| `CITED_AS_BOTH` | The same id is listed as both supporting and refuting. |

`LLMJudge` already rejects unknown ids, duplicate ids, and dual citations before an assessment is returned. The analyzer still records those findings so a saved raw prediction can be scored without another model call.

Confidence is one number on the chosen label. Brier score and expected calibration error treat it as the probability that the predicted label is the gold label. The default threshold 0.80 is the published support and refute threshold. It is an analysis parameter. This module does not change policy and does not write memory.

Label accuracy and contradiction F1 (gold `REFUTED` as the positive class) come from `sekf.judge.metrics`.

## How to run

```bash
python scripts/analyze_judge_evidence.py observations.jsonl
```

Each JSONL line needs `case_id`, `category`, `gold_label`, `predicted_label`, `confidence`, `supporting_evidence_ids`, `refuting_evidence_ids`, and `provided_evidence_ids`. The script prints counts, rates, per-category counts, per-gold-label counts, calibration, and the case list. It does not call a provider.

## What this does not assign

Extraction, retrieval, policy, and storage errors need outputs this module does not own. A wrong final action with a correct Judge label is not a Judge error. Those handoffs are listed in `part1-remaining-dependencies.md`.
