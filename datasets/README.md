# Datasets

- `CASE_TEMPLATE.json` — one evaluation case (SPEC.md section 13.1). Copy it, fill it in, delete the `_instructions` key.
- `dev/` — cases used to tune prompts and thresholds.
- `test/` — frozen held-out cases, double-labeled; do not tune against them.
- `ANNOTATION_GUIDE.md` — labeling rules (owner: Mahalakshmi).

Sprint 0 task: every member writes 5 `dev` cases. Required categories are listed in SPEC.md section 13.3. The case file format will be validated by a loader in Sprint 2 (Shailen), so keep to the template's field names.

Sprint 0 cases written so far: `dev/` (Sai Chetan), `dev/krishna/`, `dev/mahalakshmi/`, `dev/shailen/` (promote, reject, revise, quarantine on similar-reliability sources, prompt injection).
