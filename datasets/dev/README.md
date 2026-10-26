# Dev evaluation cases

Labeled cases under this folder are used for prompt/threshold work on `dev` only.

- Contributor cases from Sai Chetan live as `*.json` here (template: `datasets/CASE_TEMPLATE.json`).
- Regenerate Sai Chetan’s set with: `python scripts/generate_sai_chetan_cases.py`
- Do not tune against a frozen `test/` split.

Gold final actions should be re-checked against `datasets/ANNOTATION_GUIDE.md` once that guide is published (Mahalakshmi).
