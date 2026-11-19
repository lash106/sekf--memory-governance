# Part-one work that still needs another owner

Owner: Sai Chetan Muppalla. These items sit on the SPEC.md section 18 plan through Sprint 5. They are not implemented in the Judge, provider, or Sai Chetan baseline folders because each one needs another module or a second person.

| Blocked item | Who has to move first | Why it is not in this change |
|---|---|---|
| Grow the shared set toward 120–200 cases and freeze `test` | A second annotator, then Shailen for the split | Every `test` case needs two independent labels. One person cannot freeze that split. |
| Dataset loader, experiment manifest, resumable runner | Shailen (`src/sekf/evaluation/`) | The package is his. The Judge scorer reads a JSONL file he can emit later. |
| Required ablations and paired bootstrap intervals | Shailen | Ablations turn council modules off. Bootstrap belongs on the saved pilot rows. |
| First official held-out run | Shailen, after the frozen `test` split | Prompts above are frozen so that run does not tune them. The run itself is not started here. |
| Storage and state error analysis | Krishna | Needs memory deltas and repository failures. |
| Provenance, time, and conflict error analysis | Mahalakshmi | Needs Historian labels. A correct Judge label can still be the wrong final action. |
| Policy, veto, and cost error analysis | Shailen | Rule ids and the veto live in policy. |
| `direct_storage`, `latest_write_wins`, `static_rag` | Krishna | Different baselines. The dev pilot still notes where those packages were missing. |
| Docker image, pinned clean-environment boot, tag `v0.1.0` | Shailen, with the team | Sprint 4 packaging and the tag are shared. |
| Demo rehearsal of promote, revise, and quarantine | All four members | Section 19 needs Historian, Judge, Synthesizer, policy, and memory on one running server. |
| Live `single_judge` row in the pilot | Whoever holds the API budget | The adapter is implemented. `.env` must supply the key. No key is committed. |
| Second model, extra seeds, local model | Semester 2 | SPEC.md section 18 stops part one at the December demonstration. |
