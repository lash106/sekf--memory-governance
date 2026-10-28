# ADR 0004: Baseline audit record on DecideResponse

- Status: accepted
- Date: 2026-10-27
- Owner: Sai Chetan Muppalla (single_judge). Shailen owns `DecideResponse`; the field is optional and default-null, so the council response is unchanged.
- Contract version: v1, additive.

## Context

SPEC.md section 15 requires every experiment run to record the provider, the exact model id, prompt version and hash, temperature, and seed. `single_judge` kept only the parsed action and a cost summary. `RunConfig.model` is not that record: `fake/none` resolves to the provider's default model, and the response did not keep the prompt hash, the observed temperature and seed, the recommendation confidence, or the validated payload.

## Decision

1. `DecideResponse.baseline_audit` is an optional `BaselineAudit`.
2. `BaselineAudit` carries call metadata plus `baseline_name`, `recommended_action`, `confidence`, `rationale`, `target_claim_id`, and `raw_output`.
3. `single_judge` fills it from the provider result (`model`, `provider`, `temperature`, `seed`) and the loaded prompt (`prompt_version`, `prompt_hash`).
4. The council leaves the field null. Its audit stays on the Historian, Judge, and Synthesizer outputs.
5. A call that fails before the provider returns a result has `baseline_audit` null. Usage that the error does carry still lands on `RoleFailure` and `DecisionRecord.cost`.

## Consequences

- Regenerated `contracts/v1/BaselineAudit.json` and `DecideResponse.json`.
- Existing responses without the field still validate.
- Comparing models in the experiment uses `baseline_audit.model`, not `RunConfig.model`.
