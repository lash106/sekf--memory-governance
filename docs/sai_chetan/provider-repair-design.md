# Provider retry / repair design

Owner: Sai Chetan Muppalla
Implements SPEC.md section 11. See also `docs/decisions/0002-llm-provider-d1.md`.

## Goals

- Strict Pydantic validation of structured outputs.
- At most **two repair attempts** after the initial call (≤3 total provider calls).
- Failures become typed errors (`RoleOutputInvalid`, `ProviderUnavailable`); policy rule R1 escalates.
- Never silently switch model, prompt, or evidence.
- Failed attempts still report tokens, latency, call count, and cost. They are not free.

## Algorithm (`OpenAICompatibleProvider.complete_json`)

1. Call the chat-completions endpoint with `response_format={"type":"json_object"}` (or provider equivalent).
2. Parse the assistant content as JSON; validate with `response_model.model_validate`.
3. On success: return `LLMResult` with `llm_calls = attempts`, summed tokens/latency/cost.
4. On `ValidationError` or JSON decode error, if attempts remain:
   - Append a short repair user message listing the validation errors.
   - Retry with the same model, temperature, and seed.
5. After repairs are exhausted: raise `RoleOutputInvalid` with `usage` set to the summed attempts.
6. On network/HTTP/auth failures: raise `ProviderUnavailable` (no repair — the provider is down). If earlier attempts in the same call already spent tokens, that usage is attached too.

## Cost rates

Prices live in `MODEL_USD_PER_MTOK`, version `COST_RATE_VERSION` (`openai-public-2026-10`). `gpt-4o-mini` is priced. Any other model records `estimated_cost_usd=0` unless the process sets both `SEKF_LLM_INPUT_COST_PER_MTOK` and `SEKF_LLM_OUTPUT_COST_PER_MTOK`. Those overrides apply to the model actually called. The mini rate is never reused for a different model.

## What is not repaired

- Timeouts at the role/`asyncio.wait_for` layer (`RoleTimeout`).
- Missing API key / misconfiguration (`ProviderUnavailable`).
- Hallucinated evidence IDs (Judge post-check raises `RoleOutputInvalid` without another LLM call).

## Offline testing

`FakeProvider` remains single-shot. Repair behavior is covered by
`tests/unit/test_openai_compatible_provider.py` with `httpx.MockTransport` — no live network in CI.
