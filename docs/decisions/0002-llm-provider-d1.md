# ADR 0002: LLM provider choice (SPEC decision D1)

- Status: accepted for Sprint 1–2 development (Sai Chetan ownership)
- Date: 2026-10-26
- Owner: Sai Chetan Muppalla
- Affected owners: Sai Chetan (providers/Judge); others consume via `StructuredLLMProvider`

## Context

SPEC.md decision D1 (due Oct 13) requires choosing the LLM provider, exact model, budget source, rate limits, and whether to cache dev responses. Tests must stay offline via `FakeProvider`.

## Decision

1. **Protocol:** OpenAI-compatible Chat Completions JSON mode (`/v1/chat/completions`) through `sekf.providers.openai_compatible.OpenAICompatibleProvider`.
2. **Default model id:** `gpt-4o-mini` (low cost for council + baselines). Override with `SEKF_LLM_MODEL`.
3. **Base URL:** `https://api.openai.com/v1` by default, or any compatible endpoint via `SEKF_LLM_BASE_URL` (Azure OpenAI, Together, local gateway, etc.).
4. **Credentials:** `SEKF_LLM_API_KEY` from the environment only (never committed). `SEKF_PROVIDER=fake|openai`.
5. **Budget (team working assumption):** use a shared course/team OpenAI (or compatible) prepaid budget; keep temperature 0 for experiments; prefer `gpt-4o-mini` until Sprint 3 seed/model sweeps. Exact dollar cap is recorded per experiment manifest by Shailen.
6. **Rate limits:** client-side spacing is not implemented in v1; on HTTP 429 raise `ProviderUnavailable` (no silent retry storm). Operators set provider-side RPM limits.
7. **Dev response caching:** not enabled by default (would change scientific reproducibility if opaque). Optional future cache must key on `(model, prompt_hash, messages_hash, temperature, seed)` and record a cache hit in audit metadata.
8. **Repair:** at most **two** repair attempts after the first failed validation (three total LLM calls max per `complete_json`). See `docs/sai_chetan/provider-repair-design.md`.

## Consequences

- CI and unit tests continue to use `FakeProvider` with `SEKF_PROVIDER=fake`.
- Live demos require a key in `.env`; missing key raises `ProviderUnavailable` (no silent fallback to another model).
- Cost-normalized comparisons use `CallMetadata.llm_calls` including repairs.
