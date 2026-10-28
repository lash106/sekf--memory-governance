"""Provider-neutral structured-output LLM interface (SPEC.md section 11).

Owner: Sai Chetan. The Protocol and fake provider here are the skeleton; the real
OpenAI-compatible adapter (repair attempts, typed errors, cost) is added in Sprint 1.
"""

from typing import Protocol

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    """A validated model response plus call bookkeeping."""

    parsed: BaseModel
    model: str
    provider: str
    # Real calls made for this result, INCLUDING repair attempts. Roles copy this into
    # `CallMetadata.llm_calls`; they cannot know it otherwise. Fake providers report 0.
    llm_calls: int = Field(ge=0)
    # Token, latency, and cost totals are summed across all attempts.
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    # Settings this call actually used. Callers copy these into audit records.
    # Defaults keep older test doubles valid; live and fake providers set both.
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int | None = None


class StructuredLLMProvider(Protocol):
    """All LLM calls go through this interface. Output must validate against `response_model`."""

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResult: ...
