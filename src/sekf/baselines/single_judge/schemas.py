"""Private LLM response schemas for the single-judge baseline.

Kept separate from public contracts so recommended fields can grow (for example cited
evidence) without an ADR until the baseline API is stabilized.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sekf.contracts import FinalAction


class SingleJudgeLLMOutput(BaseModel):
    """Structured model output for one-pass memory-action recommendation."""

    model_config = ConfigDict(extra="forbid")

    recommended_action: FinalAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    # ACTIVE claim that REVISE supersedes or ATTACH_EVIDENCE updates.
    # Omitted only when exactly one ACTIVE related claim exists.
    target_claim_id: UUID | None = None
