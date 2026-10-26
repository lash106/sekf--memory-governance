"""Private LLM response schemas for the Judge role.

Not part of the public `sekf.contracts` surface. Evolve these when the prompt changes;
`JudgeAssessment` remains the stable outward contract.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sekf.contracts import JudgeLabel


class JudgeLLMOutput(BaseModel):
    """Structured model output before CallMetadata is attached by `LLMJudge`."""

    model_config = ConfigDict(extra="forbid")

    label: JudgeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[UUID] = Field(default_factory=list)
    refuting_evidence_ids: list[UUID] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rationale: str
