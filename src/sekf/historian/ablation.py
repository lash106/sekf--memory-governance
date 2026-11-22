"""Ablation switches for the Historian (SPEC.md section 12.2). Owner: Mahalakshmi.

Two of the required ablations are "full system without provenance fields" and "full system
without temporal fields". ``AblatedHistorian`` wraps any Historian and hides those fields
from the request *before* the Historian sees it. Everything else in the council (Judge,
Synthesizer, policy) still gets the full request, so the ablation measures what the
Historian's provenance or temporal analysis adds (RQ3).

Fields are masked, not removed, because the contracts require them:

- provenance off: every source becomes ``"unknown"``, every evidence reliability becomes
  the same neutral value, and ``published_at`` is dropped. The Historian can no longer tell
  a strong source from a weak one.
- temporal off: validity windows are dropped and every timestamp is set to one fixed
  instant. The Historian can no longer tell newer from older, or one period from another.

Claim and evidence IDs are never changed, so citation checks still work.

Example::

    historian = AblatedHistorian(DeterministicHistorian(), drop_temporal=True)
"""

from datetime import UTC, datetime

from sekf.contracts import (
    CandidateClaim,
    EvidenceRecord,
    ExistingClaim,
    HistorianAssessment,
    HistorianRequest,
)
from sekf.historian.interface import Historian

UNKNOWN_SOURCE = "unknown"
NEUTRAL_RELIABILITY = 0.5
FIXED_TIME = datetime(2000, 1, 1, tzinfo=UTC)

ABLATIONS = ("none", "provenance", "temporal")


def mask_provenance(request: HistorianRequest) -> HistorianRequest:
    """Return a copy of ``request`` with source identity and reliability hidden."""
    candidate: CandidateClaim = request.candidate.model_copy(
        update={"source_id": UNKNOWN_SOURCE, "extractor_confidence": 1.0}
    )
    evidence: list[EvidenceRecord] = [
        e.model_copy(
            update={
                "source_id": UNKNOWN_SOURCE,
                "source_type": UNKNOWN_SOURCE,
                "reliability": NEUTRAL_RELIABILITY,
                "published_at": None,
            }
        )
        for e in request.evidence
    ]
    return request.model_copy(update={"candidate": candidate, "evidence": evidence})


def mask_temporal(request: HistorianRequest) -> HistorianRequest:
    """Return a copy of ``request`` with validity windows and timestamps hidden."""
    candidate: CandidateClaim = request.candidate.model_copy(
        update={"valid_from": None, "valid_to": None, "observed_at": FIXED_TIME}
    )
    related: list[ExistingClaim] = [
        c.model_copy(
            update={
                "valid_from": None,
                "valid_to": None,
                "created_at": FIXED_TIME,
                "updated_at": FIXED_TIME,
            }
        )
        for c in request.related_claims
    ]
    evidence: list[EvidenceRecord] = [
        e.model_copy(update={"published_at": None, "retrieved_at": FIXED_TIME})
        for e in request.evidence
    ]
    return request.model_copy(
        update={"candidate": candidate, "related_claims": related, "evidence": evidence}
    )


class AblatedHistorian:
    """Wrap a Historian and hide provenance and/or temporal fields from it."""

    def __init__(
        self,
        inner: Historian,
        *,
        drop_provenance: bool = False,
        drop_temporal: bool = False,
    ) -> None:
        self._inner = inner
        self.drop_provenance = drop_provenance
        self.drop_temporal = drop_temporal

    @classmethod
    def from_name(cls, inner: Historian, ablation: str) -> Historian:
        """Build from a name in ``ABLATIONS``; ``"none"`` returns ``inner`` unchanged."""
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {ABLATIONS}")
        if ablation == "none":
            return inner
        return cls(
            inner,
            drop_provenance=ablation == "provenance",
            drop_temporal=ablation == "temporal",
        )

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        masked = request
        if self.drop_provenance:
            masked = mask_provenance(masked)
        if self.drop_temporal:
            masked = mask_temporal(masked)
        return await self._inner.analyze(masked)
