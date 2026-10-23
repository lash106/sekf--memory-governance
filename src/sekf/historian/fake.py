"""Deterministic fake Historian for the skeleton and offline tests (no LLM)."""

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.contracts import (
    ConflictType,
    HistorianAssessment,
    HistorianRequest,
    ProvenanceQuality,
    TemporalRelation,
)


class FakeHistorian:
    """Trivial rules: no related claim -> new; same object -> duplicate; newer -> update."""

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        cand = request.candidate
        reliability = max((e.reliability for e in request.evidence), default=0.0)
        related = request.related_claims
        base = {
            "assessment_id": uuid5(
                NAMESPACE_URL,
                "historian:"
                + ":".join(
                    [str(cand.claim_id)]
                    + [str(e.evidence_id) + e.content_hash for e in request.evidence]
                    + [str(c.claim_id) + c.updated_at.isoformat() for c in related]
                ),
            ),
            "candidate_claim_id": cand.claim_id,
            "related_claim_ids": [c.claim_id for c in related],
            "provenance_quality": (
                ProvenanceQuality.HIGH if reliability >= 0.8 else ProvenanceQuality.LOW
            ),
            "source_reliability_score": reliability,
            "confidence": 0.9,
            "model": "fake/historian",
            "provider": "fake",
            "prompt_version": "fake-historian-v0",
            "prompt_hash": sha256(b"fake-historian-v0").hexdigest(),
            "temperature": request.run_config.temperature,
            "seed": request.run_config.seed,
            "llm_calls": 0,
        }
        if not related:
            return HistorianAssessment(
                **base,
                temporal_relation=TemporalRelation.NO_RELATED_CLAIM,
                conflict_type=ConflictType.NONE,
                rationale="No related claim in memory.",
            )
        prior = related[0]
        if prior.object_text.strip().lower() == cand.object_text.strip().lower():
            return HistorianAssessment(
                **base,
                temporal_relation=TemporalRelation.SAME_PERIOD,
                conflict_type=ConflictType.DUPLICATE,
                rationale="Same value as an existing claim.",
            )
        if cand.observed_at > prior.updated_at:
            return HistorianAssessment(
                **base,
                temporal_relation=TemporalRelation.NEWER_UPDATE,
                conflict_type=ConflictType.TEMPORAL_UPDATE,
                supersedes_claim_id=prior.claim_id,
                rationale="Different value observed after the existing claim was last updated.",
            )
        return HistorianAssessment(
            **base,
            temporal_relation=TemporalRelation.OLDER_INFORMATION,
            conflict_type=ConflictType.VALUE_CONFLICT,
            rationale="Different value, not newer than the existing claim.",
        )
