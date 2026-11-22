"""Tests for the Historian ablation switches (SPEC.md section 12.2)."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    ConflictType,
    EvidenceRecord,
    ExistingClaim,
    HistorianAssessment,
    HistorianRequest,
    ProvenanceQuality,
    TemporalRelation,
)
from sekf.historian import ABLATIONS, AblatedHistorian, DeterministicHistorian
from sekf.historian.ablation import (
    FIXED_TIME,
    NEUTRAL_RELIABILITY,
    UNKNOWN_SOURCE,
    mask_provenance,
    mask_temporal,
)


def d(month: int, day: int = 1) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC)


def update_request(reliability: float = 0.95) -> HistorianRequest:
    """The SPEC section 4 example: 17:00 since Jan, a reliable 18:00 from Oct."""
    prior = ExistingClaim(
        claim_id=uuid4(),
        claim_text="The lab closes at 17:00.",
        subject_key="lab",
        predicate_key="closing_time",
        object_text="17:00",
        status=ClaimStatus.ACTIVE,
        version=1,
        valid_from=d(1),
        confidence=0.9,
        created_at=d(1),
        updated_at=d(1),
    )
    cand = CandidateClaim(
        claim_id=uuid4(),
        claim_text="The lab closes at 18:00.",
        subject_key="lab",
        predicate_key="closing_time",
        object_text="18:00",
        source_id="registrar",
        observed_at=d(10),
        valid_from=d(10),
    )
    ev = EvidenceRecord(
        evidence_id=uuid4(),
        content="Registrar notice: the lab closes at 18:00 from October 1.",
        source_id="registrar",
        source_type="official",
        published_at=d(9, 28),
        retrieved_at=d(10),
        reliability=reliability,
    )
    return HistorianRequest(candidate=cand, related_claims=[prior], evidence=[ev])


def analyze(historian: object, request: HistorianRequest) -> HistorianAssessment:
    return asyncio.run(historian.analyze(request))  # type: ignore[attr-defined]


def test_mask_provenance_hides_sources_but_keeps_ids_and_times() -> None:
    req = update_request()
    masked = mask_provenance(req)
    assert masked.candidate.source_id == UNKNOWN_SOURCE
    ev = masked.evidence[0]
    assert (ev.source_id, ev.source_type) == (UNKNOWN_SOURCE, UNKNOWN_SOURCE)
    assert ev.reliability == NEUTRAL_RELIABILITY
    assert ev.published_at is None
    assert ev.evidence_id == req.evidence[0].evidence_id
    assert masked.candidate.valid_from == req.candidate.valid_from
    assert masked.related_claims == req.related_claims


def test_mask_temporal_hides_times_but_keeps_ids_and_sources() -> None:
    req = update_request()
    masked = mask_temporal(req)
    cand = masked.candidate
    assert (cand.valid_from, cand.valid_to, cand.observed_at) == (None, None, FIXED_TIME)
    prior = masked.related_claims[0]
    assert (prior.valid_from, prior.valid_to) == (None, None)
    assert prior.updated_at == prior.created_at == FIXED_TIME
    assert prior.claim_id == req.related_claims[0].claim_id
    assert masked.evidence[0].published_at is None
    assert masked.evidence[0].reliability == req.evidence[0].reliability
    assert cand.source_id == req.candidate.source_id


def test_masking_does_not_change_the_original_request() -> None:
    req = update_request()
    mask_provenance(req)
    mask_temporal(req)
    assert req.candidate.valid_from == d(10)
    assert req.evidence[0].reliability == 0.95


def test_full_historian_sees_the_update() -> None:
    out = analyze(DeterministicHistorian(), update_request())
    assert out.temporal_relation is TemporalRelation.NEWER_UPDATE
    assert out.conflict_type is ConflictType.TEMPORAL_UPDATE
    assert out.provenance_quality is ProvenanceQuality.HIGH


def test_without_temporal_fields_the_update_is_missed() -> None:
    req = update_request()
    out = analyze(AblatedHistorian(DeterministicHistorian(), drop_temporal=True), req)
    assert out.temporal_relation is not TemporalRelation.NEWER_UPDATE
    assert out.supersedes_claim_id is None
    assert out.provenance_quality is ProvenanceQuality.HIGH  # provenance untouched


def test_without_provenance_fields_weak_and_strong_sources_look_the_same() -> None:
    hist = AblatedHistorian(DeterministicHistorian(), drop_provenance=True)
    strong = analyze(hist, update_request(reliability=0.95))
    weak = analyze(hist, update_request(reliability=0.10))
    assert strong.provenance_quality == weak.provenance_quality
    assert strong.source_reliability_score == weak.source_reliability_score
    assert strong.temporal_relation is TemporalRelation.NEWER_UPDATE  # time untouched


def test_assessment_still_points_at_the_real_candidate() -> None:
    req = update_request()
    both = AblatedHistorian(DeterministicHistorian(), drop_provenance=True, drop_temporal=True)
    out = analyze(both, req)
    assert out.candidate_claim_id == req.candidate.claim_id
    assert out.related_claim_ids == [req.related_claims[0].claim_id]


def test_from_name() -> None:
    inner = DeterministicHistorian()
    assert AblatedHistorian.from_name(inner, "none") is inner
    temporal = AblatedHistorian.from_name(inner, "temporal")
    assert isinstance(temporal, AblatedHistorian)
    assert temporal.drop_temporal and not temporal.drop_provenance
    provenance = AblatedHistorian.from_name(inner, "provenance")
    assert isinstance(provenance, AblatedHistorian)
    assert provenance.drop_provenance and not provenance.drop_temporal
    assert set(ABLATIONS) == {"none", "provenance", "temporal"}
    with pytest.raises(ValueError):
        AblatedHistorian.from_name(inner, "no-time")
