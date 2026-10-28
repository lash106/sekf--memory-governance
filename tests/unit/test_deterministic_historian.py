"""Tests for the deterministic Historian v1 (offline, no LLM)."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

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
from sekf.errors import RoleOutputInvalid
from sekf.historian import DeterministicHistorian, validate_citations

CASES = Path(__file__).resolve().parents[2] / "datasets" / "dev" / "mahalakshmi"


def d(month: int, day: int = 1) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC)


def existing(
    obj: str = "17:00",
    text: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    status: ClaimStatus = ClaimStatus.ACTIVE,
    updated: datetime | None = None,
    subject: str = "lab",
) -> ExistingClaim:
    return ExistingClaim(
        claim_id=uuid4(),
        claim_text=text or f"The lab closes at {obj}.",
        subject_key=subject,
        predicate_key="closing_time",
        object_text=obj,
        status=status,
        version=1,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=0.9,
        created_at=updated or d(1),
        updated_at=updated or d(1),
    )


def candidate(
    obj: str = "18:00",
    text: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    observed: datetime | None = None,
) -> CandidateClaim:
    return CandidateClaim(
        claim_id=uuid4(),
        claim_text=text or f"The lab closes at {obj}.",
        subject_key="lab",
        predicate_key="closing_time",
        object_text=obj,
        source_id="src",
        observed_at=observed or d(10),
        valid_from=valid_from,
        valid_to=valid_to,
    )


def evidence(reliability: float) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=uuid4(),
        content="Some evidence.",
        source_id="src",
        source_type="doc",
        retrieved_at=d(10),
        reliability=reliability,
    )


def analyze(
    cand: CandidateClaim, related: list[ExistingClaim], rel: float | None = 0.95
) -> HistorianAssessment:
    req = HistorianRequest(
        candidate=cand,
        related_claims=related,
        evidence=[] if rel is None else [evidence(rel)],
    )
    return asyncio.run(DeterministicHistorian().analyze(req))


# --- core scenarios ----------------------------------------------------------------------


def test_no_related_claim() -> None:
    a = analyze(candidate(), [])
    assert (a.temporal_relation, a.conflict_type) == (
        TemporalRelation.NO_RELATED_CLAIM,
        ConflictType.NONE,
    )
    assert a.llm_calls == 0 and a.provider == "deterministic"


def test_spec_lab_example_is_newer_update_superseding_old_claim() -> None:
    old = existing("17:00", valid_from=d(1))
    a = analyze(candidate("18:00", valid_from=d(10)), [old])
    assert a.temporal_relation is TemporalRelation.NEWER_UPDATE
    assert a.conflict_type is ConflictType.TEMPORAL_UPDATE
    assert a.supersedes_claim_id == old.claim_id
    assert a.provenance_quality is ProvenanceQuality.HIGH


def test_non_overlapping_periods_do_not_conflict() -> None:
    current = existing("Dr. Rivera", valid_from=d(7))
    a = analyze(candidate("Dr. Okafor", valid_from=d(1), valid_to=d(7)), [current])
    assert a.temporal_relation is TemporalRelation.NON_OVERLAPPING_VALIDITY
    assert a.conflict_type is ConflictType.NONE
    assert a.supersedes_claim_id is None


def test_same_period_different_value_is_overlapping_conflict() -> None:
    old = existing("Nov 14", valid_from=d(8), valid_to=d(12))
    a = analyze(candidate("Nov 7", valid_from=d(8), valid_to=d(12)), [old])
    assert a.temporal_relation is TemporalRelation.OVERLAPPING_VALIDITY
    assert a.conflict_type is ConflictType.VALUE_CONFLICT


def test_candidate_for_earlier_period_is_older_information() -> None:
    old = existing("17:00", valid_from=d(6))
    a = analyze(candidate("16:00", valid_from=d(3)), [old])
    assert a.temporal_relation is TemporalRelation.OLDER_INFORMATION
    assert a.conflict_type is ConflictType.VALUE_CONFLICT


def test_duplicate_and_paraphrase() -> None:
    old = existing("17:00")
    dup = analyze(candidate("17:00"), [old])
    assert dup.conflict_type is ConflictType.DUPLICATE
    para = analyze(candidate(" 17:00 ", text="Lab shuts at 17:00."), [old])
    assert para.conflict_type is ConflictType.PARAPHRASE
    assert para.temporal_relation is TemporalRelation.SAME_PERIOD


@pytest.mark.parametrize(
    ("observed", "relation", "conflict"),
    [
        (d(10), TemporalRelation.NEWER_UPDATE, ConflictType.TEMPORAL_UPDATE),
        (d(1, 1), TemporalRelation.SAME_PERIOD, ConflictType.VALUE_CONFLICT),
    ],
)
def test_no_validity_dates_falls_back_to_observation_time(
    observed: datetime, relation: TemporalRelation, conflict: ConflictType
) -> None:
    old = existing("17:00", updated=d(1))
    a = analyze(candidate("18:00", observed=observed), [old])
    assert (a.temporal_relation, a.conflict_type) == (relation, conflict)


def test_older_observation_without_dates() -> None:
    old = existing("17:00", updated=d(10))
    a = analyze(candidate("18:00", observed=d(2)), [old])
    assert a.temporal_relation is TemporalRelation.OLDER_INFORMATION


def test_two_overlapping_active_claims_is_ambiguous_low_confidence() -> None:
    a = analyze(candidate("18:00"), [existing("17:00"), existing("16:00")])
    assert a.temporal_relation is TemporalRelation.UNKNOWN
    assert a.confidence < 0.70  # policy R6 escalates


def test_superseded_and_other_subject_claims_are_ignored_for_conflict() -> None:
    old = existing("17:00", status=ClaimStatus.SUPERSEDED)
    other = existing("09:00", subject="gym")
    a = analyze(candidate("18:00"), [old, other])
    assert a.temporal_relation is TemporalRelation.NO_RELATED_CLAIM
    assert a.related_claim_ids == [old.claim_id]  # same key, kept as history


def test_provenance_unknown_without_evidence_and_low_for_weak_source() -> None:
    none = analyze(candidate(), [], rel=None)
    assert none.provenance_quality is ProvenanceQuality.UNKNOWN
    assert none.source_reliability_score == 0.0
    weak = analyze(candidate(), [], rel=0.2)
    assert weak.provenance_quality is ProvenanceQuality.LOW


def test_assessment_id_is_deterministic() -> None:
    cand, old = candidate(valid_from=d(10)), existing(valid_from=d(1))
    req = HistorianRequest(candidate=cand, related_claims=[old])
    h = DeterministicHistorian()
    assert asyncio.run(h.analyze(req)).assessment_id == asyncio.run(h.analyze(req)).assessment_id


# --- citation validation -----------------------------------------------------------------


def test_validate_citations_rejects_unknown_and_inactive_claims() -> None:
    old = existing("17:00", valid_from=d(1))
    req = HistorianRequest(candidate=candidate(valid_from=d(10)), related_claims=[old])
    good = asyncio.run(DeterministicHistorian().analyze(req))
    validate_citations(good, req)  # no error

    with pytest.raises(RoleOutputInvalid, match="unknown related"):
        validate_citations(good.model_copy(update={"related_claim_ids": [uuid4()]}), req)
    with pytest.raises(RoleOutputInvalid, match="unknown claim"):
        validate_citations(good.model_copy(update={"supersedes_claim_id": uuid4()}), req)

    retired = old.model_copy(update={"status": ClaimStatus.SUPERSEDED})
    req2 = HistorianRequest(candidate=req.candidate, related_claims=[retired])
    with pytest.raises(RoleOutputInvalid, match="non-active"):
        validate_citations(good, req2)


# --- gold labels from datasets/dev/mahalakshmi -------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("path", sorted(CASES.glob("*.json")), ids=lambda p: p.stem)
def test_matches_gold_historian_labels(path: Path) -> None:
    case = _load(path)
    req = HistorianRequest(
        candidate=CandidateClaim.model_validate(case["candidate"]),
        related_claims=[ExistingClaim.model_validate(c) for c in case["initial_memory"]],
        evidence=[EvidenceRecord.model_validate(e) for e in case["evidence"]],
    )
    a = asyncio.run(DeterministicHistorian().analyze(req))
    gold = case["gold"]
    assert a.temporal_relation.value == gold["temporal_relation"]
    assert a.conflict_type.value == gold["conflict_type"]
    superseded = gold["expected_memory_delta"]["superseded_claim_ids"]
    if superseded:
        assert a.supersedes_claim_id == UUID(superseded[0])
