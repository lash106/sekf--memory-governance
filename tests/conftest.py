import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from sekf.api import Governance
from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    ConflictType,
    DecideRequest,
    ExistingClaim,
    FinalAction,
    HistorianAssessment,
    JudgeAssessment,
    JudgeLabel,
    PolicyContext,
    ProvenanceQuality,
    SynthesizerRecommendation,
    TemporalRelation,
)
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 21, tzinfo=UTC)
META: dict[str, Any] = {
    "model": "test",
    "provider": "test",
    "prompt_version": "test",
    "prompt_hash": "test-hash",
    "temperature": 0.0,
    "llm_calls": 0,
}


def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def make_candidate() -> CandidateClaim:
    return CandidateClaim(
        claim_id=uuid4(),
        claim_text="The lab closes at 18:00 on Fridays.",
        subject_key="laboratory",
        predicate_key="closing_time_friday",
        object_text="18:00",
        source_id="s1",
        observed_at=NOW,
    )


def make_existing(status: ClaimStatus = ClaimStatus.ACTIVE) -> ExistingClaim:
    return ExistingClaim(
        claim_id=uuid4(),
        claim_text="The lab closes at 17:00 on Fridays.",
        subject_key="laboratory",
        predicate_key="closing_time_friday",
        object_text="17:00",
        status=status,
        version=1,
        confidence=0.9,
        created_at=NOW,
        updated_at=NOW,
    )


def make_historian(
    cand: CandidateClaim,
    *,
    conflict: ConflictType = ConflictType.NONE,
    temporal: TemporalRelation = TemporalRelation.NO_RELATED_CLAIM,
    related: list[UUID] | None = None,
    supersedes: UUID | None = None,
    reliability: float = 0.95,
    confidence: float = 0.9,
) -> HistorianAssessment:
    return HistorianAssessment(
        assessment_id=uuid4(),
        candidate_claim_id=cand.claim_id,
        related_claim_ids=related or [],
        provenance_quality=ProvenanceQuality.HIGH,
        temporal_relation=temporal,
        conflict_type=conflict,
        supersedes_claim_id=supersedes,
        source_reliability_score=reliability,
        confidence=confidence,
        rationale="test",
        **META,
    )


def make_judge(
    cand: CandidateClaim, label: JudgeLabel = JudgeLabel.SUPPORTED, confidence: float = 0.9
) -> JudgeAssessment:
    return JudgeAssessment(
        assessment_id=uuid4(),
        candidate_claim_id=cand.claim_id,
        label=label,
        confidence=confidence,
        rationale="test",
        **META,
    )


def make_synth(
    cand: CandidateClaim,
    h: HistorianAssessment,
    j: JudgeAssessment,
    action: FinalAction,
    confidence: float = 0.9,
) -> SynthesizerRecommendation:
    return SynthesizerRecommendation(
        recommendation_id=uuid4(),
        candidate_claim_id=cand.claim_id,
        recommended_action=action,
        confidence=confidence,
        historian_assessment_id=h.assessment_id,
        judge_assessment_id=j.assessment_id,
        rationale="test",
        **META,
    )


def make_context(**overrides: Any) -> PolicyContext:
    cand = overrides.pop("candidate", None) or make_candidate()
    return PolicyContext(candidate=cand, now=NOW, **overrides)


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def lab_fixture() -> dict[str, Any]:
    return load_fixture("lab_closing_revise.json")


@pytest.fixture
def repo(lab_fixture: dict[str, Any]) -> InMemoryRepository:
    return InMemoryRepository(
        [ExistingClaim.model_validate(c) for c in lab_fixture["existing_claims"]]
    )


@pytest.fixture
def governance(repo: InMemoryRepository) -> Governance:
    return Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )


@pytest.fixture
def decide_request(lab_fixture: dict[str, Any]) -> DecideRequest:
    return DecideRequest.model_validate(lab_fixture["request"])
