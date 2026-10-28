"""Tests for the deterministic Synthesizer v1 (offline, no LLM)."""

import asyncio
from uuid import UUID, uuid4

import pytest
from tests.conftest import make_candidate, make_context, make_existing, make_historian, make_judge

from sekf.contracts import (
    CandidateClaim,
    ConflictType,
    FinalAction,
    HistorianAssessment,
    JudgeAssessment,
    JudgeLabel,
    ProvenanceQuality,
    SynthesisRequest,
    SynthesizerRecommendation,
    TemporalRelation,
)
from sekf.errors import RoleOutputInvalid
from sekf.policy import PolicyEngine
from sekf.synthesizer import DeterministicSynthesizer


def judge(
    cand: CandidateClaim,
    label: JudgeLabel = JudgeLabel.SUPPORTED,
    confidence: float = 0.9,
    cite: bool = True,
    missing: list[str] | None = None,
) -> JudgeAssessment:
    j = make_judge(cand, label, confidence)
    return j.model_copy(
        update={
            "supporting_evidence_ids": [uuid4()] if cite else [],
            "missing_information": missing or [],
        }
    )


def historian(
    cand: CandidateClaim,
    conflict: ConflictType = ConflictType.NONE,
    supersedes: UUID | None = None,
    provenance: ProvenanceQuality = ProvenanceQuality.HIGH,
    confidence: float = 0.9,
) -> HistorianAssessment:
    temporal = (
        TemporalRelation.NEWER_UPDATE
        if conflict is ConflictType.TEMPORAL_UPDATE
        else TemporalRelation.NO_RELATED_CLAIM
    )
    h = make_historian(
        cand, conflict=conflict, temporal=temporal, supersedes=supersedes, confidence=confidence
    )
    return h.model_copy(update={"provenance_quality": provenance})


def synth(
    cand: CandidateClaim, h: HistorianAssessment, j: JudgeAssessment
) -> SynthesizerRecommendation:
    req = SynthesisRequest(candidate=cand, historian=h, judge=j)
    return asyncio.run(DeterministicSynthesizer().synthesize(req))


# --- base recommendation (roles agree, no concerns) --------------------------------------


@pytest.mark.parametrize(
    ("label", "conflict", "expected"),
    [
        (JudgeLabel.SUPPORTED, ConflictType.NONE, FinalAction.PROMOTE),
        (JudgeLabel.SUPPORTED, ConflictType.DUPLICATE, FinalAction.ATTACH_EVIDENCE),
        (JudgeLabel.SUPPORTED, ConflictType.PARAPHRASE, FinalAction.ATTACH_EVIDENCE),
        (JudgeLabel.SUPPORTED, ConflictType.VALUE_CONFLICT, FinalAction.QUARANTINE),
        (JudgeLabel.SUPPORTED, ConflictType.SOURCE_CONFLICT, FinalAction.QUARANTINE),
        (JudgeLabel.SUPPORTED, ConflictType.UNKNOWN, FinalAction.ESCALATE_FOR_REVIEW),
        (JudgeLabel.REFUTED, ConflictType.NONE, FinalAction.REJECT),
        (JudgeLabel.INSUFFICIENT_EVIDENCE, ConflictType.NONE, FinalAction.KEEP_EPISODIC),
        (JudgeLabel.MIXED_EVIDENCE, ConflictType.NONE, FinalAction.QUARANTINE),
        (JudgeLabel.INVALID_INPUT, ConflictType.NONE, FinalAction.ESCALATE_FOR_REVIEW),
    ],
)
def test_base_recommendation(
    label: JudgeLabel, conflict: ConflictType, expected: FinalAction
) -> None:
    cand = make_candidate()
    rec = synth(cand, historian(cand, conflict), judge(cand, label))
    assert rec.recommended_action is expected
    assert rec.llm_calls == 0 and rec.provider == "deterministic"


def test_clean_revise_has_no_questions() -> None:
    cand = make_candidate()
    h = historian(cand, ConflictType.TEMPORAL_UPDATE, supersedes=uuid4())
    rec = synth(cand, h, judge(cand))
    assert rec.recommended_action is FinalAction.REVISE
    assert rec.confidence == 0.9
    assert rec.unresolved_questions == []


# --- strong concerns block a mutation ----------------------------------------------------


def test_s1_supported_without_cited_evidence_blocks() -> None:
    cand = make_candidate()
    rec = synth(cand, historian(cand), judge(cand, cite=False))
    assert rec.recommended_action is FinalAction.QUARANTINE
    assert rec.confidence >= 0.80
    assert any(q.startswith("S1") for q in rec.unresolved_questions)


def test_s2_revise_without_superseded_claim_blocks() -> None:
    cand = make_candidate()
    rec = synth(cand, historian(cand, ConflictType.TEMPORAL_UPDATE), judge(cand))
    assert rec.recommended_action is FinalAction.QUARANTINE
    assert any(q.startswith("S2") for q in rec.unresolved_questions)


def test_s3_uncertain_extraction_blocks() -> None:
    cand = make_candidate().model_copy(update={"extractor_confidence": 0.3})
    rec = synth(cand, historian(cand), judge(cand))
    assert rec.recommended_action is FinalAction.QUARANTINE
    assert any(q.startswith("S3") for q in rec.unresolved_questions)


@pytest.mark.parametrize("quality", [ProvenanceQuality.LOW, ProvenanceQuality.UNKNOWN])
def test_s4_weak_provenance_blocks_mutation(quality: ProvenanceQuality) -> None:
    cand = make_candidate()
    rec = synth(cand, historian(cand, provenance=quality), judge(cand))
    assert rec.recommended_action is FinalAction.QUARANTINE
    assert any(q.startswith("S4") for q in rec.unresolved_questions)


def test_strong_concern_does_not_change_non_mutating_action() -> None:
    cand = make_candidate().model_copy(update={"extractor_confidence": 0.3})
    rec = synth(cand, historian(cand), judge(cand, JudgeLabel.REFUTED))
    assert rec.recommended_action is FinalAction.REJECT


# --- mild concerns are recorded but cannot veto ------------------------------------------


def test_mild_concerns_keep_action_below_veto_threshold() -> None:
    cand = make_candidate()
    rec = synth(cand, historian(cand, confidence=0.6), judge(cand, missing=["closing date"]))
    assert rec.recommended_action is FinalAction.PROMOTE
    assert rec.confidence < 0.80
    assert sum(q.startswith("M") for q in rec.unresolved_questions) == 2


# --- input validation and determinism ----------------------------------------------------


def test_rejects_assessments_for_another_candidate() -> None:
    cand, other = make_candidate(), make_candidate()
    with pytest.raises(RoleOutputInvalid, match="Historian"):
        synth(cand, historian(other), judge(cand))
    with pytest.raises(RoleOutputInvalid, match="Judge"):
        synth(cand, historian(cand), judge(other))


def test_recommendation_id_is_deterministic() -> None:
    cand = make_candidate()
    h, j = historian(cand), judge(cand)
    assert synth(cand, h, j).recommendation_id == synth(cand, h, j).recommendation_id


# --- end to end with the real policy engine ----------------------------------------------


def test_policy_follows_clean_revise() -> None:
    cand, old = make_candidate(), make_existing()
    h = historian(cand, ConflictType.TEMPORAL_UPDATE, supersedes=old.claim_id)
    h = h.model_copy(update={"related_claim_ids": [old.claim_id]})
    j = judge(cand)
    ctx = make_context(
        candidate=cand, related_claims=[old], historian=h, judge=j, synthesizer=synth(cand, h, j)
    )
    decision = PolicyEngine().decide(ctx)
    assert decision.final_action is FinalAction.REVISE
    assert "R_SYN_VETO" not in decision.rule_ids_fired


def test_policy_vetoes_promotion_without_cited_evidence() -> None:
    cand = make_candidate()
    h, j = historian(cand), judge(cand, cite=False)
    ctx = make_context(candidate=cand, historian=h, judge=j, synthesizer=synth(cand, h, j))
    decision = PolicyEngine().decide(ctx)
    assert decision.rule_action is FinalAction.PROMOTE
    assert decision.final_action is FinalAction.QUARANTINE
    assert "R_SYN_VETO" in decision.rule_ids_fired
