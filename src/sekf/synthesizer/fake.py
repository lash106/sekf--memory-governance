"""Deterministic fake Synthesizer for the skeleton and offline tests (no LLM)."""

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.contracts import (
    ConflictType,
    FinalAction,
    JudgeLabel,
    SynthesisRequest,
    SynthesizerRecommendation,
)


def _recommend(request: SynthesisRequest) -> FinalAction:
    label = request.judge.label
    if label is JudgeLabel.REFUTED:
        return FinalAction.REJECT
    if label is JudgeLabel.INSUFFICIENT_EVIDENCE:
        return FinalAction.KEEP_EPISODIC
    if label is JudgeLabel.MIXED_EVIDENCE:
        return FinalAction.QUARANTINE
    if label is not JudgeLabel.SUPPORTED:
        return FinalAction.ESCALATE_FOR_REVIEW
    conflict = request.historian.conflict_type
    if conflict is ConflictType.NONE:
        return FinalAction.PROMOTE
    if conflict in (ConflictType.DUPLICATE, ConflictType.PARAPHRASE):
        return FinalAction.ATTACH_EVIDENCE
    if conflict is ConflictType.TEMPORAL_UPDATE:
        return FinalAction.REVISE
    return FinalAction.QUARANTINE


class FakeSynthesizer:
    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation:
        cand = request.candidate
        action = _recommend(request)
        return SynthesizerRecommendation(
            recommendation_id=uuid5(
                NAMESPACE_URL,
                f"synthesizer:{request.historian.assessment_id}:{request.judge.assessment_id}",
            ),
            candidate_claim_id=cand.claim_id,
            recommended_action=action,
            confidence=0.9,
            historian_assessment_id=request.historian.assessment_id,
            judge_assessment_id=request.judge.assessment_id,
            rationale=f"Fake mapping of judge={request.judge.label}, "
            f"conflict={request.historian.conflict_type}.",
            model="fake/synthesizer",
            provider="fake",
            prompt_version="fake-synthesizer-v0",
            prompt_hash=sha256(b"fake-synthesizer-v0").hexdigest(),
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )
