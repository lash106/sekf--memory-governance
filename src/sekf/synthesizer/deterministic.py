"""Deterministic Synthesizer v1 (SPEC.md section 18, Sprint 1). Owner: Mahalakshmi.

Advisory only. It reads the Historian and Judge outputs, recommends an action, and lists
what is still unresolved. The policy engine makes the final decision; the Synthesizer can
only make it more conservative (veto rule ``R_SYN_VETO``: a *different* recommendation with
confidence >= ``synthesizer_veto_threshold`` turns PROMOTE / REVISE / ATTACH_EVIDENCE into
QUARANTINE).

Why it is not a copy of the policy: if it only repeated the rules, the veto could never fire
and the "without Synthesizer" ablation would measure nothing. So besides a base
recommendation it runs **cross-role checks** that no single role or policy rule makes:

Strong concerns (block a memory change, confidence ``VETO`` >= 0.80):
- S1 Judge says SUPPORTED but cites no supporting evidence.
- S2 the Historian proposes a revision but names no claim to supersede.
- S3 the extracted claim itself is uncertain (``extractor_confidence`` < 0.50).
- S4 provenance is LOW or UNKNOWN for a change to memory.

Mild concerns (keep the base action, lower confidence, record a question):
- M1 Judge lists missing information despite SUPPORTED.
- M2 the Judge or Historian is unsure (confidence below 0.80 / 0.70).

The LLM layer (planned) will add judgement these rules cannot make, e.g. whether the
Historian's rationale and the Judge's rationale describe the same fact.
"""

from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from sekf.contracts import (
    ConflictType,
    FinalAction,
    JudgeLabel,
    ProvenanceQuality,
    SynthesisRequest,
    SynthesizerRecommendation,
)
from sekf.errors import RoleOutputInvalid

PROMPT_VERSION = "deterministic-synthesizer-v1"
PROMPT_HASH = sha256(PROMPT_VERSION.encode()).hexdigest()

VETO = 0.85  # above the default synthesizer_veto_threshold (0.80): strong concerns can veto
UNSURE = 0.75  # below it: mild concerns are recorded but never veto
MIN_EXTRACTOR_CONFIDENCE = 0.50
JUDGE_SURE = 0.80  # mirrors support_threshold
HISTORIAN_SURE = 0.70  # mirrors historian_confidence_threshold

MUTATING = {FinalAction.PROMOTE, FinalAction.REVISE, FinalAction.ATTACH_EVIDENCE}


def base_action(request: SynthesisRequest) -> FinalAction:
    """What the Judge (is it supported?) and the Historian (how does it fit memory?) jointly say."""
    label = request.judge.label
    if label is JudgeLabel.INVALID_INPUT:
        return FinalAction.ESCALATE_FOR_REVIEW
    if label is JudgeLabel.REFUTED:
        return FinalAction.REJECT
    if label is JudgeLabel.INSUFFICIENT_EVIDENCE:
        return FinalAction.KEEP_EPISODIC
    if label is JudgeLabel.MIXED_EVIDENCE:
        return FinalAction.QUARANTINE
    conflict = request.historian.conflict_type
    if conflict is ConflictType.NONE:
        return FinalAction.PROMOTE
    if conflict in (ConflictType.DUPLICATE, ConflictType.PARAPHRASE):
        return FinalAction.ATTACH_EVIDENCE
    if conflict is ConflictType.TEMPORAL_UPDATE:
        return FinalAction.REVISE
    if conflict is ConflictType.UNKNOWN:
        return FinalAction.ESCALATE_FOR_REVIEW
    return FinalAction.QUARANTINE  # VALUE_CONFLICT, SOURCE_CONFLICT


def strong_concerns(request: SynthesisRequest, action: FinalAction) -> list[str]:
    """Cross-role problems serious enough to block a change to memory."""
    h, j, c = request.historian, request.judge, request.candidate
    out: list[str] = []
    if j.label is JudgeLabel.SUPPORTED and not j.supporting_evidence_ids:
        out.append("S1: Judge says SUPPORTED but cites no supporting evidence.")
    if action is FinalAction.REVISE and h.supersedes_claim_id is None:
        out.append("S2: Revision proposed but the Historian names no claim to supersede.")
    if c.extractor_confidence < MIN_EXTRACTOR_CONFIDENCE:
        out.append(
            f"S3: Extracted claim is uncertain (extractor confidence {c.extractor_confidence})."
        )
    if action in MUTATING and h.provenance_quality in (
        ProvenanceQuality.LOW,
        ProvenanceQuality.UNKNOWN,
    ):
        out.append(f"S4: Provenance is {h.provenance_quality.value} for a change to memory.")
    return out


def mild_concerns(request: SynthesisRequest) -> list[str]:
    """Doubts worth recording that should not block the action on their own."""
    h, j = request.historian, request.judge
    out: list[str] = []
    if j.label is JudgeLabel.SUPPORTED and j.missing_information:
        out.append("M1: Judge still lists missing information: " + "; ".join(j.missing_information))
    if j.confidence < JUDGE_SURE:
        out.append(f"M2: Judge is unsure (confidence {j.confidence}).")
    if h.confidence < HISTORIAN_SURE:
        out.append(f"M2: Historian is unsure (confidence {h.confidence}).")
    return out


def validate_inputs(request: SynthesisRequest) -> None:
    """Both role outputs must be about the candidate being synthesized."""
    cid = request.candidate.claim_id
    if request.historian.candidate_claim_id != cid:
        raise RoleOutputInvalid("Historian assessment is for a different candidate")
    if request.judge.candidate_claim_id != cid:
        raise RoleOutputInvalid("Judge assessment is for a different candidate")


class DeterministicSynthesizer:
    """Rule-based Synthesizer. Reports ``llm_calls=0`` and provider ``deterministic``."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation:
        """Recommend an action for ``request.candidate`` from the Historian and Judge outputs.

        Example: lab-closing REVISE (Judge SUPPORTED citing the official schedule, Historian
        TEMPORAL_UPDATE superseding the 17:00 claim, HIGH provenance) -> REVISE, confidence
        0.90, no unresolved questions. The same case with no cited evidence -> QUARANTINE,
        confidence 0.85, which makes the policy veto the revision.
        """
        validate_inputs(request)
        h, j = request.historian, request.judge
        action = base_action(request)
        strong = strong_concerns(request, action)
        mild = mild_concerns(request)

        if strong and action in MUTATING:
            recommended, confidence = FinalAction.QUARANTINE, VETO
            why = f"Base action {action.value} blocked by cross-role checks."
        elif mild:
            recommended, confidence = action, UNSURE
            why = f"Recommends {action.value}, with open doubts."
        else:
            recommended = action
            confidence = round(min(0.90, j.confidence, max(h.confidence, HISTORIAN_SURE)), 2)
            why = f"Judge {j.label.value} and Historian {h.conflict_type.value} agree."

        return SynthesizerRecommendation(
            recommendation_id=self._recommendation_id(h.assessment_id, j.assessment_id),
            candidate_claim_id=request.candidate.claim_id,
            recommended_action=recommended,
            confidence=confidence,
            historian_assessment_id=h.assessment_id,
            judge_assessment_id=j.assessment_id,
            rationale=(
                f"{why} Judge={j.label.value} ({j.confidence}); "
                f"Historian={h.temporal_relation.value}/{h.conflict_type.value} "
                f"({h.confidence}), provenance {h.provenance_quality.value}."
            ),
            unresolved_questions=strong + mild,
            model="deterministic/synthesizer",
            provider="deterministic",
            prompt_version=PROMPT_VERSION,
            prompt_hash=PROMPT_HASH,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )

    @staticmethod
    def _recommendation_id(historian_id: UUID, judge_id: UUID) -> UUID:
        return uuid5(NAMESPACE_URL, f"{PROMPT_VERSION}:{historian_id}:{judge_id}")
