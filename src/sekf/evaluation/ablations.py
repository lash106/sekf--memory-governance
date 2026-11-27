"""Council ablations from SPEC.md section 12.2. Owner: Shailen.

Three of the five required ablations live here. The other two ("without provenance fields"
and "without temporal fields") are Mahalakshmi's `sekf.historian.AblatedHistorian`.

- **Without Historian**: `NeutralHistorian` replaces the Historian with a stub that has no
  time or provenance signal. It reports no conflict, an unknown temporal relation, and trusts
  every source fully (HIGH provenance, reliability 1.0), so the policy decides on the Judge
  alone. Provenance is HIGH, not UNKNOWN, on purpose: UNKNOWN would make the Synthesizer veto
  every change (its check S4), and the ablation would then measure that veto, not the
  missing Historian.
  Every Judge-supported claim is promoted, even when it contradicts an active claim. That is
  exactly the behavior the Historian exists to prevent, so the drop measures its value.
- **Without Synthesizer**: `DisabledSynthesizer` returns a recommendation with confidence 0,
  so `R_SYN_VETO` can never fire. Nothing else changes.
- **Without deterministic policy**: `no_policy_action` reads the Synthesizer's recommended
  action from an ordinary dry-run response and uses it as the final action. It is computed
  from a response, never applied, so it cannot write to any store (SPEC.md: dry-run only).

All stubs make no model call and report `llm_calls=0`.
"""

from __future__ import annotations

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.contracts import (
    ConflictType,
    DecideMode,
    DecideResponse,
    FinalAction,
    HistorianAssessment,
    HistorianRequest,
    ProvenanceQuality,
    SynthesisRequest,
    SynthesizerRecommendation,
    TemporalRelation,
)

NEUTRAL_HISTORIAN_VERSION = "ablation-no-historian-v1"
DISABLED_SYNTHESIZER_VERSION = "ablation-no-synthesizer-v1"


def _hash(version: str) -> str:
    return sha256(version.encode()).hexdigest()


class NeutralHistorian:
    """Historian stub for the "without Historian" ablation: no time or provenance signal."""

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        """Report no conflict and full trust, whatever memory and evidence say."""

        cand = request.candidate
        return HistorianAssessment(
            assessment_id=uuid5(NAMESPACE_URL, f"{NEUTRAL_HISTORIAN_VERSION}:{cand.claim_id}"),
            candidate_claim_id=cand.claim_id,
            related_claim_ids=[],
            provenance_quality=ProvenanceQuality.HIGH,
            temporal_relation=TemporalRelation.UNKNOWN,
            conflict_type=ConflictType.NONE,
            supersedes_claim_id=None,
            source_reliability_score=1.0,
            confidence=1.0,
            rationale="Ablation: Historian removed; no time, provenance, or conflict analysis.",
            model="ablation/no-historian",
            provider="ablation",
            prompt_version=NEUTRAL_HISTORIAN_VERSION,
            prompt_hash=_hash(NEUTRAL_HISTORIAN_VERSION),
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )


class DisabledSynthesizer:
    """Synthesizer stub for the "without Synthesizer" ablation: zero confidence, so no veto."""

    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation:
        """Return a zero-confidence recommendation that the policy veto ignores."""

        cand = request.candidate
        return SynthesizerRecommendation(
            recommendation_id=uuid5(
                NAMESPACE_URL, f"{DISABLED_SYNTHESIZER_VERSION}:{cand.claim_id}"
            ),
            candidate_claim_id=cand.claim_id,
            recommended_action=FinalAction.KEEP_EPISODIC,
            confidence=0.0,
            historian_assessment_id=request.historian.assessment_id,
            judge_assessment_id=request.judge.assessment_id,
            rationale="Ablation: Synthesizer removed; confidence 0 so no veto can fire.",
            model="ablation/no-synthesizer",
            provider="ablation",
            prompt_version=DISABLED_SYNTHESIZER_VERSION,
            prompt_hash=_hash(DISABLED_SYNTHESIZER_VERSION),
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )


def no_policy_action(response: DecideResponse) -> FinalAction:
    """Final action for the "without deterministic policy" ablation.

    The Synthesizer's recommendation is used directly. If the Synthesizer produced nothing
    (a role failed), the claim is escalated, which is what any system must do without an
    answer. Only dry-run responses are accepted, so the ablation can never mutate memory.
    """

    if response.mode is not DecideMode.DRY_RUN:
        raise ValueError("the no-policy ablation is dry-run only (SPEC.md section 12.2)")
    if response.synthesizer is None:
        return FinalAction.ESCALATE_FOR_REVIEW
    return response.synthesizer.recommended_action
