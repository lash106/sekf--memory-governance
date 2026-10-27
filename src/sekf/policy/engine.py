"""Deterministic decision policy v1 (SPEC.md section 9.4). Owner: Shailen.

No LLM calls here. Step A picks a `rule_action` by first-match priority rules R1..R11.
Step B is the Synthesizer veto: it may only downgrade a mutating action to QUARANTINE.
"""

from typing import Protocol
from uuid import UUID

from sekf.contracts import (
    ClaimStatus,
    ConflictType,
    CostSummary,
    DecisionRecord,
    FinalAction,
    JudgeLabel,
    PolicyContext,
    TemporalRelation,
    compute_decision_id,
    compute_request_hash,
)

POLICY_VERSION = "policy-v1"

_MUTATING = {FinalAction.PROMOTE, FinalAction.REVISE, FinalAction.ATTACH_EVIDENCE}

_EXPLANATIONS = {
    "R1": "A role failed or returned invalid output; escalating for human review.",
    "R2": "Evidence refutes the claim with high confidence; rejecting.",
    "R3": "Evidence is insufficient; keeping the claim as episodic memory only.",
    "R4": "Evidence is mixed or weakly refuting; quarantining.",
    "R5": "Support is below the confidence threshold; keeping as episodic memory only.",
    "R6": "Historian confidence is too low or conflict type unknown; escalating for review.",
    "R7": "Supported duplicate of an active claim; attaching the new evidence.",
    "R8": "Supported, reliable newer update of an active claim; revising.",
    "R9": "Unresolved conflict or unreliable source; quarantining.",
    "R10": "Supported new claim with no conflict and an adequate source; promoting.",
    "R11": "No rule matched; escalating for review.",
    "R_SYN_VETO": "The Synthesizer confidently disagreed with a mutating action; quarantining.",
}


def request_hash(context: PolicyContext) -> str:
    """Stable-request fingerprint of a policy context (see `compute_request_hash`)."""
    return compute_request_hash(
        policy_version=POLICY_VERSION,
        candidate=context.candidate,
        evidence=context.evidence,
        thresholds=context.thresholds,
        run_config=context.run_config,
        idempotency_key=context.idempotency_key,
    )


def decision_id_for(hash_hex: str) -> UUID:
    """Deterministic decision ID for a `request_hash` under the current policy version."""
    return compute_decision_id(POLICY_VERSION, hash_hex)


class DecisionPolicy(Protocol):
    def identify(self, context: PolicyContext) -> tuple[UUID, str]:
        """Return `(decision_id, request_hash)` from the stable inputs, without running roles."""
        ...

    def decide(self, context: PolicyContext) -> DecisionRecord: ...


class PolicyEngine:
    """The sole authority over memory mutation."""

    def identify(self, context: PolicyContext) -> tuple[UUID, str]:
        digest = request_hash(context)
        return decision_id_for(digest), digest

    def decide(self, context: PolicyContext) -> DecisionRecord:
        rule_id, rule_action, previous, resulting = self._apply_rules(context)
        fired = [rule_id]
        action = rule_action

        syn = context.synthesizer
        if (
            rule_action in _MUTATING
            and syn is not None
            and syn.recommended_action is not rule_action
            and syn.confidence >= context.thresholds.synthesizer_veto_threshold
        ):
            action = FinalAction.QUARANTINE
            fired.append("R_SYN_VETO")
            previous = []
            resulting = context.candidate.claim_id

        h, j, s = context.historian, context.judge, context.synthesizer
        parts = [r for r in (h, j, s) if r is not None]
        failed = context.role_errors
        cost = CostSummary(
            llm_calls=sum(r.llm_calls for r in parts) + sum(e.llm_calls for e in failed),
            input_tokens=sum(r.input_tokens for r in parts) + sum(e.input_tokens for e in failed),
            output_tokens=sum(r.output_tokens for r in parts)
            + sum(e.output_tokens for e in failed),
            latency_ms=sum(r.latency_ms for r in parts) + sum(e.latency_ms for e in failed),
            estimated_cost_usd=sum(r.estimated_cost_usd for r in parts)
            + sum(e.estimated_cost_usd for e in failed),
        )
        decision_id, digest = self.identify(context)
        return DecisionRecord(
            decision_id=decision_id,
            candidate_claim_id=context.candidate.claim_id,
            final_action=action,
            rule_action=rule_action,
            policy_version=POLICY_VERSION,
            request_hash=digest,
            thresholds=context.thresholds,
            rule_ids_fired=fired,
            previous_claim_ids=previous,
            resulting_claim_id=resulting,
            historian_assessment_id=h.assessment_id if h else None,
            judge_assessment_id=j.assessment_id if j else None,
            synthesizer_recommendation_id=s.recommendation_id if s else None,
            explanation=" ".join(_EXPLANATIONS[r] for r in fired),
            created_at=context.now,
            cost=cost,
        )

    def _apply_rules(self, ctx: PolicyContext) -> tuple[str, FinalAction, list[UUID], UUID | None]:
        t = ctx.thresholds
        cand_id = ctx.candidate.claim_id
        h, j = ctx.historian, ctx.judge

        # R1: failed or invalid role output.
        if ctx.role_errors or h is None or j is None or j.label is JudgeLabel.INVALID_INPUT:
            return "R1", FinalAction.ESCALATE_FOR_REVIEW, [], None

        # R2-R5: Judge-driven outcomes.
        if j.label is JudgeLabel.REFUTED and j.confidence >= t.refute_threshold:
            return "R2", FinalAction.REJECT, [], cand_id
        if j.label is JudgeLabel.INSUFFICIENT_EVIDENCE:
            return "R3", FinalAction.KEEP_EPISODIC, [], None
        if j.label is JudgeLabel.MIXED_EVIDENCE or j.label is JudgeLabel.REFUTED:
            return "R4", FinalAction.QUARANTINE, [], cand_id
        if j.confidence < t.support_threshold:  # SUPPORTED but weak
            return "R5", FinalAction.KEEP_EPISODIC, [], None

        # R6: Historian too unsure to act on.
        if (
            h.confidence < t.historian_confidence_threshold
            or h.conflict_type is ConflictType.UNKNOWN
        ):
            return "R6", FinalAction.ESCALATE_FOR_REVIEW, [], None

        reliable = h.source_reliability_score >= t.source_reliability_threshold
        active_ids = {c.claim_id for c in ctx.related_claims if c.status is ClaimStatus.ACTIVE}

        # R7: supported duplicate/paraphrase of an active claim.
        if h.conflict_type in (ConflictType.DUPLICATE, ConflictType.PARAPHRASE):
            matched = [cid for cid in h.related_claim_ids if cid in active_ids]
            if matched:
                return "R7", FinalAction.ATTACH_EVIDENCE, [matched[0]], matched[0]

        # R8: reliable newer update that clearly supersedes an active claim.
        if (
            h.conflict_type is ConflictType.TEMPORAL_UPDATE
            and h.supersedes_claim_id is not None
            and h.supersedes_claim_id in active_ids
            and reliable
        ):
            return "R8", FinalAction.REVISE, [h.supersedes_claim_id], cand_id

        # R9: unresolved conflict or unreliable source.
        if (
            h.conflict_type
            in (
                ConflictType.VALUE_CONFLICT,
                ConflictType.SOURCE_CONFLICT,
                ConflictType.TEMPORAL_UPDATE,
            )
            or h.temporal_relation is TemporalRelation.OVERLAPPING_VALIDITY
            or not reliable
        ):
            return "R9", FinalAction.QUARANTINE, [], cand_id

        # R10: clean, supported, reliable new claim.
        if h.conflict_type is ConflictType.NONE:
            return "R10", FinalAction.PROMOTE, [], cand_id

        # R11: anything else.
        return "R11", FinalAction.ESCALATE_FOR_REVIEW, [], None
