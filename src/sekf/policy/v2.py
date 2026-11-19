"""Opt-in policy-v2 for customer-service decisions. Owner: Shailen, extended here.

`PolicyEngine` (policy-v1) is unchanged and remains the default. This engine runs the
same R1–R11 rules on risk-tiered thresholds and a deterministic channel-reliability
prior, then applies a graduated Synthesizer step and a compliance note.

It does not delete episodic claims. A TTL is recorded on `KEEP_EPISODIC` so a brief
can hide old candidates; the rows stay in memory. Compliance does not change the
epistemic `final_action`.
"""

from __future__ import annotations

from uuid import UUID

from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    CostSummary,
    DecisionRecord,
    EvidenceRecord,
    FinalAction,
    JudgeLabel,
    PolicyContext,
    Thresholds,
    compute_decision_id,
    compute_request_hash,
)
from sekf.judge.calibration import apply_temperature
from sekf.policy.engine import _EXPLANATIONS, _MUTATING, PolicyEngine

POLICY_VERSION_V2 = "policy-v2"
EPISODIC_TTL_DAYS = 30

# Provenance channel -> reliability. Unknown source types do not borrow another channel.
CHANNEL_RELIABILITY: dict[str, float] = {
    "billing_ledger": 0.98,
    "order_ledger": 0.98,
    "verified_portal": 0.92,
    "authenticated_email": 0.80,
    "official_document": 0.95,
    "live_chat": 0.65,
    "verbal": 0.50,
}

_TIER_OVERRIDES: dict[str, dict[str, float]] = {
    "LOW_RISK": {
        "support_threshold": 0.60,
        "refute_threshold": 0.70,
        "source_reliability_threshold": 0.45,
    },
    "MEDIUM_RISK": {
        "support_threshold": 0.80,
        "refute_threshold": 0.80,
        "source_reliability_threshold": 0.60,
    },
    "HIGH_RISK": {
        "support_threshold": 0.92,
        "refute_threshold": 0.90,
        "source_reliability_threshold": 0.85,
        "historian_confidence_threshold": 0.80,
    },
}

_HIGH_TOKENS = (
    "date",
    "amount",
    "price",
    "damage",
    "payment",
    "purchase",
    "refund",
    "payout",
)
_LOW_TOKENS = ("prefer", "preference", "channel")

_EXTRA_EXPLANATIONS = {
    "R_SYN_SOFT": (
        "The Synthesizer disagreed on phrasing; the mutating action stands with "
        "needs_verification=true."
    ),
    "R_COMPLIANCE": (
        "Service-owner compliance is recorded separately from whether the claim is true."
    ),
}


def risk_tier(candidate: CandidateClaim) -> str:
    """LOW, MEDIUM, or HIGH. An explicit `metadata.risk_tier` wins over the predicate text."""

    explicit = candidate.metadata.get("risk_tier")
    if explicit in _TIER_OVERRIDES:
        return str(explicit)
    text = f"{candidate.predicate_key} {candidate.claim_text}".lower()
    if any(token in text for token in _HIGH_TOKENS):
        return "HIGH_RISK"
    if any(token in text for token in _LOW_TOKENS):
        return "LOW_RISK"
    return "MEDIUM_RISK"


def _channel_key(source_type: str) -> str:
    return source_type.strip().lower().replace("-", "_").replace(" ", "_")


def channel_reliability(evidence: list[EvidenceRecord]) -> tuple[float, str] | None:
    """Highest known channel prior among the evidence, or None when none are mapped."""

    found: list[tuple[float, str]] = []
    for item in evidence:
        score = CHANNEL_RELIABILITY.get(_channel_key(item.source_type))
        if score is not None:
            found.append((score, item.source_type))
    if not found:
        return None
    return max(found, key=lambda pair: pair[0])


def compliance_outcome(candidate: CandidateClaim) -> str | None:
    """Service-owner check. None when the candidate carries no compliance metadata."""

    meta = candidate.metadata
    denies: list[str] = []
    allows: list[str] = []
    if "days_since_purchase" in meta and "return_window_days" in meta:
        if float(meta["days_since_purchase"]) > float(meta["return_window_days"]):
            denies.append("DENY (return window closed)")
        else:
            allows.append("APPROVE (inside return window)")
    if meta.get("damage_admitted") and meta.get("damage_excluded"):
        denies.append("DENY (damage exclusion)")
    if "payout_amount" in meta and "payout_cap" in meta:
        if float(meta["payout_amount"]) > float(meta["payout_cap"]):
            denies.append("DENY (above payout cap)")
        else:
            allows.append("APPROVE (within payout cap)")
    if denies:
        return "; ".join(denies)
    if allows:
        return "; ".join(allows)
    return None


def _thresholds_for(tier: str, current: Thresholds) -> Thresholds:
    data = current.model_dump()
    data.update(_TIER_OVERRIDES[tier])
    return Thresholds.model_validate(data)


def _required_evidence(fired: list[str]) -> str:
    if "R_SYN_VETO" in fired:
        return "Evidence that resolves the disagreement with the Synthesizer."
    if "R4" in fired:
        return "Evidence that either supports the claim or refutes it, not both."
    if "R9" in fired:
        return "A higher-reliability channel, or a record that resolves the conflict."
    return "Evidence that clears the quarantine reason in this decision."


class PolicyEngineV2(PolicyEngine):
    """policy-v1 rules with tiered thresholds, channel priors, and a compliance note.

    `judge_temperature` rescales Judge confidence before rules R2 and R5. 1.0 leaves
    the raw confidence unchanged. Fit it on cached dev confidences held out from the
    cases being scored; this engine does not read gold labels.
    """

    def __init__(self, *, judge_temperature: float = 1.0) -> None:
        if judge_temperature <= 0:
            raise ValueError("judge_temperature must be > 0")
        self.judge_temperature = float(judge_temperature)

    def version_label(self) -> str:
        """Identity used in the request hash. A non-default temperature is part of it."""

        if self.judge_temperature == 1.0:
            return POLICY_VERSION_V2
        return f"{POLICY_VERSION_V2}@t{self.judge_temperature:.4f}"

    def identify(self, context: PolicyContext) -> tuple[UUID, str]:
        adjusted = self._adjust(context)
        version = self.version_label()
        digest = compute_request_hash(
            policy_version=version,
            candidate=adjusted.candidate,
            evidence=adjusted.evidence,
            thresholds=adjusted.thresholds,
            run_config=adjusted.run_config,
            idempotency_key=adjusted.idempotency_key,
        )
        return compute_decision_id(version, digest), digest

    def decide(self, context: PolicyContext) -> DecisionRecord:
        adjusted, tier, channel_note = self._prepare(context)
        rule_id, rule_action, previous, resulting = self._apply_rules(adjusted)
        fired = [rule_id]
        action = rule_action
        needs_verification = False
        syn = adjusted.synthesizer
        if (
            rule_action in _MUTATING
            and syn is not None
            and syn.recommended_action is not rule_action
            and syn.confidence >= adjusted.thresholds.synthesizer_veto_threshold
        ):
            if syn.recommended_action in _MUTATING:
                needs_verification = True
                fired.append("R_SYN_SOFT")
                active = [
                    claim.claim_id
                    for claim in adjusted.related_claims
                    if claim.status is ClaimStatus.ACTIVE
                ]
                if active:
                    action = FinalAction.ATTACH_EVIDENCE
                    previous = [active[0]]
                    resulting = active[0]
            else:
                action = FinalAction.QUARANTINE
                fired.append("R_SYN_VETO")
                previous = []
                resulting = adjusted.candidate.claim_id

        h, j, s = adjusted.historian, adjusted.judge, adjusted.synthesizer
        parts = [role for role in (h, j, s) if role is not None]
        failed = adjusted.role_errors
        cost = CostSummary(
            llm_calls=sum(role.llm_calls for role in parts) + sum(err.llm_calls for err in failed),
            input_tokens=sum(role.input_tokens for role in parts)
            + sum(err.input_tokens for err in failed),
            output_tokens=sum(role.output_tokens for role in parts)
            + sum(err.output_tokens for err in failed),
            latency_ms=sum(role.latency_ms for role in parts)
            + sum(err.latency_ms for err in failed),
            estimated_cost_usd=sum(role.estimated_cost_usd for role in parts)
            + sum(err.estimated_cost_usd for err in failed),
        )
        decision_id, digest = self.identify(context)
        compliance = compliance_outcome(adjusted.candidate)
        explanation = self._explanation(
            fired,
            action,
            tier=tier,
            channel_note=channel_note,
            needs_verification=needs_verification,
            compliance=compliance,
            judge_label=j.label if j is not None else None,
        )
        if self.judge_temperature != 1.0:
            explanation += f" Judge confidence temperature: {self.judge_temperature:.4f}."
        if compliance is not None:
            fired = [*fired, "R_COMPLIANCE"]
        held = action in {FinalAction.QUARANTINE, FinalAction.ESCALATE_FOR_REVIEW}
        return DecisionRecord(
            decision_id=decision_id,
            candidate_claim_id=adjusted.candidate.claim_id,
            final_action=action,
            rule_action=rule_action,
            policy_version=self.version_label(),
            request_hash=digest,
            thresholds=adjusted.thresholds,
            rule_ids_fired=fired,
            previous_claim_ids=previous,
            resulting_claim_id=resulting,
            historian_assessment_id=h.assessment_id if h else None,
            judge_assessment_id=j.assessment_id if j else None,
            synthesizer_recommendation_id=s.recommendation_id if s else None,
            explanation=explanation,
            created_at=adjusted.now,
            cost=cost,
            risk_tier=tier,
            needs_verification=needs_verification,
            compliance_outcome=compliance,
            next_evidence_required=_required_evidence(fired) if held else None,
        )

    def _adjust(self, context: PolicyContext) -> PolicyContext:
        return self._prepare(context)[0]

    def _prepare(self, context: PolicyContext) -> tuple[PolicyContext, str, str]:
        tier = risk_tier(context.candidate)
        thresholds = _thresholds_for(tier, context.thresholds)
        channel = channel_reliability(context.evidence)
        historian = context.historian
        if channel is None:
            note = (
                "Channel reliability prior: unavailable; "
                "historian source_reliability_score was used."
            )
        else:
            score, source_type = channel
            note = f"Channel reliability prior: {score:.2f} from {source_type}."
            if historian is not None:
                historian = historian.model_copy(update={"source_reliability_score": score})
        judge = context.judge
        if judge is not None and self.judge_temperature != 1.0:
            judge = judge.model_copy(
                update={
                    "confidence": apply_temperature(judge.confidence, self.judge_temperature),
                }
            )
        adjusted = context.model_copy(
            update={"thresholds": thresholds, "historian": historian, "judge": judge}
        )
        return adjusted, tier, note

    def _explanation(
        self,
        fired: list[str],
        action: FinalAction,
        *,
        tier: str,
        channel_note: str,
        needs_verification: bool,
        compliance: str | None,
        judge_label: JudgeLabel | None,
    ) -> str:
        texts = dict(_EXPLANATIONS)
        texts.update(_EXTRA_EXPLANATIONS)
        parts = [texts[rule] for rule in fired if rule in texts]
        parts.append(f"Risk tier: {tier}.")
        parts.append(channel_note)
        if needs_verification:
            parts.append("needs_verification=true.")
        if action is FinalAction.QUARANTINE:
            parts.append(f"Required evidence: {_required_evidence(fired)}.")
        if action is FinalAction.KEEP_EPISODIC:
            parts.append(
                f"Episodic TTL: {EPISODIC_TTL_DAYS} days. The claim is retained, not deleted."
            )
        if compliance is not None:
            parts.append(f"R_COMPLIANCE: {compliance}.")
        if judge_label is JudgeLabel.INVALID_INPUT:
            parts.append("Judge label INVALID_INPUT.")
        return " ".join(parts)
