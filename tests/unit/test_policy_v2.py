"""policy-v2 is opt-in. policy-v1 stays the default and is not rewritten here."""

from uuid import uuid4

from tests.conftest import (
    NOW,
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
    make_synth,
)

from sekf.contracts import EvidenceRecord, FinalAction, JudgeLabel
from sekf.policy import PolicyEngine, PolicyEngineV2

A = FinalAction


def test_high_risk_requires_more_support_than_policy_v1() -> None:
    cand = make_candidate().model_copy(update={"metadata": {"risk_tier": "HIGH_RISK"}})
    historian = make_historian(cand)
    judge = make_judge(cand, confidence=0.85)
    context = make_context(candidate=cand, historian=historian, judge=judge)
    v1 = PolicyEngine().decide(context)
    v2 = PolicyEngineV2().decide(context)
    assert v1.final_action is A.PROMOTE
    assert v1.policy_version == "policy-v1"
    assert v1.risk_tier is None
    assert v1.needs_verification is False
    assert v1.compliance_outcome is None
    assert v1.next_evidence_required is None
    assert v2.final_action is A.KEEP_EPISODIC
    assert v2.policy_version == "policy-v2"
    assert "HIGH_RISK" in v2.explanation
    assert "retained, not deleted" in v2.explanation


def test_channel_prior_replaces_a_low_historian_score() -> None:
    cand = make_candidate()
    evidence = [
        EvidenceRecord(
            evidence_id=uuid4(),
            content="The billing ledger records the Friday closing time.",
            source_id="billing",
            source_type="billing_ledger",
            retrieved_at=NOW,
            reliability=0.2,
        )
    ]
    historian = make_historian(cand, reliability=0.2)
    judge = make_judge(cand, confidence=0.95)
    context = make_context(candidate=cand, evidence=evidence, historian=historian, judge=judge)
    assert PolicyEngine().decide(context).final_action is A.QUARANTINE
    v2 = PolicyEngineV2().decide(context)
    assert v2.final_action is A.PROMOTE
    assert "billing_ledger" in v2.explanation
    assert v2.thresholds.source_reliability_threshold == 0.60


def test_phrasing_disagreement_is_a_soft_flag_and_a_real_conflict_quarantines() -> None:
    cand = make_candidate()
    related = make_existing()
    historian = make_historian(cand)
    judge = make_judge(cand, confidence=0.95)
    soft = make_synth(cand, historian, judge, A.REVISE, confidence=0.9)
    hard = make_synth(cand, historian, judge, A.QUARANTINE, confidence=0.9)
    base = dict(candidate=cand, historian=historian, judge=judge, related_claims=[related])
    v1 = PolicyEngine().decide(make_context(synthesizer=soft, **base))
    phrasing = PolicyEngineV2().decide(make_context(synthesizer=soft, **base))
    conflict = PolicyEngineV2().decide(make_context(synthesizer=hard, **base))
    assert v1.final_action is A.QUARANTINE
    assert "R_SYN_VETO" in v1.rule_ids_fired
    assert phrasing.final_action is A.ATTACH_EVIDENCE
    assert "R_SYN_SOFT" in phrasing.rule_ids_fired
    assert "needs_verification=true" in phrasing.explanation
    assert conflict.final_action is A.QUARANTINE
    assert "Required evidence:" in conflict.explanation


def test_compliance_does_not_change_the_epistemic_action() -> None:
    cand = make_candidate().model_copy(
        update={"metadata": {"days_since_purchase": 40, "return_window_days": 30}}
    )
    historian = make_historian(cand)
    judge = make_judge(cand, confidence=0.95)
    decision = PolicyEngineV2().decide(
        make_context(candidate=cand, historian=historian, judge=judge)
    )
    assert decision.final_action is A.PROMOTE
    assert "R_COMPLIANCE" in decision.rule_ids_fired
    assert "DENY (return window closed)" in decision.explanation


def test_unknown_channel_says_the_historian_score_was_used() -> None:
    cand = make_candidate()
    evidence = [
        EvidenceRecord(
            evidence_id=uuid4(),
            content="A note from an unnamed channel.",
            source_id="scratch",
            source_type="sticky_note",
            retrieved_at=NOW,
            reliability=0.4,
        )
    ]
    decision = PolicyEngineV2().decide(
        make_context(
            candidate=cand,
            evidence=evidence,
            historian=make_historian(cand, reliability=0.4),
            judge=make_judge(cand, JudgeLabel.INSUFFICIENT_EVIDENCE),
        )
    )
    assert "unavailable" in decision.explanation
    assert decision.final_action is A.KEEP_EPISODIC
