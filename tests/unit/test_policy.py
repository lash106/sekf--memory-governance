from uuid import uuid4

import pytest
from tests.conftest import (
    NOW,
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
    make_synth,
)

from sekf.contracts import (
    ClaimStatus,
    ConflictType,
    FinalAction,
    JudgeLabel,
    RoleFailure,
    TemporalRelation,
    Thresholds,
)
from sekf.policy import PolicyEngine

A = FinalAction


def run(engine: PolicyEngine, **kw):  # type: ignore[no-untyped-def]
    return engine.decide(make_context(**kw))


def test_r1_role_error_escalates(engine: PolicyEngine) -> None:
    d = run(
        engine,
        role_errors=[RoleFailure(role="judge", error_type="RoleTimeout", message="exceeded 30s")],
    )
    assert d.final_action is A.ESCALATE_FOR_REVIEW
    assert d.rule_ids_fired == ["R1"]
    assert d.resulting_claim_id is None


def test_r1_missing_role_output_escalates(engine: PolicyEngine) -> None:
    cand = make_candidate()
    d = run(engine, candidate=cand, judge=make_judge(cand))
    assert d.rule_ids_fired == ["R1"]


def test_r1_invalid_input_label_escalates(engine: PolicyEngine) -> None:
    cand = make_candidate()
    d = run(
        engine,
        candidate=cand,
        historian=make_historian(cand),
        judge=make_judge(cand, JudgeLabel.INVALID_INPUT),
    )
    assert d.rule_ids_fired == ["R1"]


@pytest.mark.parametrize(
    ("label", "confidence", "expected_rule", "expected_action"),
    [
        (JudgeLabel.REFUTED, 0.9, "R2", A.REJECT),
        (JudgeLabel.REFUTED, 0.5, "R4", A.QUARANTINE),
        (JudgeLabel.INSUFFICIENT_EVIDENCE, 0.9, "R3", A.KEEP_EPISODIC),
        (JudgeLabel.MIXED_EVIDENCE, 0.9, "R4", A.QUARANTINE),
        (JudgeLabel.SUPPORTED, 0.5, "R5", A.KEEP_EPISODIC),
    ],
)
def test_judge_driven_rules(
    engine: PolicyEngine,
    label: JudgeLabel,
    confidence: float,
    expected_rule: str,
    expected_action: FinalAction,
) -> None:
    cand = make_candidate()
    d = run(
        engine,
        candidate=cand,
        historian=make_historian(cand),
        judge=make_judge(cand, label, confidence),
    )
    assert d.rule_ids_fired == [expected_rule]
    assert d.final_action is expected_action


def test_r6_low_historian_confidence_escalates(engine: PolicyEngine) -> None:
    cand = make_candidate()
    d = run(
        engine,
        candidate=cand,
        historian=make_historian(cand, confidence=0.4),
        judge=make_judge(cand),
    )
    assert d.rule_ids_fired == ["R6"]


def test_r6_unknown_conflict_escalates(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h = make_historian(cand, conflict=ConflictType.UNKNOWN)
    assert run(engine, candidate=cand, historian=h, judge=make_judge(cand)).rule_ids_fired == ["R6"]


def test_r7_duplicate_attaches_evidence(engine: PolicyEngine) -> None:
    cand, active = make_candidate(), make_existing()
    h = make_historian(
        cand,
        conflict=ConflictType.DUPLICATE,
        temporal=TemporalRelation.SAME_PERIOD,
        related=[active.claim_id],
    )
    d = run(engine, candidate=cand, related_claims=[active], historian=h, judge=make_judge(cand))
    assert d.final_action is A.ATTACH_EVIDENCE
    assert d.rule_ids_fired == ["R7"]
    assert d.previous_claim_ids == [active.claim_id]
    assert d.resulting_claim_id == active.claim_id


def test_r8_reliable_update_revises(engine: PolicyEngine) -> None:
    cand, active = make_candidate(), make_existing()
    h = make_historian(
        cand,
        conflict=ConflictType.TEMPORAL_UPDATE,
        temporal=TemporalRelation.NEWER_UPDATE,
        related=[active.claim_id],
        supersedes=active.claim_id,
    )
    d = run(engine, candidate=cand, related_claims=[active], historian=h, judge=make_judge(cand))
    assert d.final_action is A.REVISE
    assert d.rule_ids_fired == ["R8"]
    assert d.previous_claim_ids == [active.claim_id]
    assert d.resulting_claim_id == cand.claim_id


def test_unreliable_newer_update_is_quarantined_not_revised(engine: PolicyEngine) -> None:
    """The core idea: a newer claim is not automatically a true claim."""
    cand, active = make_candidate(), make_existing()
    h = make_historian(
        cand,
        conflict=ConflictType.TEMPORAL_UPDATE,
        temporal=TemporalRelation.NEWER_UPDATE,
        related=[active.claim_id],
        supersedes=active.claim_id,
        reliability=0.3,
    )
    d = run(engine, candidate=cand, related_claims=[active], historian=h, judge=make_judge(cand))
    assert d.final_action is A.QUARANTINE
    assert d.rule_ids_fired == ["R9"]


def test_update_of_non_active_claim_is_not_revised(engine: PolicyEngine) -> None:
    cand, old = make_candidate(), make_existing(ClaimStatus.SUPERSEDED)
    h = make_historian(
        cand, conflict=ConflictType.TEMPORAL_UPDATE, related=[old.claim_id], supersedes=old.claim_id
    )
    d = run(engine, candidate=cand, related_claims=[old], historian=h, judge=make_judge(cand))
    assert d.final_action is A.QUARANTINE


@pytest.mark.parametrize("conflict", [ConflictType.VALUE_CONFLICT, ConflictType.SOURCE_CONFLICT])
def test_r9_unresolved_conflict_quarantines(engine: PolicyEngine, conflict: ConflictType) -> None:
    cand = make_candidate()
    d = run(
        engine,
        candidate=cand,
        historian=make_historian(cand, conflict=conflict),
        judge=make_judge(cand),
    )
    assert d.rule_ids_fired == ["R9"]


def test_r9_overlapping_validity_quarantines(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h = make_historian(cand, temporal=TemporalRelation.OVERLAPPING_VALIDITY)
    assert run(engine, candidate=cand, historian=h, judge=make_judge(cand)).rule_ids_fired == ["R9"]


def test_r10_clean_new_claim_promotes(engine: PolicyEngine) -> None:
    cand = make_candidate()
    d = run(engine, candidate=cand, historian=make_historian(cand), judge=make_judge(cand))
    assert d.final_action is A.PROMOTE
    assert d.rule_ids_fired == ["R10"]
    assert d.resulting_claim_id == cand.claim_id


def test_r11_unmatched_escalates(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h = make_historian(cand, conflict=ConflictType.PARAPHRASE)  # paraphrase with no active match
    d = run(engine, candidate=cand, historian=h, judge=make_judge(cand))
    assert d.rule_ids_fired == ["R11"]
    assert d.final_action is A.ESCALATE_FOR_REVIEW


# --------------------------------------------------------------- Synthesizer veto


def test_synthesizer_veto_downgrades_promote(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    s = make_synth(cand, h, j, A.KEEP_EPISODIC, confidence=0.9)
    d = run(engine, candidate=cand, historian=h, judge=j, synthesizer=s)
    assert d.rule_action is A.PROMOTE
    assert d.final_action is A.QUARANTINE
    assert d.rule_ids_fired == ["R10", "R_SYN_VETO"]
    assert d.previous_claim_ids == []


def test_low_confidence_synthesizer_cannot_veto(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    s = make_synth(cand, h, j, A.KEEP_EPISODIC, confidence=0.5)
    d = run(engine, candidate=cand, historian=h, judge=j, synthesizer=s)
    assert d.final_action is A.PROMOTE


def test_synthesizer_cannot_upgrade(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand, JudgeLabel.INSUFFICIENT_EVIDENCE)
    s = make_synth(cand, h, j, A.PROMOTE, confidence=1.0)
    d = run(engine, candidate=cand, historian=h, judge=j, synthesizer=s)
    assert d.final_action is A.KEEP_EPISODIC
    assert "R_SYN_VETO" not in d.rule_ids_fired


# ------------------------------------------------------------- determinism / config


def test_decision_is_deterministic_and_idempotent_by_id(engine: PolicyEngine) -> None:
    cand = make_candidate()
    ctx = make_context(candidate=cand, historian=make_historian(cand), judge=make_judge(cand))
    assert engine.decide(ctx) == engine.decide(ctx)
    assert engine.decide(ctx).decision_id == engine.decide(ctx).decision_id


def test_thresholds_are_configurable_and_recorded(engine: PolicyEngine) -> None:
    cand = make_candidate()
    strict = Thresholds(support_threshold=0.95)
    d = run(
        engine,
        candidate=cand,
        historian=make_historian(cand),
        judge=make_judge(cand, confidence=0.9),
        thresholds=strict,
    )
    assert d.rule_ids_fired == ["R5"]
    assert d.thresholds.support_threshold == 0.95


def test_cost_is_aggregated(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    h.input_tokens, j.input_tokens = 10, 20
    h.llm_calls, j.llm_calls = 1, 2  # e.g. the Judge needed one repair attempt
    h.estimated_cost_usd, j.estimated_cost_usd = 0.01, 0.02
    d = run(engine, candidate=cand, historian=h, judge=j)
    assert d.cost.llm_calls == 3
    assert d.cost.input_tokens == 30
    assert d.cost.estimated_cost_usd == pytest.approx(0.03)


def test_r7_unreliable_duplicate_still_attaches_evidence(engine: PolicyEngine) -> None:
    """SPEC 9.4: R7 has no reliability condition; attaching evidence is non-destructive."""
    cand, active = make_candidate(), make_existing()
    h = make_historian(
        cand,
        conflict=ConflictType.DUPLICATE,
        temporal=TemporalRelation.SAME_PERIOD,
        related=[active.claim_id],
        reliability=0.1,
    )
    d = run(engine, candidate=cand, related_claims=[active], historian=h, judge=make_judge(cand))
    assert d.final_action is A.ATTACH_EVIDENCE
    assert d.rule_ids_fired == ["R7"]


def test_decision_id_changes_when_thresholds_change(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    a = run(engine, candidate=cand, historian=h, judge=j)
    b = run(
        engine, candidate=cand, historian=h, judge=j, thresholds=Thresholds(support_threshold=0.95)
    )
    assert a.decision_id != b.decision_id
    assert a.final_action != b.final_action


def test_decision_id_ignores_the_clock(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    ctx = make_context(candidate=cand, historian=h, judge=j)
    later = ctx.model_copy(update={"now": ctx.now.replace(year=2030)})
    assert engine.decide(ctx).decision_id == engine.decide(later).decision_id


def test_decision_id_is_stable_across_reruns_of_the_same_request(engine: PolicyEngine) -> None:
    """A retry against a real LLM yields new assessment UUIDs, latency, and wording."""
    cand = make_candidate()
    first = run(engine, candidate=cand, historian=make_historian(cand), judge=make_judge(cand))
    h2, j2 = make_historian(cand), make_judge(cand)
    h2.latency_ms, j2.latency_ms = 812, 640
    h2.rationale, j2.rationale = "reworded", "reworded"
    h2.estimated_cost_usd = 0.02
    second = run(engine, candidate=cand, historian=h2, judge=j2)
    assert first.decision_id == second.decision_id


def test_idempotency_key_forces_a_new_decision(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    a = run(engine, candidate=cand, historian=h, judge=j)
    b = run(engine, candidate=cand, historian=h, judge=j, idempotency_key="retry-2")
    assert a.decision_id != b.decision_id


def test_decision_id_changes_with_evidence_and_model(engine: PolicyEngine) -> None:
    from sekf.contracts import EvidenceRecord, RunConfig

    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    ev = EvidenceRecord(
        evidence_id=uuid4(),
        content="x",
        source_id="s",
        source_type="doc",
        retrieved_at=NOW,
        reliability=0.9,
    )
    base = run(engine, candidate=cand, historian=h, judge=j)
    with_ev = run(engine, candidate=cand, historian=h, judge=j, evidence=[ev])
    other_model = run(
        engine, candidate=cand, historian=h, judge=j, run_config=RunConfig(model="other/model")
    )
    assert len({base.decision_id, with_ev.decision_id, other_model.decision_id}) == 3


def _ev(reliability: float, evidence_id=None):  # type: ignore[no-untyped-def]
    from sekf.contracts import EvidenceRecord

    return EvidenceRecord(
        evidence_id=evidence_id or uuid4(),
        content="same text",
        source_id="s",
        source_type="doc",
        retrieved_at=NOW,
        reliability=reliability,
    )


def test_decision_id_changes_with_evidence_reliability(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    eid = uuid4()
    low = run(engine, candidate=cand, historian=h, judge=j, evidence=[_ev(0.30, eid)])
    high = run(engine, candidate=cand, historian=h, judge=j, evidence=[_ev(0.95, eid)])
    assert low.decision_id != high.decision_id
    assert low.request_hash != high.request_hash


def test_decision_id_changes_with_timeout(engine: PolicyEngine) -> None:
    from sekf.contracts import RunConfig

    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    short = run(engine, candidate=cand, historian=h, judge=j, run_config=RunConfig(timeout_s=0.05))
    long = run(engine, candidate=cand, historian=h, judge=j, run_config=RunConfig(timeout_s=30))
    assert short.decision_id != long.decision_id


def test_decision_id_ignores_the_memory_snapshot(engine: PolicyEngine) -> None:
    """A committed REVISE changes memory; the retry must still be recognized afterwards."""
    cand, active = make_candidate(), make_existing()
    h, j = make_historian(cand), make_judge(cand)
    before = run(engine, candidate=cand, related_claims=[active], historian=h, judge=j)
    superseded = active.model_copy(update={"status": ClaimStatus.SUPERSEDED})
    after = run(engine, candidate=cand, related_claims=[superseded], historian=h, judge=j)
    assert before.decision_id == after.decision_id
    assert before.request_hash == after.request_hash


def test_identify_matches_decide_without_running_roles(engine: PolicyEngine) -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    early = engine.identify(make_context(candidate=cand))
    full = engine.decide(make_context(candidate=cand, historian=h, judge=j))
    assert early == (full.decision_id, full.request_hash)
