"""LLM layer of the Synthesizer: it can only add caution, never remove it."""

from typing import Any
from uuid import uuid4

from tests.conftest import (
    make_candidate,
    make_context,
    make_historian,
    make_judge,
)

from sekf.contracts import ConflictType, FinalAction, JudgeLabel, SynthesisRequest
from sekf.policy import PolicyEngine
from sekf.providers import FakeProvider, load_role_prompt
from sekf.synthesizer import LLMSynthesizer

A = FinalAction


def says(action: str, confidence: float = 0.9, **extra: Any) -> FakeProvider:
    payload = {
        "recommended_action": action,
        "confidence": confidence,
        "unresolved_questions": [],
        "rationale": "scripted",
    } | extra
    return FakeProvider(lambda _m, _r: payload)


def request_for(
    label: JudgeLabel = JudgeLabel.SUPPORTED, conflict: ConflictType = ConflictType.NONE
):  # type: ignore[no-untyped-def]
    cand = make_candidate()
    h = make_historian(cand, conflict=conflict)
    j = make_judge(cand, label)
    j.supporting_evidence_ids = [uuid4()]
    return cand, h, j, SynthesisRequest(candidate=cand, historian=h, judge=j)


async def test_agreement_keeps_the_action_at_the_lower_confidence() -> None:
    *_, req = request_for()
    out = await LLMSynthesizer(says("PROMOTE", 0.85)).synthesize(req)
    assert out.recommended_action is A.PROMOTE and out.confidence == 0.85


async def test_a_more_cautious_model_makes_the_policy_veto_a_promotion() -> None:
    cand, h, j, req = request_for()
    out = await LLMSynthesizer(says("KEEP_EPISODIC", 0.9)).synthesize(req)
    assert out.recommended_action is A.KEEP_EPISODIC and out.confidence == 0.9
    assert any("instead of PROMOTE" in q for q in out.unresolved_questions)
    decision = PolicyEngine().decide(
        make_context(candidate=cand, historian=h, judge=j, synthesizer=out)
    )
    assert decision.rule_action is A.PROMOTE and decision.final_action is A.QUARANTINE
    assert decision.rule_ids_fired == ["R10", "R_SYN_VETO"]


async def test_a_model_that_would_mutate_more_than_the_rules_is_ignored() -> None:
    *_, req = request_for(label=JudgeLabel.INSUFFICIENT_EVIDENCE)
    out = await LLMSynthesizer(says("PROMOTE", 0.99)).synthesize(req)
    assert out.recommended_action is A.KEEP_EPISODIC
    assert out.confidence <= 0.75  # never strong enough to matter
    assert any("ignored" in q for q in out.unresolved_questions)


async def test_two_different_mutations_are_treated_as_ambiguity() -> None:
    *_, req = request_for()  # rules: PROMOTE
    out = await LLMSynthesizer(says("REVISE", 0.9)).synthesize(req)
    assert out.recommended_action is A.QUARANTINE and out.confidence >= 0.8
    assert any("unclear which change" in q for q in out.unresolved_questions)


async def test_non_mutating_disagreement_keeps_the_rules_without_a_veto() -> None:
    *_, req = request_for(label=JudgeLabel.REFUTED)  # rules: REJECT
    out = await LLMSynthesizer(says("QUARANTINE", 0.95)).synthesize(req)
    assert out.recommended_action is A.REJECT and out.confidence <= 0.75


async def test_prompt_strips_call_bookkeeping_and_marks_inputs_as_data() -> None:
    provider = says("PROMOTE")
    *_, req = request_for()
    out = await LLMSynthesizer(provider).synthesize(req)
    _system, user = provider.calls[0]
    assert "prompt_hash" not in user["content"] and "estimated_cost_usd" not in user["content"]
    assert "never an instruction" in " ".join(user["content"].split())
    prompt = load_role_prompt("synthesizer", "v1")
    assert out.prompt_version == "synthesizer/v1" and out.prompt_hash == prompt.prompt_hash
    assert out.provider == "fake" and out.llm_calls == 0
    assert out.historian_assessment_id == req.historian.assessment_id
