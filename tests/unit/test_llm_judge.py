"""Sprint 1: LLMJudge + evidence-ID validation + Governance wiring."""

from typing import Any
from uuid import UUID

import pytest
from tests.conftest import NOW

from sekf.api import Governance
from sekf.contracts import DecideRequest, EvidenceRecord, FinalAction, JudgeLabel, JudgeRequest
from sekf.errors import CallUsage, RoleOutputInvalid
from sekf.historian import FakeHistorian
from sekf.judge import LLMJudge
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine
from sekf.providers import FakeProvider, load_role_prompt
from sekf.synthesizer import FakeSynthesizer

EVIDENCE_ID = UUID("33333333-3333-4333-8333-333333333333")
UNKNOWN_ID = UUID("99999999-9999-4999-8999-999999999999")


def _supported_payload(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "label": "SUPPORTED",
        "confidence": 0.91,
        "supporting_evidence_ids": [str(EVIDENCE_ID)],
        "refuting_evidence_ids": [],
        "missing_information": [],
        "rationale": "The official schedule states Friday closing is 18:00.",
    }


def _hallucinated_payload(_messages: Any, _model: Any) -> dict[str, Any]:
    payload = _supported_payload(_messages, _model)
    payload["supporting_evidence_ids"] = [str(UNKNOWN_ID)]
    return payload


def _judge_request(decide_request: DecideRequest) -> JudgeRequest:
    return JudgeRequest(
        candidate=decide_request.candidate,
        evidence=decide_request.evidence,
        run_config=decide_request.run_config,
    )


async def test_llm_judge_supports_claim_with_real_evidence_id(
    decide_request: DecideRequest,
) -> None:
    judge = LLMJudge(FakeProvider(_supported_payload))
    assessment = await judge.evaluate(_judge_request(decide_request))
    assert assessment.label is JudgeLabel.SUPPORTED
    assert assessment.supporting_evidence_ids == [EVIDENCE_ID]
    assert assessment.prompt_version == "judge/v1"
    assert assessment.prompt_hash == load_role_prompt("judge", "v1").prompt_hash
    assert assessment.provider == "fake"
    assert assessment.candidate_claim_id == decide_request.candidate.claim_id
    again = await judge.evaluate(_judge_request(decide_request))
    assert again.assessment_id != assessment.assessment_id


async def test_llm_judge_rejects_hallucinated_evidence_id(
    decide_request: DecideRequest,
) -> None:
    judge = LLMJudge(FakeProvider(_hallucinated_payload))
    with pytest.raises(RoleOutputInvalid, match="not in the request"):
        await judge.evaluate(_judge_request(decide_request))


async def test_governance_with_llm_judge_revises_lab_fixture(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    governance = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=LLMJudge(FakeProvider(_supported_payload)),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    response = await governance.decide(decide_request)
    assert response.decision.final_action is FinalAction.REVISE
    assert response.decision.rule_ids_fired == ["R8"]
    assert response.judge is not None
    assert response.judge.label is JudgeLabel.SUPPORTED
    assert response.historian is not None
    assert response.synthesizer is not None
    assert response.memory_delta is None


async def test_hallucinated_evidence_id_escalates_in_governance(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    governance = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=LLMJudge(FakeProvider(_hallucinated_payload)),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    response = await governance.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.rule_ids_fired == ["R1"]
    assert response.judge is None
    assert any(
        err.role == "judge" and err.error_type == "RoleOutputInvalid"
        for err in response.role_errors
    )


async def test_governance_records_cost_when_judge_repairs_fail(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    class _FailingJudge:
        async def evaluate(self, _request: JudgeRequest) -> None:
            raise RoleOutputInvalid(
                "repairs exhausted",
                usage=CallUsage(llm_calls=3, input_tokens=15, estimated_cost_usd=0.01),
            )

    governance = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=_FailingJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    response = await governance.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.cost.llm_calls == 3
    assert response.decision.cost.input_tokens == 15
    assert response.decision.cost.estimated_cost_usd == pytest.approx(0.01)
    assert response.role_errors[0].llm_calls == 3


def test_lab_fixture_evidence_id(decide_request: DecideRequest) -> None:
    evidence = decide_request.evidence[0]
    assert isinstance(evidence, EvidenceRecord)
    assert evidence.evidence_id == EVIDENCE_ID
