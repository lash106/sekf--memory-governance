"""Sprint 2: single_judge baseline decide() and R1 fail-safe."""

from typing import Any

from tests.conftest import NOW

from sekf.baselines import POLICY_VERSION, SingleJudgeBaseline
from sekf.baselines.single_judge import DEFAULT_POLICY_VERSION
from sekf.contracts import DecideRequest, FinalAction
from sekf.errors import SekfError
from sekf.memory import InMemoryRepository
from sekf.providers import FakeProvider, load_baseline_prompt


def _revise_payload(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "recommended_action": "REVISE",
        "confidence": 0.88,
        "rationale": "Newer official schedule should supersede the 17:00 claim.",
    }


def _invalid_action_payload(_messages: Any, _model: Any) -> dict[str, Any]:
    return {"recommended_action": "NOT_AN_ACTION", "confidence": 0.5, "rationale": "bad"}


def _boom(_messages: Any, _model: Any) -> dict[str, Any]:
    raise SekfError("provider exploded")


async def test_single_judge_baseline_returns_decide_response(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_revise_payload), repo, clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    assert response.mode is decide_request.mode
    assert response.decision.final_action is FinalAction.REVISE
    assert response.decision.rule_action is FinalAction.REVISE
    assert response.decision.policy_version == POLICY_VERSION == DEFAULT_POLICY_VERSION
    assert response.decision.rule_ids_fired == []
    assert response.historian is None
    assert response.judge is None
    assert response.synthesizer is None
    assert response.memory_delta is None
    assert response.role_errors == []
    assert repo.decision_count() == 0
    assert baseline.policy_version == POLICY_VERSION
    assert load_baseline_prompt("single_judge", "v1").version == "baselines/single_judge/v1"


async def test_single_judge_invalid_output_escalates(
    decide_request: DecideRequest,
) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_invalid_action_payload), clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.rule_ids_fired == ["R1"]
    assert response.role_errors[0].error_type == "RoleOutputInvalid"


async def test_single_judge_sekf_error_escalates(decide_request: DecideRequest) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_boom), clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.rule_ids_fired == ["R1"]
    assert response.role_errors[0].error_type == "SekfError"
