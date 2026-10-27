"""Sprint 2: single_judge baseline decide() and R1 fail-safe."""

import asyncio
import time
from typing import Any
from uuid import UUID

import pytest
from tests.conftest import NOW

from sekf.baselines import POLICY_VERSION, SingleJudgeBaseline
from sekf.baselines.single_judge import DEFAULT_POLICY_VERSION
from sekf.baselines.single_judge.schemas import SingleJudgeLLMOutput
from sekf.contracts import DecideMode, DecideRequest, FinalAction
from sekf.errors import CommitRejected, SekfError
from sekf.memory import InMemoryRepository
from sekf.providers import FakeProvider, load_baseline_prompt
from sekf.providers.base import LLMResult


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
    assert response.mode is DecideMode.DRY_RUN
    assert response.decision.final_action is FinalAction.REVISE
    assert response.decision.rule_action is FinalAction.REVISE
    assert response.decision.policy_version == POLICY_VERSION == DEFAULT_POLICY_VERSION
    assert response.decision.rule_ids_fired == []
    assert response.decision.previous_claim_ids == [UUID("11111111-1111-4111-8111-111111111111")]
    assert response.decision.resulting_claim_id == decide_request.candidate.claim_id
    assert response.historian is None
    assert response.judge is None
    assert response.synthesizer is None
    assert response.memory_delta is None
    assert response.role_errors == []
    assert repo.decision_count() == 0
    audit = response.baseline_audit
    assert audit is not None
    prompt = load_baseline_prompt("single_judge", "v1")
    assert audit.provider == "fake"
    assert audit.model == "fake/deterministic"
    assert audit.model != decide_request.run_config.model
    assert audit.prompt_version == prompt.version
    assert audit.prompt_hash == prompt.prompt_hash
    assert audit.temperature == decide_request.run_config.temperature
    assert audit.seed == decide_request.run_config.seed
    assert audit.confidence == pytest.approx(0.88)
    assert audit.recommended_action is FinalAction.REVISE
    assert audit.raw_output["recommended_action"] == "REVISE"
    assert audit.raw_output["rationale"].startswith("Newer official")
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
    assert response.decision.resulting_claim_id is None
    assert response.decision.previous_claim_ids == []
    assert response.decision.rule_ids_fired == ["R1"]
    assert response.role_errors[0].error_type == "SekfError"
    assert response.baseline_audit is None


def _keep(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "recommended_action": "KEEP_EPISODIC",
        "confidence": 0.4,
        "rationale": "not enough",
    }


def _attach(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "recommended_action": "ATTACH_EVIDENCE",
        "confidence": 0.9,
        "rationale": "same fact",
        "target_claim_id": "11111111-1111-4111-8111-111111111111",
    }


async def test_single_judge_keep_episodic_has_no_resulting_claim(
    decide_request: DecideRequest,
) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_keep), clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.KEEP_EPISODIC
    assert response.decision.resulting_claim_id is None
    assert response.decision.previous_claim_ids == []


async def test_single_judge_attach_names_the_active_claim(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_attach), repo, clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    target = UUID("11111111-1111-4111-8111-111111111111")
    assert response.decision.final_action is FinalAction.ATTACH_EVIDENCE
    assert response.decision.previous_claim_ids == [target]
    assert response.decision.resulting_claim_id == target


async def test_single_judge_rejects_commit_mode(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    baseline = SingleJudgeBaseline(FakeProvider(_revise_payload), repo, clock=lambda: NOW)
    request = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    with pytest.raises(CommitRejected, match="dry_run"):
        await baseline.decide(request)
    assert repo.decision_count() == 0


class _SlowProvider:
    async def complete_json(self, **_kwargs: Any) -> LLMResult:
        await asyncio.sleep(0.2)
        raise AssertionError("timeout should cancel this call")


async def test_single_judge_honors_timeout(decide_request: DecideRequest) -> None:
    request = decide_request.model_copy(
        update={"run_config": decide_request.run_config.model_copy(update={"timeout_s": 0.05})}
    )
    baseline = SingleJudgeBaseline(_SlowProvider(), clock=lambda: NOW)
    started = time.perf_counter()
    response = await baseline.decide(request)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.15
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.role_errors[0].error_type == "RoleTimeout"


class _PaidAmbiguousProvider:
    async def complete_json(self, **_kwargs: Any) -> LLMResult:
        return LLMResult(
            parsed=SingleJudgeLLMOutput(
                recommended_action=FinalAction.ATTACH_EVIDENCE,
                confidence=0.9,
                rationale="two active claims",
            ),
            model="fake/deterministic",
            provider="fake",
            llm_calls=2,
            input_tokens=11,
        )


async def test_ambiguous_target_keeps_the_successful_call_cost(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    from tests.conftest import make_existing

    first = repo.get_claim(UUID("11111111-1111-4111-8111-111111111111"))
    assert first is not None
    second = make_existing().model_copy(
        update={
            "subject_key": decide_request.candidate.subject_key,
            "predicate_key": decide_request.candidate.predicate_key,
        }
    )
    memory = InMemoryRepository([first, second])
    baseline = SingleJudgeBaseline(_PaidAmbiguousProvider(), memory, clock=lambda: NOW)
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.cost.llm_calls == 2
    assert response.decision.cost.input_tokens == 11
    assert response.baseline_audit is not None
    assert response.baseline_audit.confidence == pytest.approx(0.9)
    assert response.baseline_audit.llm_calls == 2
    assert response.baseline_audit.raw_output["recommended_action"] == "ATTACH_EVIDENCE"
    assert memory.decision_count() == 0
