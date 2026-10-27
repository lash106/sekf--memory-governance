"""Offline tests for generic_ensemble majority vote and baseline wiring."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID, uuid4

import pytest
from tests.conftest import NOW, make_judge

from sekf.baselines import GenericEnsembleBaseline, GenericEnsembleJudge, majority_judge_label
from sekf.baselines.generic_ensemble.aggregation import label_with_quorum, quorum_size
from sekf.contracts import (
    DecideMode,
    DecideRequest,
    FinalAction,
    JudgeAssessment,
    JudgeLabel,
    JudgeRequest,
)
from sekf.errors import CallUsage, CommitRejected, RoleOutputInvalid, RoleTimeout
from sekf.judge import LLMJudge
from sekf.memory import InMemoryRepository
from sekf.providers import FakeProvider

EVIDENCE_ID = UUID("33333333-3333-4333-8333-333333333333")


def test_quorum_is_a_majority_of_the_panel() -> None:
    assert quorum_size(3) == 2
    assert quorum_size(1) == 1
    labels = [JudgeLabel.SUPPORTED, JudgeLabel.REFUTED, JudgeLabel.MIXED_EVIDENCE]
    assert label_with_quorum(labels, 3) is None
    assert (
        label_with_quorum([JudgeLabel.SUPPORTED, JudgeLabel.SUPPORTED], 3) is JudgeLabel.SUPPORTED
    )
    assert label_with_quorum([JudgeLabel.SUPPORTED], 3) is None


def test_majority_and_ties() -> None:
    assert (
        majority_judge_label([JudgeLabel.SUPPORTED, JudgeLabel.SUPPORTED, JudgeLabel.REFUTED])
        is JudgeLabel.SUPPORTED
    )
    assert (
        majority_judge_label([JudgeLabel.SUPPORTED, JudgeLabel.REFUTED, JudgeLabel.MIXED_EVIDENCE])
        is JudgeLabel.INSUFFICIENT_EVIDENCE
    )
    assert majority_judge_label([]) is JudgeLabel.INSUFFICIENT_EVIDENCE


def _supported(_messages: Any, _model: Any) -> dict[str, Any]:
    return {
        "label": "SUPPORTED",
        "confidence": 0.9,
        "supporting_evidence_ids": [str(EVIDENCE_ID)],
        "refuting_evidence_ids": [],
        "missing_information": [],
        "rationale": "ok",
    }


@pytest.mark.asyncio
async def test_ensemble_judge_three_calls(
    decide_request: DecideRequest, repo: InMemoryRepository
) -> None:
    member = LLMJudge(FakeProvider(_supported))
    ensemble = GenericEnsembleJudge(member, n_members=3)
    assessment = await ensemble.evaluate(
        JudgeRequest(
            candidate=decide_request.candidate,
            evidence=decide_request.evidence,
            run_config=decide_request.run_config,
        )
    )
    assert assessment.label is JudgeLabel.SUPPORTED
    assert assessment.confidence == pytest.approx(1.0)
    assert assessment.llm_calls == 0  # fake provider


@pytest.mark.asyncio
async def test_ensemble_baseline_decide(
    decide_request: DecideRequest, repo: InMemoryRepository
) -> None:
    baseline = GenericEnsembleBaseline(
        LLMJudge(FakeProvider(_supported)),
        memory=repo,
        clock=lambda: NOW,
    )
    response = await baseline.decide(decide_request)
    # Supported vote promotes. It must not REVISE via FakeHistorian + PolicyEngine.
    assert response.decision.final_action is FinalAction.PROMOTE
    assert response.historian is None
    assert response.synthesizer is None
    assert response.judge is not None
    assert response.judge.label is JudgeLabel.SUPPORTED
    assert response.mode is DecideMode.DRY_RUN
    committed = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    with pytest.raises(CommitRejected, match="dry_run"):
        await baseline.decide(committed)
    assert repo.decision_count() == 0


class _ScriptedJudge:
    def __init__(self, outcomes: list[JudgeAssessment | Exception]) -> None:
        self._outcomes = list(outcomes)

    async def evaluate(self, _request: JudgeRequest) -> JudgeAssessment:
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _paid_failure(calls: int, tokens: int) -> RoleOutputInvalid:
    return RoleOutputInvalid(
        "member failed",
        usage=CallUsage(
            llm_calls=calls, input_tokens=tokens, output_tokens=4, estimated_cost_usd=0.01
        ),
    )


@pytest.mark.asyncio
async def test_one_surviving_vote_does_not_promote(decide_request: DecideRequest) -> None:
    success = make_judge(decide_request.candidate).model_copy(
        update={"llm_calls": 1, "input_tokens": 10, "assessment_id": uuid4()}
    )
    ensemble = GenericEnsembleJudge(
        _ScriptedJudge([_paid_failure(3, 100), _paid_failure(3, 100), success]),
        n_members=3,
    )
    request = JudgeRequest(
        candidate=decide_request.candidate,
        evidence=decide_request.evidence,
        run_config=decide_request.run_config,
    )
    with pytest.raises(RoleOutputInvalid, match="no operational quorum") as raised:
        await ensemble.evaluate(request)
    usage = raised.value.usage
    assert isinstance(usage, CallUsage)
    assert usage.llm_calls == 7
    assert usage.input_tokens == 210


@pytest.mark.asyncio
async def test_all_member_failures_keep_their_usage(decide_request: DecideRequest) -> None:
    baseline = GenericEnsembleBaseline(
        _ScriptedJudge([_paid_failure(3, 40), _paid_failure(3, 40), _paid_failure(3, 40)]),
        clock=lambda: NOW,
    )
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.decision.cost.llm_calls == 9
    assert response.decision.cost.input_tokens == 120
    assert response.decision.cost.output_tokens == 12
    assert response.decision.cost.estimated_cost_usd == pytest.approx(0.03)
    assert response.role_errors[0].error_type == "RoleOutputInvalid"


@pytest.mark.asyncio
async def test_failed_member_usage_is_kept_on_escalate(
    decide_request: DecideRequest,
) -> None:
    success = make_judge(decide_request.candidate).model_copy(
        update={"llm_calls": 1, "input_tokens": 10, "assessment_id": uuid4()}
    )
    baseline = GenericEnsembleBaseline(
        _ScriptedJudge([_paid_failure(3, 100), _paid_failure(3, 100), success]),
        clock=lambda: NOW,
    )
    response = await baseline.decide(decide_request)
    assert response.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert response.judge is None
    assert response.decision.cost.llm_calls == 7
    assert response.decision.cost.input_tokens == 210
    assert response.role_errors[0].llm_calls == 7
    assert response.role_errors[0].input_tokens == 210


@pytest.mark.asyncio
async def test_two_agreeing_votes_include_the_failed_member(
    decide_request: DecideRequest,
) -> None:
    first = make_judge(decide_request.candidate).model_copy(update={"assessment_id": uuid4()})
    second = make_judge(decide_request.candidate).model_copy(update={"assessment_id": uuid4()})
    ensemble = GenericEnsembleJudge(
        _ScriptedJudge([first, _paid_failure(3, 50), second]),
        n_members=3,
    )
    assessment = await ensemble.evaluate(
        JudgeRequest(
            candidate=decide_request.candidate,
            evidence=decide_request.evidence,
            run_config=decide_request.run_config,
        )
    )
    assert assessment.label is JudgeLabel.SUPPORTED
    assert assessment.confidence == pytest.approx(2 / 3)
    assert assessment.llm_calls == 3
    assert assessment.input_tokens == 50


@pytest.mark.asyncio
async def test_split_panel_does_not_promote(decide_request: DecideRequest) -> None:
    labels = [JudgeLabel.SUPPORTED, JudgeLabel.REFUTED, JudgeLabel.MIXED_EVIDENCE]
    members = [
        make_judge(decide_request.candidate, label=label).model_copy(
            update={"assessment_id": uuid4()}
        )
        for label in labels
    ]
    ensemble = GenericEnsembleJudge(_ScriptedJudge(members), n_members=3)
    assessment = await ensemble.evaluate(
        JudgeRequest(
            candidate=decide_request.candidate,
            evidence=decide_request.evidence,
            run_config=decide_request.run_config,
        )
    )
    assert assessment.label is JudgeLabel.INSUFFICIENT_EVIDENCE
    assert assessment.confidence == 0


class _SlowMember:
    def __init__(self, assessment: JudgeAssessment, delay_s: float) -> None:
        self._assessment = assessment
        self._delay_s = delay_s
        self.calls = 0

    async def evaluate(self, _request: JudgeRequest) -> JudgeAssessment:
        self.calls += 1
        await asyncio.sleep(self._delay_s)
        return self._assessment.model_copy(update={"assessment_id": uuid4()})


@pytest.mark.asyncio
async def test_members_share_one_timeout_concurrently(decide_request: DecideRequest) -> None:
    member = _SlowMember(make_judge(decide_request.candidate), delay_s=0.3)
    ensemble = GenericEnsembleJudge(member, n_members=3)
    request = JudgeRequest(
        candidate=decide_request.candidate,
        evidence=decide_request.evidence,
        run_config=decide_request.run_config.model_copy(update={"timeout_s": 2.0}),
    )
    started = time.perf_counter()
    assessment = await ensemble.evaluate(request)
    elapsed = time.perf_counter() - started
    assert assessment.label is JudgeLabel.SUPPORTED
    assert member.calls == 3
    # Three 0.3s calls in sequence cannot finish in under 0.75s.
    assert elapsed < 0.75


@pytest.mark.asyncio
async def test_overall_timeout_keeps_completed_usage(decide_request: DecideRequest) -> None:
    class _HangAfterOne:
        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, _request: JudgeRequest) -> JudgeAssessment:
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(0.05)
                return make_judge(decide_request.candidate).model_copy(
                    update={"llm_calls": 2, "input_tokens": 9, "assessment_id": uuid4()}
                )
            await asyncio.sleep(5)
            raise AssertionError("the overall timeout should cancel this member")

    ensemble = GenericEnsembleJudge(_HangAfterOne(), n_members=3)
    request = JudgeRequest(
        candidate=decide_request.candidate,
        evidence=decide_request.evidence,
        run_config=decide_request.run_config.model_copy(update={"timeout_s": 0.2}),
    )
    with pytest.raises(RoleTimeout, match="no operational quorum") as raised:
        await ensemble.evaluate(request)
    usage = raised.value.usage
    assert isinstance(usage, CallUsage)
    assert usage.llm_calls == 2
    assert usage.input_tokens == 9
