"""Offline tests for generic_ensemble majority vote and baseline wiring."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from tests.conftest import NOW

from sekf.baselines import GenericEnsembleBaseline, GenericEnsembleJudge, majority_judge_label
from sekf.contracts import DecideMode, DecideRequest, FinalAction, JudgeLabel, JudgeRequest
from sekf.errors import CommitRejected
from sekf.judge import LLMJudge
from sekf.memory import InMemoryRepository
from sekf.providers import FakeProvider

EVIDENCE_ID = UUID("33333333-3333-4333-8333-333333333333")


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
