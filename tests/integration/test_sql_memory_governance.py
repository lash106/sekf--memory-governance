"""End to end with the persistent store: orchestrator -> policy -> SqlMemoryRepository."""

from __future__ import annotations

import asyncio
from typing import Any

from tests.conftest import NOW

from sekf.api import Governance
from sekf.contracts import (
    ClaimStatus,
    DecideMode,
    DecideRequest,
    ExistingClaim,
    FinalAction,
)
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.memory import SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def _setup(lab_fixture: dict[str, Any]) -> tuple[Governance, SqlMemoryRepository, DecideRequest]:
    repo = SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW)
    repo.seed_claims([ExistingClaim.model_validate(c) for c in lab_fixture["existing_claims"]])
    gov = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    request = DecideRequest.model_validate({**lab_fixture["request"], "mode": "commit"})
    return gov, repo, request


def test_worked_example_commit_revises_memory(lab_fixture: dict[str, Any]) -> None:
    gov, repo, request = _setup(lab_fixture)
    old_id = ExistingClaim.model_validate(lab_fixture["existing_claims"][0]).claim_id
    new_id = request.candidate.claim_id

    resp = asyncio.run(gov.decide(request))
    assert resp.decision.final_action is FinalAction.REVISE
    assert resp.memory_delta is not None and resp.memory_delta.applied

    old, new = repo.get_claim(old_id), repo.get_claim(new_id)
    assert old is not None and old.status is ClaimStatus.SUPERSEDED
    assert new is not None and new.status is ClaimStatus.ACTIVE and new.version == 2
    assert new.supersedes_claim_id == old_id

    # A retry is answered from the stored bundle, even though memory has changed.
    retry = asyncio.run(gov.decide(request))
    assert retry.memory_delta is not None and retry.memory_delta.replayed
    assert retry.decision == resp.decision
    assert repo.decision_count() == 1


def test_dry_run_leaves_store_untouched(lab_fixture: dict[str, Any]) -> None:
    gov, repo, request = _setup(lab_fixture)
    dry = request.model_copy(update={"mode": DecideMode.DRY_RUN})
    asyncio.run(gov.decide(dry))
    assert repo.decision_count() == 0
    assert repo.get_claim(request.candidate.claim_id) is None
