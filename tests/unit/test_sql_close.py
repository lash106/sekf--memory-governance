"""SqlMemoryRepository.close() and the application shutdown that calls it."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import make_candidate, make_existing

from sekf.api import Governance, create_app
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.memory import SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def test_close_is_idempotent_and_the_store_stays_usable(tmp_path: Path) -> None:
    repo = SqlMemoryRepository.from_url(f"sqlite:///{tmp_path / 'm.sqlite3'}")
    claim = make_existing()
    repo.seed_claims([claim])
    repo.close()
    repo.close()  # a second call is harmless
    assert repo.get_claim(claim.claim_id) == claim  # fresh connections open on demand
    assert repo.search_related_claims(make_candidate()) == [claim]
    repo.close()


def _governance(memory: SqlMemoryRepository) -> Governance:
    return Governance(
        memory=memory,
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )


def test_governance_aclose_closes_the_sql_store(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = SqlMemoryRepository.from_url("sqlite://")
    closed: list[bool] = []
    monkeypatch.setattr(repo, "close", lambda: closed.append(True))
    asyncio.run(_governance(repo).aclose())
    assert closed == [True]


def test_app_shutdown_closes_the_sql_store(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = SqlMemoryRepository.from_url("sqlite://")
    closed: list[bool] = []
    monkeypatch.setattr(repo, "close", lambda: closed.append(True))
    with TestClient(create_app(_governance(repo))):
        assert closed == []
    assert closed == [True]
