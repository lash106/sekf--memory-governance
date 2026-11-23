"""The storage and retrieval baselines pick their fixed actions and write only on commit."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.conftest import NOW

from sekf.baselines import DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline
from sekf.baselines.static_rag import retrieve
from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    DecideMode,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
)
from sekf.memory import SqlMemoryRepository

OLD = NOW - timedelta(days=60)


def _claim(
    text: str = "The lab closes at 17:00.",
    obj: str = "17:00",
    *,
    status: ClaimStatus = ClaimStatus.ACTIVE,
    updated: datetime = OLD,
) -> ExistingClaim:
    return ExistingClaim(
        claim_id=uuid4(),
        claim_text=text,
        subject_key="lab",
        predicate_key="closing_time",
        object_text=obj,
        status=status,
        version=1,
        confidence=0.9,
        created_at=updated,
        updated_at=updated,
    )


def _candidate(observed: datetime = NOW, text: str = "The lab closes at 18:00.") -> CandidateClaim:
    return CandidateClaim(
        claim_id=uuid4(),
        claim_text=text,
        subject_key="lab",
        predicate_key="closing_time",
        object_text="18:00",
        source_id="chat",
        observed_at=observed,
    )


def _evidence(content: str = "Lab hours end at 18:00.") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=uuid4(),
        content=content,
        source_id="site",
        source_type="official_document",
        retrieved_at=NOW,
        reliability=0.9,
    )


def _store(*claims: ExistingClaim) -> SqlMemoryRepository:
    repo = SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW)
    repo.seed_claims(list(claims))
    return repo


def _run(system: object, request: DecideRequest):  # type: ignore[no-untyped-def]
    return asyncio.run(system.decide(request))  # type: ignore[attr-defined]


def _req(cand: CandidateClaim, mode: DecideMode = DecideMode.DRY_RUN, **kw) -> DecideRequest:  # type: ignore[no-untyped-def]
    return DecideRequest(candidate=cand, evidence=[_evidence()], mode=mode, **kw)


# ------------------------------------------------------------------ direct_storage


def test_direct_storage_promotes_even_when_an_active_claim_disagrees() -> None:
    old = _claim()
    repo = _store(old)
    cand = _candidate()
    resp = _run(DirectStorageBaseline(repo, clock=lambda: NOW), _req(cand, DecideMode.COMMIT))
    assert resp.decision.final_action is FinalAction.PROMOTE
    assert resp.decision.rule_ids_fired == ["DIRECT_STORAGE"]
    stored_old, stored_new = repo.get_claim(old.claim_id), repo.get_claim(cand.claim_id)
    assert stored_old is not None and stored_old.status is ClaimStatus.ACTIVE
    assert stored_new is not None and stored_new.status is ClaimStatus.ACTIVE
    # Two conflicting ACTIVE claims for one key: the failure this baseline exists to show.
    assert len(repo.search_related_claims(_candidate())) == 2


# ----------------------------------------------------------------- latest_write_wins


def test_lww_promotes_when_nothing_is_active() -> None:
    repo = _store(_claim(status=ClaimStatus.QUARANTINED))
    resp = _run(LatestWriteWinsBaseline(repo, clock=lambda: NOW), _req(_candidate()))
    assert resp.decision.final_action is FinalAction.PROMOTE


def test_lww_revises_when_candidate_is_newer() -> None:
    old = _claim()
    repo = _store(old)
    cand = _candidate()
    resp = _run(LatestWriteWinsBaseline(repo, clock=lambda: NOW), _req(cand, DecideMode.COMMIT))
    assert resp.decision.final_action is FinalAction.REVISE
    assert resp.decision.previous_claim_ids == [old.claim_id]
    stored_old, stored_new = repo.get_claim(old.claim_id), repo.get_claim(cand.claim_id)
    assert stored_old is not None and stored_old.status is ClaimStatus.SUPERSEDED
    assert stored_new is not None and stored_new.version == 2


def test_lww_keeps_episodic_when_candidate_is_older() -> None:
    repo = _store(_claim(updated=NOW))
    cand = _candidate(observed=NOW - timedelta(days=1))
    resp = _run(LatestWriteWinsBaseline(repo, clock=lambda: NOW), _req(cand, DecideMode.COMMIT))
    assert resp.decision.final_action is FinalAction.KEEP_EPISODIC
    assert repo.get_claim(cand.claim_id) is None  # no claim is created


def test_lww_targets_the_most_recently_updated_active_claim() -> None:
    older, newer = _claim(updated=OLD), _claim(obj="16:00", updated=OLD + timedelta(days=30))
    repo = _store(older, newer)
    resp = _run(LatestWriteWinsBaseline(repo, clock=lambda: NOW), _req(_candidate()))
    assert resp.decision.previous_claim_ids == [newer.claim_id]


def test_lww_ignores_a_candidate_that_is_not_strictly_newer() -> None:
    repo = _store(_claim(updated=NOW))
    resp = _run(LatestWriteWinsBaseline(repo, clock=lambda: NOW), _req(_candidate(observed=NOW)))
    assert resp.decision.final_action is FinalAction.KEEP_EPISODIC


# ---------------------------------------------------------------------- static_rag


def test_static_rag_never_writes_governed_memory() -> None:
    old = _claim()
    repo = _store(old)
    cand = _candidate()
    resp = _run(StaticRagBaseline(repo, clock=lambda: NOW), _req(cand, DecideMode.COMMIT))
    assert resp.decision.final_action is FinalAction.KEEP_EPISODIC
    assert repo.get_claim(cand.claim_id) is None
    stored_old = repo.get_claim(old.claim_id)
    assert stored_old is not None and stored_old.status is ClaimStatus.ACTIVE
    assert str(old.claim_id) in resp.decision.explanation  # retrieval is recorded


def test_static_rag_retrieval_is_ranked_and_deterministic() -> None:
    close = _claim("The lab closes at 18:00 on weekdays.")
    far = _claim("Parking is free.", obj="free")
    cand = _candidate(text="The lab closes at 18:00 on weekdays.")
    hits = retrieve(cand, [far, close], [], top_k=3)
    assert [h.item_id for h in hits] == [close.claim_id]  # zero-overlap items are dropped
    assert hits == retrieve(cand, [close, far], [], top_k=3)
    assert retrieve(cand, [close, far], [], top_k=0) == []


def test_static_rag_retrieves_evidence_too() -> None:
    ev = _evidence("The lab closes at 18:00.")
    hits = retrieve(_candidate(), [], [ev])
    assert [(h.kind, h.item_id) for h in hits] == [("evidence", ev.evidence_id)]


# ----------------------------------------------------------- shared commit behaviour


@pytest.mark.parametrize("cls", [DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline])
def test_dry_run_never_writes(cls: type) -> None:
    repo = _store(_claim())
    resp = _run(cls(repo, clock=lambda: NOW), _req(_candidate()))
    assert resp.mode is DecideMode.DRY_RUN and resp.memory_delta is None
    assert repo.decision_count() == 0


@pytest.mark.parametrize("cls", [DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline])
def test_commit_is_idempotent_and_costs_nothing(cls: type) -> None:
    repo = _store(_claim())
    system = cls(repo, clock=lambda: NOW)
    request = _req(_candidate(), DecideMode.COMMIT)
    first = _run(system, request)
    retry = _run(system, request)
    assert first.memory_delta is not None and not first.memory_delta.replayed
    assert retry.memory_delta is not None and retry.memory_delta.replayed
    assert retry.decision == first.decision
    assert repo.decision_count() == 1
    assert first.decision.cost.llm_calls == 0 and first.baseline_audit is None


def test_decision_id_differs_between_baselines() -> None:
    # The same request under different policy versions must not collide in a shared store.
    request = _req(_candidate())
    repo = _store()
    ids = {
        _run(c(repo, clock=lambda: NOW), request).decision.decision_id
        for c in (DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline)
    }
    assert len(ids) == 3


def test_utc_clock_default_is_aware() -> None:
    repo = _store()
    resp = _run(DirectStorageBaseline(repo), _req(_candidate()))
    assert resp.decision.created_at.tzinfo is UTC
