"""SqlMemoryRepository: state transitions, idempotency, staleness, rollback (ADR 0001, 0004)."""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from tests.conftest import (
    NOW,
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
)

from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    CommitBundle,
    ConflictType,
    EpisodeRecord,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeLabel,
    RelationshipType,
    TemporalRelation,
)
from sekf.errors import CommitRejected
from sekf.memory import SqlMemoryRepository, build_memory_repository, create_memory_engine
from sekf.memory.migrate import alembic_config, run_migrations
from sekf.memory.schema import Base
from sekf.policy import PolicyEngine

ENGINE = PolicyEngine()


@pytest.fixture
def repo() -> SqlMemoryRepository:
    return SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW)


def make_evidence(content: str = "Official schedule: Fridays close at 18:00.") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=uuid4(),
        content=content,
        source_id="official-schedule",
        source_type="official_document",
        published_at=NOW - timedelta(days=3),
        retrieved_at=NOW,
        reliability=0.95,
    )


def bundle_for(
    repo: SqlMemoryRepository,
    cand: CandidateClaim | None = None,
    *,
    evidence: list[EvidenceRecord] | None = None,
    judge: dict[str, Any] | None = None,
    historian: dict[str, Any] | None = None,
    key: str | None = None,
    episode: EpisodeRecord | None = None,
) -> CommitBundle:
    """Run the real policy on crafted role outputs against the repository's current memory."""
    cand = cand or make_candidate()
    evidence = [make_evidence()] if evidence is None else evidence
    related = repo.search_related_claims(cand)
    h = make_historian(cand, **(historian or {}))
    j = make_judge(cand, **(judge or {}))
    decision = ENGINE.decide(
        make_context(
            candidate=cand,
            evidence=evidence,
            related_claims=related,
            historian=h,
            judge=j,
            idempotency_key=key,
        )
    )
    return CommitBundle(
        episode=episode,
        candidate=cand,
        evidence=evidence,
        decision=decision,
        historian=h,
        judge=j,
        memory_before=related,
        idempotency_key=key,
    )


def seeded(repo: SqlMemoryRepository, status: ClaimStatus = ClaimStatus.ACTIVE) -> ExistingClaim:
    claim = make_existing(status)
    repo.seed_claims([claim])
    return claim


def revise_kwargs(target: ExistingClaim) -> dict[str, Any]:
    return {
        "conflict": ConflictType.TEMPORAL_UPDATE,
        "temporal": TemporalRelation.NEWER_UPDATE,
        "related": [target.claim_id],
        "supersedes": target.claim_id,
    }


# --------------------------------------------------------------------------- schema


def test_migrated_schema_matches_models(repo: SqlMemoryRepository) -> None:
    with repo.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_migrations_are_idempotent_and_reversible(tmp_path: Path) -> None:
    from alembic import command

    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    run_migrations(url)
    run_migrations(url)  # second upgrade is a no-op
    command.downgrade(alembic_config(url), "base")
    engine = create_memory_engine(url)
    with engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master"))}
    assert "claims" not in tables
    run_migrations(url)


def test_store_persists_across_instances(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'nested' / 'sekf.sqlite3'}"
    first = build_memory_repository(url)
    bundle = bundle_for(first)
    first.apply_decision(bundle)
    second = build_memory_repository(url)
    claim = second.get_claim(bundle.candidate.claim_id)
    assert claim is not None and claim.status is ClaimStatus.ACTIVE
    assert second.get_bundle(bundle.decision.decision_id) == bundle


def test_build_memory_repository_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SEKF_DB_URL", f"sqlite:///{tmp_path / 'env.sqlite3'}")
    build_memory_repository()
    assert (tmp_path / "env.sqlite3").exists()


# ---------------------------------------------------------------------------- reads


def test_search_returns_same_keys_any_status_excluding_candidate(
    repo: SqlMemoryRepository,
) -> None:
    active = make_existing()
    superseded = make_existing(ClaimStatus.SUPERSEDED)
    other = make_existing()
    other.subject_key = "library"
    repo.seed_claims([active, superseded, other])
    cand = make_candidate()
    found = {c.claim_id for c in repo.search_related_claims(cand)}
    assert found == {active.claim_id, superseded.claim_id}

    same_id = cand.model_copy(update={"claim_id": active.claim_id})
    assert {c.claim_id for c in repo.search_related_claims(same_id)} == {superseded.claim_id}


def test_reads_return_copies(repo: SqlMemoryRepository) -> None:
    claim = seeded(repo)
    got = repo.get_claim(claim.claim_id)
    assert got == claim
    assert got is not None
    got.status = ClaimStatus.REJECTED
    assert repo.get_claim(claim.claim_id) == claim
    assert repo.get_claim(uuid4()) is None


def test_seed_rejects_duplicate_ids(repo: SqlMemoryRepository) -> None:
    claim = seeded(repo)
    with pytest.raises(ValueError):
        repo.seed_claims([claim])


# ---------------------------------------------------------------------- transitions


def test_promote_creates_active_claim_with_evidence(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo)
    assert bundle.decision.final_action is FinalAction.PROMOTE
    delta = repo.apply_decision(bundle)
    cid = bundle.candidate.claim_id
    assert delta.applied and delta.created_claim_ids == [cid]
    assert [(c.claim_id, c.new_status) for c in delta.status_changes] == [(cid, ClaimStatus.ACTIVE)]
    claim = repo.get_claim(cid)
    assert claim is not None
    assert claim.status is ClaimStatus.ACTIVE and claim.version == 1
    assert claim.evidence_ids == [bundle.evidence[0].evidence_id]
    assert claim.confidence == pytest.approx(0.9)


def test_revise_supersedes_and_preserves_versions(repo: SqlMemoryRepository) -> None:
    old = seeded(repo)
    bundle = bundle_for(repo, historian=revise_kwargs(old))
    assert bundle.decision.final_action is FinalAction.REVISE
    delta = repo.apply_decision(bundle)
    new_id = bundle.candidate.claim_id

    old_after = repo.get_claim(old.claim_id)
    new = repo.get_claim(new_id)
    assert old_after is not None and new is not None
    assert old_after.status is ClaimStatus.SUPERSEDED  # kept, never deleted
    assert old_after.object_text == old.object_text
    assert new.status is ClaimStatus.ACTIVE
    assert new.version == old.version + 1 and new.supersedes_claim_id == old.claim_id
    assert [c.claim_id for c in repo.get_claim_history(new_id)] == [new_id, old.claim_id]
    assert (new_id, old.claim_id, RelationshipType.SUPERSEDES) in repo.get_relationships(new_id)
    assert {(c.claim_id, c.new_status) for c in delta.status_changes} == {
        (old.claim_id, ClaimStatus.SUPERSEDED),
        (new_id, ClaimStatus.ACTIVE),
    }


def test_attach_evidence_adds_links_without_new_version(repo: SqlMemoryRepository) -> None:
    target = seeded(repo)
    kwargs = {"conflict": ConflictType.DUPLICATE, "related": [target.claim_id]}
    bundle = bundle_for(repo, historian=kwargs)
    assert bundle.decision.final_action is FinalAction.ATTACH_EVIDENCE
    delta = repo.apply_decision(bundle)
    ev = bundle.evidence[0].evidence_id
    after = repo.get_claim(target.claim_id)
    assert after is not None
    assert after.version == target.version and after.status is ClaimStatus.ACTIVE
    assert after.evidence_ids == [ev]
    assert delta.attached_evidence_ids == [ev] and delta.created_claim_ids == []
    assert repo.get_claim(bundle.candidate.claim_id) is None  # no new claim
    assert (
        bundle.candidate.claim_id,
        target.claim_id,
        RelationshipType.DUPLICATES,
    ) in repo.get_relationships(target.claim_id)

    # A second paraphrase with the same evidence attaches nothing new.
    again = bundle_for(repo, historian=kwargs, evidence=bundle.evidence)
    delta2 = repo.apply_decision(again)
    assert delta2.attached_evidence_ids == [] and not delta2.applied


def test_quarantine_leaves_active_knowledge_unchanged(repo: SqlMemoryRepository) -> None:
    active = seeded(repo)
    bundle = bundle_for(
        repo, historian={"conflict": ConflictType.VALUE_CONFLICT, "related": [active.claim_id]}
    )
    assert bundle.decision.final_action is FinalAction.QUARANTINE
    repo.apply_decision(bundle)
    assert repo.get_claim(active.claim_id) == active
    cand = repo.get_claim(bundle.candidate.claim_id)
    assert cand is not None and cand.status is ClaimStatus.QUARANTINED
    assert (
        bundle.candidate.claim_id,
        active.claim_id,
        RelationshipType.CONTRADICTS,
    ) in repo.get_relationships(active.claim_id)


def test_reject_keeps_claim_for_audit(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo, judge={"label": JudgeLabel.REFUTED})
    assert bundle.decision.final_action is FinalAction.REJECT
    repo.apply_decision(bundle)
    claim = repo.get_claim(bundle.candidate.claim_id)
    assert claim is not None and claim.status is ClaimStatus.REJECTED


def test_keep_episodic_stores_inputs_only(repo: SqlMemoryRepository) -> None:
    episode = EpisodeRecord(
        episode_id=uuid4(), content="chat log", source_id="student-chat", observed_at=NOW
    )
    bundle = bundle_for(repo, judge={"label": JudgeLabel.INSUFFICIENT_EVIDENCE}, episode=episode)
    assert bundle.decision.final_action is FinalAction.KEEP_EPISODIC
    delta = repo.apply_decision(bundle)
    assert not delta.applied and delta.created_claim_ids == []
    assert repo.get_claim(bundle.candidate.claim_id) is None
    trace = repo.get_audit_trace(bundle.candidate.claim_id)
    assert [e.bundle for e in trace.entries] == [bundle]
    assert trace.entries[0].memory_delta == delta


def test_escalate_keeps_candidate_and_queues_review(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo, historian={"confidence": 0.3})
    assert bundle.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    repo.apply_decision(bundle)
    claim = repo.get_claim(bundle.candidate.claim_id)
    assert claim is not None and claim.status is ClaimStatus.CANDIDATE
    queue = repo.list_review_queue()
    assert [(d, c) for d, c, _ in queue] == [
        (bundle.decision.decision_id, bundle.candidate.claim_id)
    ]


def test_quarantined_claim_can_be_promoted_on_reevaluation(repo: SqlMemoryRepository) -> None:
    cand = make_candidate()
    first = bundle_for(repo, cand, historian={"reliability": 0.2})
    assert first.decision.final_action is FinalAction.QUARANTINE
    repo.apply_decision(first)
    second = bundle_for(repo, cand, key="re-run-1")
    assert second.decision.final_action is FinalAction.PROMOTE
    delta = repo.apply_decision(second)
    claim = repo.get_claim(cand.claim_id)
    assert claim is not None and claim.status is ClaimStatus.ACTIVE
    assert delta.created_claim_ids == []  # existing row transitioned, not duplicated
    assert len(repo.get_audit_trace(cand.claim_id).entries) == 2


def test_governed_claim_cannot_be_redecided(repo: SqlMemoryRepository) -> None:
    cand = make_candidate()
    repo.apply_decision(bundle_for(repo, cand))
    rerun = bundle_for(repo, cand, key="again", judge={"label": JudgeLabel.REFUTED})
    with pytest.raises(CommitRejected, match="already ACTIVE"):
        repo.apply_decision(rerun)
    assert repo.decision_count() == 1


# --------------------------------------------------------------- idempotency & audit


def test_replay_is_idempotent(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo)
    first = repo.apply_decision(bundle)
    again = repo.apply_decision(bundle)
    assert again.replayed and not again.applied and not first.replayed
    assert repo.decision_count() == 1


def test_retry_after_committed_revise_is_a_replay(repo: SqlMemoryRepository) -> None:
    old = seeded(repo)
    bundle = bundle_for(repo, historian=revise_kwargs(old))
    repo.apply_decision(bundle)
    # Memory changed (old claim SUPERSEDED), but the same request is still recognized.
    assert repo.apply_decision(bundle).replayed
    assert repo.get_bundle(bundle.decision.decision_id) == bundle
    assert repo.decision_count() == 1


def test_reused_decision_id_with_different_inputs_is_rejected(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo)
    repo.apply_decision(bundle)
    with repo.engine.begin() as conn:
        conn.execute(text("UPDATE decisions SET request_hash = :h"), {"h": "0" * 64})
    with pytest.raises(CommitRejected, match="different inputs"):
        repo.apply_decision(bundle)


def test_audit_trace_keeps_original_bundle_and_delta(repo: SqlMemoryRepository) -> None:
    old = seeded(repo)
    bundle = bundle_for(repo, historian=revise_kwargs(old))
    delta = repo.apply_decision(bundle)
    (entry,) = repo.get_audit_trace(bundle.candidate.claim_id).entries
    assert entry.bundle == bundle and entry.bundle.memory_before == [old]
    assert entry.memory_delta == delta


def test_inconsistent_bundle_writes_nothing(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo)
    with pytest.raises(CommitRejected):
        repo.apply_decision(bundle.model_copy(update={"judge": None}))
    assert repo.decision_count() == 0
    assert repo.get_claim(bundle.candidate.claim_id) is None


def test_decision_links_must_match_action(repo: SqlMemoryRepository) -> None:
    bundle = bundle_for(repo)
    bad = bundle.decision.model_copy(update={"previous_claim_ids": [uuid4()]})
    with pytest.raises(CommitRejected, match="inconsistent claim links"):
        repo.apply_decision(bundle.model_copy(update={"decision": bad}))
    assert repo.decision_count() == 0


def test_evidence_id_reuse_with_new_content_is_rejected(repo: SqlMemoryRepository) -> None:
    ev = make_evidence()
    repo.apply_decision(bundle_for(repo, evidence=[ev]))
    forged = ev.model_copy(update={"content": "different", "content_hash": ""})
    forged = EvidenceRecord.model_validate(forged.model_dump())
    other = make_candidate()
    other.subject_key = "library"
    with pytest.raises(CommitRejected, match="different contents"):
        repo.apply_decision(bundle_for(repo, other, evidence=[forged]))
    assert repo.decision_count() == 1


# ------------------------------------------------------------- staleness & rollback


def test_stale_snapshot_changed_claim_is_rejected(repo: SqlMemoryRepository) -> None:
    old = seeded(repo)
    bundle = bundle_for(repo, historian=revise_kwargs(old))
    # A competing decision supersedes the same claim first.
    repo.apply_decision(bundle_for(repo, historian=revise_kwargs(old)))
    with pytest.raises(CommitRejected, match="memory changed"):
        repo.apply_decision(bundle)
    assert repo.get_claim(bundle.candidate.claim_id) is None


def test_stale_snapshot_phantom_claim_is_rejected(repo: SqlMemoryRepository) -> None:
    first = bundle_for(repo)
    second = bundle_for(repo)  # both read an empty memory
    repo.apply_decision(first)
    with pytest.raises(CommitRejected, match="memory changed"):
        repo.apply_decision(second)
    assert repo.decision_count() == 1


def test_failure_mid_commit_rolls_back_everything(
    repo: SqlMemoryRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = seeded(repo)
    bundle = bundle_for(repo, historian=revise_kwargs(old))

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(SqlMemoryRepository, "_relate", staticmethod(boom))
    with pytest.raises(RuntimeError):
        repo.apply_decision(bundle)
    monkeypatch.undo()

    assert repo.get_claim(old.claim_id) == old  # not SUPERSEDED
    assert repo.get_claim(bundle.candidate.claim_id) is None
    assert repo.decision_count() == 0
    assert repo.get_audit_trace(bundle.candidate.claim_id).entries == []
    # Memory untouched, so the same bundle commits cleanly on retry.
    assert repo.apply_decision(bundle).applied


def test_concurrent_promotes_from_same_snapshot_commit_once(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'race.sqlite3'}"
    repos = [SqlMemoryRepository.from_url(url) for _ in range(2)]
    bundles = [bundle_for(repos[0]) for _ in range(2)]  # both see empty memory
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def run(i: int) -> None:
        barrier.wait()
        try:
            repos[i].apply_decision(bundles[i])
            outcomes.append("ok")
        except CommitRejected:
            outcomes.append("rejected")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok", "rejected"]
    active = [
        c
        for c in repos[0].search_related_claims(make_candidate())
        if c.status is ClaimStatus.ACTIVE
    ]
    assert len(active) == 1


def test_naive_timestamps_are_refused(repo: SqlMemoryRepository) -> None:
    claim = make_existing()
    with pytest.raises(Exception, match="timezone-aware"):
        with repo.engine.begin() as conn:
            conn.execute(
                Base.metadata.tables["claims"].insert(),
                {
                    **claim.model_dump(exclude={"evidence_ids"}),
                    "claim_id": str(UUID(int=1)),
                    "status": "ACTIVE",
                    "supersedes_claim_id": None,
                    "created_at": NOW.replace(tzinfo=None),
                    "row_version": 1,
                },
            )


# ------------------------------------------------------------ review regressions


def test_escalating_a_governed_claim_attaches_nothing_to_it(repo: SqlMemoryRepository) -> None:
    """Regression: an escalation must not link new evidence to an already-ACTIVE claim."""
    cand = make_candidate()
    repo.apply_decision(bundle_for(repo, cand))
    before = repo.get_claim(cand.claim_id)
    escalate = bundle_for(repo, cand, key="re-run", historian={"confidence": 0.3})
    assert escalate.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    delta = repo.apply_decision(escalate)
    assert repo.get_claim(cand.claim_id) == before
    assert not delta.applied
    assert [c for _, c, _ in repo.list_review_queue()] == [cand.claim_id]


def test_reevaluated_candidate_must_match_stored_claim(repo: SqlMemoryRepository) -> None:
    """Regression: the same claim_id with different content is a data error, not a re-run."""
    cand = make_candidate()
    repo.apply_decision(bundle_for(repo, cand, historian={"reliability": 0.2}))  # QUARANTINE
    changed = cand.model_copy(update={"object_text": "19:00"})
    with pytest.raises(CommitRejected, match="different content"):
        repo.apply_decision(bundle_for(repo, changed, key="re-run"))
    assert repo.decision_count() == 1


def test_rejected_claim_confidence_is_truth_confidence(repo: SqlMemoryRepository) -> None:
    """Regression: a REJECTED claim stored the Judge's refutation confidence as if it were true."""
    bundle = bundle_for(repo, judge={"label": JudgeLabel.REFUTED, "confidence": 0.9})
    repo.apply_decision(bundle)
    claim = repo.get_claim(bundle.candidate.claim_id)
    assert claim is not None and claim.confidence == pytest.approx(0.1)


def test_claim_history_stops_on_supersedes_cycle(repo: SqlMemoryRepository) -> None:
    a, b = make_existing(), make_existing(ClaimStatus.SUPERSEDED)
    repo.seed_claims([b])
    repo.seed_claims([a.model_copy(update={"supersedes_claim_id": b.claim_id})])
    with repo.engine.begin() as conn:  # corrupt data: b -> a -> b
        conn.execute(
            text("UPDATE claims SET supersedes_claim_id = :a WHERE claim_id = :b"),
            {"a": str(a.claim_id), "b": str(b.claim_id)},
        )
    assert [c.claim_id for c in repo.get_claim_history(a.claim_id)] == [a.claim_id, b.claim_id]


# ------------------------------------------------- PR #9 review (Shailen) regressions


@pytest.mark.parametrize(
    ("kwargs", "action", "status"),
    [
        ({}, FinalAction.PROMOTE, ClaimStatus.ACTIVE),
        ({"historian": {"reliability": 0.2}}, FinalAction.QUARANTINE, ClaimStatus.QUARANTINED),
        ({"judge": {"label": JudgeLabel.REFUTED}}, FinalAction.REJECT, ClaimStatus.REJECTED),
        (
            {"historian": {"confidence": 0.3}},
            FinalAction.ESCALATE_FOR_REVIEW,
            ClaimStatus.CANDIDATE,
        ),
    ],
)
def test_claim_creating_actions_with_evidence_commit_replay_and_roll_back(
    repo: SqlMemoryRepository,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    action: FinalAction,
    status: ClaimStatus,
) -> None:
    """Review item 1: every claim-creating action links its evidence exactly once."""
    evidence = [make_evidence(), make_evidence("Second source agrees.")]
    bundle = bundle_for(repo, evidence=evidence, **kwargs)
    assert bundle.decision.final_action is action

    original = SqlMemoryRepository._link_evidence

    def link_then_crash(*args: Any, **kw: Any) -> list[str]:
        original(*args, **kw)  # links are written...
        raise RuntimeError("crash after links")  # ...then the commit dies

    monkeypatch.setattr(SqlMemoryRepository, "_link_evidence", staticmethod(link_then_crash))
    with pytest.raises(RuntimeError):
        repo.apply_decision(bundle)
    monkeypatch.undo()
    assert repo.decision_count() == 0 and repo.get_claim(bundle.candidate.claim_id) is None

    repo.apply_decision(bundle)
    claim = repo.get_claim(bundle.candidate.claim_id)
    assert claim is not None and claim.status is status
    assert claim.evidence_ids == [e.evidence_id for e in evidence]
    assert repo.apply_decision(bundle).replayed


def test_revise_with_evidence_links_it_to_the_new_version(repo: SqlMemoryRepository) -> None:
    """Review item 1, REVISE path."""
    old = seeded(repo)
    evidence = [make_evidence(), make_evidence("Memo confirms 18:00.")]
    bundle = bundle_for(repo, evidence=evidence, historian=revise_kwargs(old))
    repo.apply_decision(bundle)
    new = repo.get_claim(bundle.candidate.claim_id)
    assert new is not None and new.evidence_ids == [e.evidence_id for e in evidence]
    assert repo.apply_decision(bundle).replayed


def test_one_episode_can_produce_several_candidate_decisions(repo: SqlMemoryRepository) -> None:
    """Review item 2: an identical episode is reused, not re-inserted."""
    episode = EpisodeRecord(
        episode_id=uuid4(), content="meeting notes", source_id="notes", observed_at=NOW
    )
    first = make_candidate()
    second = make_candidate()
    second.subject_key = "library"
    repo.apply_decision(bundle_for(repo, first, episode=episode))
    repo.apply_decision(bundle_for(repo, second, episode=episode))
    assert repo.decision_count() == 2

    changed = episode.model_copy(update={"content": "edited notes"})
    third = make_candidate()
    third.subject_key = "gym"
    with pytest.raises(CommitRejected, match="episode .* different contents"):
        repo.apply_decision(bundle_for(repo, third, episode=changed))
    assert repo.decision_count() == 2


def test_seeded_claims_keep_evidence_links(repo: SqlMemoryRepository) -> None:
    """Review item 3: seeds may cite external evidence IDs or bring the records along."""
    external = make_existing()
    external.evidence_ids = [uuid4()]
    stored_ev = make_evidence()
    with_record = make_existing()
    with_record.subject_key = "library"
    with_record.evidence_ids = [stored_ev.evidence_id]
    repo.seed_claims([external, with_record], evidence=[stored_ev])
    assert repo.get_claim(external.claim_id) == external
    assert repo.get_claim(with_record.claim_id) == with_record
    # A later decision can cite the seeded evidence record by ID without conflict.
    cand = make_candidate()
    cand.subject_key = "gym"
    repo.apply_decision(bundle_for(repo, cand, evidence=[stored_ev]))


def test_documented_default_db_url_works_without_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review item 4: SEKF_DB_URL from .env.example opens even when ./data does not exist."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEKF_DB_URL", raising=False)
    repo = build_memory_repository()
    repo.apply_decision(bundle_for(repo))
    assert (tmp_path / "data" / "sekf.sqlite3").exists()
