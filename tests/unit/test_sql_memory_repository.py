from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.conftest import make_candidate, make_context, make_existing, make_historian, make_judge

from sekf.contracts import (
    ClaimStatus,
    CommitBundle,
    ConflictType,
    EpisodeRecord,
    EvidenceRecord,
    FinalAction,
    JudgeLabel,
    TemporalRelation,
)
from sekf.errors import CommitRejected
from sekf.memory import SqlAlchemyRepository
from sekf.policy import PolicyEngine


def _promotion_bundle() -> CommitBundle:
    candidate = make_candidate()
    historian = make_historian(candidate)
    judge = make_judge(candidate)
    decision = PolicyEngine().decide(
        make_context(candidate=candidate, historian=historian, judge=judge)
    )
    return CommitBundle(
        candidate=candidate,
        decision=decision,
        historian=historian,
        judge=judge,
    )


def _revision_bundle(existing):  # type: ignore[no-untyped-def]
    candidate = make_candidate()
    historian = make_historian(
        candidate,
        conflict=ConflictType.TEMPORAL_UPDATE,
        temporal=TemporalRelation.NEWER_UPDATE,
        related=[existing.claim_id],
        supersedes=existing.claim_id,
    )
    judge = make_judge(candidate)
    decision = PolicyEngine().decide(
        make_context(
            candidate=candidate,
            related_claims=[existing],
            historian=historian,
            judge=judge,
        )
    )
    return CommitBundle(
        candidate=candidate,
        decision=decision,
        historian=historian,
        judge=judge,
        memory_before=[existing],
    )


def _action_bundle(action: FinalAction) -> CommitBundle:
    candidate = make_candidate()
    historian = make_historian(candidate)
    judge_label = JudgeLabel.REFUTED if action is FinalAction.REJECT else JudgeLabel.MIXED_EVIDENCE
    judge = make_judge(candidate, label=judge_label, confidence=0.9)
    decision = PolicyEngine().decide(
        make_context(candidate=candidate, historian=historian, judge=judge)
    )
    assert decision.final_action is action
    return CommitBundle(
        candidate=candidate,
        decision=decision,
        historian=historian,
        judge=judge,
    )


def test_promote_persists_active_claim_and_replay_is_idempotent() -> None:
    repository = SqlAlchemyRepository()
    bundle = _promotion_bundle()

    first = repository.apply_decision(bundle)
    replay = repository.apply_decision(bundle)

    stored = repository.get_claim(bundle.candidate.claim_id)
    assert first.applied and not first.replayed
    assert replay.replayed and not replay.applied
    assert stored is not None and stored.status is ClaimStatus.ACTIVE
    assert repository.get_bundle(bundle.decision.decision_id) == bundle
    (entry,) = repository.get_audit_trace(bundle.candidate.claim_id).entries
    assert entry.memory_delta == first


@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        (FinalAction.QUARANTINE, ClaimStatus.QUARANTINED),
        (FinalAction.REJECT, ClaimStatus.REJECTED),
    ],
)
def test_non_active_claim_actions_are_persisted(action, expected_status) -> None:  # type: ignore[no-untyped-def]
    repository = SqlAlchemyRepository()
    bundle = _action_bundle(action)

    delta = repository.apply_decision(bundle)

    claim = repository.get_claim(bundle.candidate.claim_id)
    assert delta.applied
    assert claim is not None and claim.status is expected_status


def test_episodic_decision_does_not_create_validated_claim() -> None:
    candidate = make_candidate()
    historian = make_historian(candidate)
    judge = make_judge(candidate, label=JudgeLabel.INSUFFICIENT_EVIDENCE)
    decision = PolicyEngine().decide(
        make_context(candidate=candidate, historian=historian, judge=judge)
    )
    bundle = CommitBundle(
        candidate=candidate,
        decision=decision,
        historian=historian,
        judge=judge,
    )
    repository = SqlAlchemyRepository()

    delta = repository.apply_decision(bundle)

    assert not delta.applied
    assert repository.get_claim(candidate.claim_id) is None
    assert repository.get_bundle(decision.decision_id) == bundle


def test_attach_evidence_updates_only_the_active_claim() -> None:
    existing = make_existing()
    candidate = make_candidate()
    evidence = EvidenceRecord(
        evidence_id=uuid4(),
        content="official Friday schedule",
        source_id="schedule",
        source_type="official_document",
        retrieved_at=datetime(2026, 10, 21, tzinfo=UTC),
        reliability=0.95,
    )
    historian = make_historian(
        candidate,
        conflict=ConflictType.DUPLICATE,
        temporal=TemporalRelation.SAME_PERIOD,
        related=[existing.claim_id],
    )
    judge = make_judge(candidate)
    decision = PolicyEngine().decide(
        make_context(
            candidate=candidate,
            evidence=[evidence],
            related_claims=[existing],
            historian=historian,
            judge=judge,
        )
    )
    bundle = CommitBundle(
        candidate=candidate,
        evidence=[evidence],
        decision=decision,
        historian=historian,
        judge=judge,
        memory_before=[existing],
    )
    repository = SqlAlchemyRepository()
    repository.seed_claim(existing)

    delta = repository.apply_decision(bundle)

    stored = repository.get_claim(existing.claim_id)
    assert decision.final_action is FinalAction.ATTACH_EVIDENCE
    assert delta.applied and delta.attached_evidence_ids == [evidence.evidence_id]
    assert stored is not None and stored.evidence_ids == [evidence.evidence_id]
    assert repository.get_claim(candidate.claim_id) is None


def test_revision_preserves_old_claim_and_increments_version() -> None:
    existing = make_existing()
    repository = SqlAlchemyRepository()
    repository.seed_claim(existing)
    bundle = _revision_bundle(existing)

    delta = repository.apply_decision(bundle)

    old_claim = repository.get_claim(existing.claim_id)
    new_claim = repository.get_claim(bundle.candidate.claim_id)
    assert old_claim is not None and old_claim.status is ClaimStatus.SUPERSEDED
    assert new_claim is not None and new_claim.status is ClaimStatus.ACTIVE
    assert new_claim.version == existing.version + 1
    assert new_claim.supersedes_claim_id == existing.claim_id
    assert delta.created_claim_ids == [bundle.candidate.claim_id]
    assert [claim.claim_id for claim in repository.get_claim_history(new_claim.claim_id)] == [
        new_claim.claim_id,
        existing.claim_id,
    ]


def test_revision_rolls_back_state_and_audit_on_late_failure(monkeypatch) -> None:
    existing = make_existing()
    repository = SqlAlchemyRepository()
    repository.seed_claim(existing)
    bundle = _revision_bundle(existing).model_copy(
        update={
            "episode": EpisodeRecord(
                episode_id=make_candidate().claim_id,
                content="raw observation",
                source_id="source",
                observed_at=existing.created_at + timedelta(minutes=1),
            )
        }
    )

    def fail_episode_insert(session, episode):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(SqlAlchemyRepository, "_insert_episode", staticmethod(fail_episode_insert))
    with pytest.raises(RuntimeError, match="injected persistence failure"):
        repository.apply_decision(bundle)

    old_claim = repository.get_claim(existing.claim_id)
    assert old_claim is not None and old_claim.status is ClaimStatus.ACTIVE
    assert repository.get_claim(bundle.candidate.claim_id) is None
    assert repository.get_bundle(bundle.decision.decision_id) is None
    assert repository.get_audit_trace(bundle.candidate.claim_id).entries == []


def test_stale_snapshot_is_rejected_without_writing_bundle() -> None:
    existing = make_existing()
    bundle = _revision_bundle(existing)
    changed = existing.model_copy(update={"claim_text": "changed after the read"})
    repository = SqlAlchemyRepository()
    repository.seed_claim(changed)

    with pytest.raises(CommitRejected, match="memory changed"):
        repository.apply_decision(bundle)

    assert repository.get_claim(existing.claim_id) == changed
    assert repository.get_bundle(bundle.decision.decision_id) is None


def test_new_related_claim_after_snapshot_is_rejected() -> None:
    repository = SqlAlchemyRepository()
    bundle = _promotion_bundle()
    repository.seed_claim(make_existing())

    with pytest.raises(CommitRejected, match="memory changed"):
        repository.apply_decision(bundle)

    assert repository.get_claim(bundle.candidate.claim_id) is None
    assert repository.get_bundle(bundle.decision.decision_id) is None


def test_file_database_survives_repository_restart(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    database_url = f"sqlite:///{tmp_path / 'memory.sqlite3'}"
    bundle = _promotion_bundle()
    first_repository = SqlAlchemyRepository(database_url)
    first_repository.apply_decision(bundle)

    reopened_repository = SqlAlchemyRepository(database_url)

    assert reopened_repository.get_claim(bundle.candidate.claim_id) is not None
    assert reopened_repository.get_bundle(bundle.decision.decision_id) == bundle
