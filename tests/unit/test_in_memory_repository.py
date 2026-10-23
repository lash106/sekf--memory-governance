import pytest
from pydantic import ValidationError
from tests.conftest import (
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
)

from sekf.contracts import ClaimStatus, CommitBundle
from sekf.errors import CommitRejected
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine


def test_search_matches_subject_and_predicate_only() -> None:
    existing = make_existing()
    other = make_existing()
    other.subject_key = "library"
    repo = InMemoryRepository([existing, other])
    related = repo.search_related_claims(make_candidate())
    assert [c.claim_id for c in related] == [existing.claim_id]


def test_claim_history_follows_supersedes_chain() -> None:
    old = make_existing(ClaimStatus.SUPERSEDED)
    new = make_existing()
    new.supersedes_claim_id = old.claim_id
    repo = InMemoryRepository([old, new])
    assert [c.claim_id for c in repo.get_claim_history(new.claim_id)] == [
        new.claim_id,
        old.claim_id,
    ]


def _bundle(engine: PolicyEngine) -> CommitBundle:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    decision = engine.decide(make_context(candidate=cand, historian=h, judge=j))
    return CommitBundle(candidate=cand, decision=decision, historian=h, judge=j)


def test_apply_decision_writes_bundle_and_is_idempotent() -> None:
    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    first = repo.apply_decision(bundle)
    again = repo.apply_decision(bundle)
    assert repo.decision_count() == 1 and repo.candidate_count() == 1
    assert again.replayed and not first.replayed and first.decision_id == again.decision_id


def test_inconsistent_bundle_is_rejected_and_writes_nothing() -> None:
    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    bad = bundle.model_copy(update={"candidate": make_candidate()})  # different candidate
    with pytest.raises(CommitRejected):
        repo.apply_decision(bad)
    assert repo.decision_count() == 0 and repo.candidate_count() == 0

    no_judge = bundle.model_copy(update={"judge": None})
    with pytest.raises(CommitRejected):
        repo.apply_decision(no_judge)
    assert repo.decision_count() == 0


def test_bundle_with_tampered_identity_is_rejected() -> None:
    """The repository recomputes request_hash and decision_id; it does not trust them."""
    from uuid import uuid4

    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    forged_hash = bundle.decision.model_copy(update={"request_hash": "f" * 64})
    with pytest.raises(CommitRejected, match="request_hash"):
        repo.apply_decision(bundle.model_copy(update={"decision": forged_hash}))
    forged_id = bundle.decision.model_copy(update={"decision_id": uuid4()})
    with pytest.raises(CommitRejected, match="decision_id"):
        repo.apply_decision(bundle.model_copy(update={"decision": forged_id}))
    assert repo.decision_count() == 0


def test_changed_evidence_under_an_existing_decision_is_rejected() -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from sekf.contracts import EvidenceRecord

    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    repo.apply_decision(bundle)
    ev = EvidenceRecord(
        evidence_id=uuid4(),
        content="other",
        source_id="s",
        source_type="doc",
        retrieved_at=datetime(2026, 10, 21, tzinfo=UTC),
        reliability=0.5,
    )
    with pytest.raises(CommitRejected, match="request_hash"):
        repo.apply_decision(bundle.model_copy(update={"evidence": [ev]}))
    assert repo.evidence_count() == 0
    assert repo.get_bundle(bundle.decision.decision_id) == bundle  # first write untouched


def test_bundle_referencing_claims_outside_memory_before_cannot_be_built() -> None:
    from uuid import uuid4

    cand = make_candidate()
    h = make_historian(cand, related=[uuid4()])  # not part of the stored snapshot
    j = make_judge(cand)
    decision = PolicyEngine().decide(make_context(candidate=cand, historian=h, judge=j))
    with pytest.raises(ValidationError, match="memory_before"):
        CommitBundle(candidate=cand, decision=decision, historian=h, judge=j)


def _bundle_with_memory(existing):  # type: ignore[no-untyped-def]
    cand = make_candidate()
    h = make_historian(cand, related=[existing.claim_id])
    j = make_judge(cand)
    decision = PolicyEngine().decide(
        make_context(candidate=cand, related_claims=[existing], historian=h, judge=j)
    )
    return CommitBundle(
        candidate=cand, decision=decision, historian=h, judge=j, memory_before=[existing]
    )


def test_memory_before_is_stored_in_the_audit_trace() -> None:
    existing = make_existing()
    repo = InMemoryRepository([existing])
    bundle = _bundle_with_memory(existing)
    repo.apply_decision(bundle)
    (entry,) = repo.get_audit_trace(bundle.candidate.claim_id).entries
    assert entry.bundle.memory_before == [existing]
    assert entry.memory_delta is not None  # what the commit actually changed is kept too


def test_stale_memory_snapshot_is_rejected_and_nothing_is_written() -> None:
    existing = make_existing()
    bundle = _bundle_with_memory(existing)
    changed = existing.model_copy(update={"status": ClaimStatus.SUPERSEDED})
    repo = InMemoryRepository([changed])  # memory moved on after the decision was made
    with pytest.raises(CommitRejected, match="memory changed"):
        repo.apply_decision(bundle)
    assert repo.candidate_count() == 0 and repo.decision_count() == 0


def test_duplicate_evidence_ids_cannot_be_committed() -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    from sekf.contracts import EvidenceRecord

    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    eid = uuid4()

    def ev(reliability: float) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=eid,
            content="x",
            source_id="s",
            source_type="doc",
            retrieved_at=datetime(2026, 10, 21, tzinfo=UTC),
            reliability=reliability,
        )

    evidence = [ev(0.3), ev(0.9)]
    decision = PolicyEngine().decide(
        make_context(candidate=cand, historian=h, judge=j, evidence=evidence)
    )
    with pytest.raises(ValidationError, match="duplicate evidence_id"):
        CommitBundle(candidate=cand, decision=decision, historian=h, judge=j, evidence=evidence)


def test_new_related_claim_after_the_snapshot_makes_the_decision_stale() -> None:
    """Phantom: two requests both read 'no related claims'; the second must not also promote."""
    repo = InMemoryRepository()
    stale_bundle = _bundle(PolicyEngine())  # decided against an EMPTY memory_before
    assert stale_bundle.memory_before == []
    repo.seed_claim(make_existing())  # a competing request promoted a related claim meanwhile
    with pytest.raises(CommitRejected, match="memory changed"):
        repo.apply_decision(stale_bundle)
    assert repo.candidate_count() == 0 and repo.decision_count() == 0


def test_removed_related_claim_also_makes_the_decision_stale() -> None:
    existing = make_existing()
    bundle = _bundle_with_memory(existing)
    with pytest.raises(CommitRejected, match="memory changed"):
        InMemoryRepository().apply_decision(bundle)  # the claim is gone from memory


def test_idempotency_key_is_stored_and_part_of_the_identity() -> None:
    cand = make_candidate()
    h, j = make_historian(cand), make_judge(cand)
    engine = PolicyEngine()
    keyed = engine.decide(make_context(candidate=cand, historian=h, judge=j, idempotency_key="k1"))
    bundle = CommitBundle(
        candidate=cand, decision=keyed, historian=h, judge=j, idempotency_key="k1"
    )
    repo = InMemoryRepository()
    repo.apply_decision(bundle)
    (entry,) = repo.get_audit_trace(cand.claim_id).entries
    assert entry.bundle.idempotency_key == "k1"
    with pytest.raises(ValidationError, match="request_hash"):  # key missing from the bundle
        CommitBundle(candidate=cand, decision=keyed, historian=h, judge=j)


def test_returned_audit_objects_are_copies() -> None:
    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    repo.apply_decision(bundle)
    got = repo.get_bundle(bundle.decision.decision_id)
    assert got is not None
    got.decision.explanation = "tampered"
    got.candidate.claim_text = "tampered"
    trace = repo.get_audit_trace(bundle.candidate.claim_id)
    trace.entries[0].bundle.decision.explanation = "tampered"
    fresh = repo.get_bundle(bundle.decision.decision_id)
    assert fresh == bundle and fresh.decision.explanation != "tampered"


def test_inputs_are_copied_on_commit() -> None:
    repo = InMemoryRepository()
    bundle = _bundle(PolicyEngine())
    repo.apply_decision(bundle)
    bundle.decision.explanation = "edited after commit"
    stored = repo.get_bundle(bundle.decision.decision_id)
    assert stored is not None and stored.decision.explanation != "edited after commit"


def test_returned_claims_are_copies_not_stored_memory() -> None:
    existing = make_existing()
    repo = InMemoryRepository([existing])
    cand = make_candidate()

    searched = repo.search_related_claims(cand)[0]
    searched.status = ClaimStatus.SUPERSEDED
    fetched = repo.get_claim(existing.claim_id)
    assert fetched is not None
    fetched.object_text = "tampered"
    repo.get_claim_history(existing.claim_id)[0].confidence = 0.0

    stored = repo.get_claim(existing.claim_id)
    assert stored == existing  # untouched by every mutation above
    assert repo.search_related_claims(cand) == [existing]


def test_claims_are_copied_on_the_way_in() -> None:
    existing = make_existing()
    repo = InMemoryRepository([existing])
    seeded = make_existing()
    repo.seed_claim(seeded)
    existing.status = ClaimStatus.SUPERSEDED  # caller edits its own object afterwards
    seeded.object_text = "tampered"
    assert repo.get_claim(existing.claim_id).status is ClaimStatus.ACTIVE  # type: ignore[union-attr]
    assert repo.get_claim(seeded.claim_id).object_text != "tampered"  # type: ignore[union-attr]


def test_mutating_a_returned_claim_cannot_defeat_stale_detection() -> None:
    existing = make_existing()
    repo = InMemoryRepository([existing])
    bundle = _bundle_with_memory(existing)
    repo.search_related_claims(bundle.candidate)[0].status = ClaimStatus.SUPERSEDED
    repo.apply_decision(bundle)  # memory was not really changed, so the snapshot is still valid
    assert repo.decision_count() == 1
