"""State transitions in the SQL store, driven by the labeled dev cases.

An "oracle" system commits each case's GOLD action. If the store implements the claim state
machine the way the annotation guide describes it, the resulting memory delta equals the gold
`expected_memory_delta` for every case. That checks the store against the labels, independent
of any model. The storage baselines are then run on the same cases to show where they differ.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sekf.baselines import DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline
from sekf.baselines.storage_common import StorageBaseline, StorageChoice, active_related
from sekf.contracts import ClaimStatus, DecideMode, DecideRequest, ExistingClaim, FinalAction
from sekf.evaluation.memory_delta import CaseOutcome, evaluate_case
from sekf.memory import MemoryRepository
from sekf.memory.fixtures import MemoryCase, build_seeded_store, case_paths, load_case

ALL = case_paths("dev")
KRISHNA = case_paths("dev", "krishna")
Clock = Callable[[], datetime]


class OracleSystem(StorageBaseline):
    """Commits the gold action (test double: it reads the label on purpose)."""

    baseline_name = "oracle"
    policy_version = "oracle-v1"

    def __init__(self, memory: MemoryRepository, clock: Clock, case: MemoryCase) -> None:
        super().__init__(memory, clock=clock)
        self._case = case

    def choose(self, request: DecideRequest, related: list[ExistingClaim]) -> StorageChoice:
        action = self._case.gold_action
        active = active_related(related)
        target = (
            active[0].claim_id
            if action in (FinalAction.REVISE, FinalAction.ATTACH_EVIDENCE)
            else None
        )
        return StorageChoice(action, "ORACLE", "gold action", target)


def _outcome(
    factory: Callable[[MemoryRepository, Clock], StorageBaseline], path: Path
) -> CaseOutcome:
    return asyncio.run(evaluate_case(factory, load_case(path)))


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_gold_action_produces_the_gold_delta(path: Path) -> None:
    case = load_case(path)
    outcome = asyncio.run(evaluate_case(lambda repo, clock: OracleSystem(repo, clock, case), case))
    assert outcome.action_correct
    assert outcome.comparison.exact_match, outcome.comparison.mismatched_buckets


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
@pytest.mark.parametrize(
    "cls",
    [DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline],
    ids=lambda c: c.__name__,
)
def test_every_baseline_commits_every_case(path: Path, cls: type[StorageBaseline]) -> None:
    outcome = _outcome(lambda repo, clock: cls(repo, clock=clock), path)
    assert outcome.response.memory_delta is not None


def test_there_are_at_least_fifteen_krishna_state_transition_cases() -> None:
    assert len(KRISHNA) >= 15
    actions = {load_case(p).gold_action for p in KRISHNA}
    assert actions == set(FinalAction)  # every action is exercised by at least one case


def _by_id(case_id: str) -> MemoryCase:
    return load_case(next(p for p in KRISHNA if p.stem == case_id))


def _commit_gold(case_id: str) -> tuple[MemoryCase, MemoryRepository]:
    case = _by_id(case_id)
    repo = build_seeded_store(case)
    system = OracleSystem(repo, lambda: datetime(2026, 10, 21, tzinfo=UTC), case)
    request = DecideRequest(
        candidate=case.candidate, evidence=case.evidence, mode=DecideMode.COMMIT
    )
    asyncio.run(system.decide(request))
    return case, repo


def _status(repo: MemoryRepository, claim_id: object) -> ClaimStatus:
    claim = repo.get_claim(claim_id)  # type: ignore[arg-type]
    assert claim is not None
    return claim.status


def test_revise_chain_keeps_older_versions_and_numbers_the_new_one_four() -> None:
    case, repo = _commit_gold("krishna-revise-chain-01")
    v1, v2, v3 = case.initial_memory
    assert [_status(repo, c.claim_id) for c in (v1, v2, v3)] == [ClaimStatus.SUPERSEDED] * 3
    new = repo.get_claim(case.candidate.claim_id)
    assert new is not None and new.version == 4 and new.supersedes_claim_id == v3.claim_id
    history = repo.get_claim_history(case.candidate.claim_id)
    assert [c.version for c in history] == [4, 3, 2, 1]


def test_quarantined_and_rejected_claims_are_never_reopened_by_a_promote() -> None:
    for case_id in ("krishna-promote-after-quarantined-01", "krishna-promote-after-rejected-01"):
        case, repo = _commit_gold(case_id)
        before = case.initial_memory[0]
        assert _status(repo, before.claim_id) is before.status
        assert _status(repo, case.candidate.claim_id) is ClaimStatus.ACTIVE


def test_revise_leaves_a_quarantined_sibling_alone() -> None:
    case, repo = _commit_gold("krishna-revise-with-quarantined-01")
    active, quarantined = case.initial_memory
    assert _status(repo, active.claim_id) is ClaimStatus.SUPERSEDED
    assert _status(repo, quarantined.claim_id) is ClaimStatus.QUARANTINED


def test_duplicate_attaches_evidence_without_a_new_claim_or_version() -> None:
    case, repo = _commit_gold("krishna-duplicate-exact-01")
    target = case.initial_memory[0]
    stored = repo.get_claim(target.claim_id)
    assert stored is not None and stored.version == target.version
    assert {e for e in stored.evidence_ids} == {e.evidence_id for e in case.evidence}
    assert repo.get_claim(case.candidate.claim_id) is None


def test_quarantine_and_reject_leave_the_active_claim_unchanged() -> None:
    for case_id in ("krishna-quarantine-weak-source-01", "krishna-reject-active-intact-01"):
        case, repo = _commit_gold(case_id)
        active = case.initial_memory[0]
        stored = repo.get_claim(active.claim_id)
        assert stored == active


def test_keep_episodic_creates_no_claim() -> None:
    for case_id in ("krishna-keep-episodic-empty-memory-01", "krishna-injection-active-intact-01"):
        case, repo = _commit_gold(case_id)
        assert repo.get_claim(case.candidate.claim_id) is None


def test_escalation_leaves_a_candidate_in_the_review_queue() -> None:
    case, repo = _commit_gold("krishna-escalate-invalid-input-01")
    assert _status(repo, case.candidate.claim_id) is ClaimStatus.CANDIDATE
    queue = repo.list_review_queue()  # type: ignore[attr-defined]
    assert len(queue) == 1


def test_past_period_claim_does_not_replace_the_current_one() -> None:
    case, repo = _commit_gold("krishna-nonoverlap-past-period-01")
    assert _status(repo, case.initial_memory[0].claim_id) is ClaimStatus.ACTIVE
    assert _status(repo, case.candidate.claim_id) is ClaimStatus.ACTIVE


def test_baselines_fail_where_the_labels_say_they_should() -> None:
    """The cases separate the baselines: this is what the pilot table will show."""

    def run(cls: type[StorageBaseline], case_id: str) -> CaseOutcome:
        path = next(p for p in KRISHNA if p.stem == case_id)
        return _outcome(lambda repo, clock: cls(repo, clock=clock), path)

    # Latest-write-wins treats a past-period claim as an update and supersedes the current one.
    lww = run(LatestWriteWinsBaseline, "krishna-nonoverlap-past-period-01")
    assert (
        lww.response.decision.final_action is FinalAction.REVISE and not lww.comparison.exact_match
    )
    # ... and lets a weak-source claim overwrite a good one.
    weak = run(LatestWriteWinsBaseline, "krishna-quarantine-weak-source-01")
    assert weak.response.decision.final_action is FinalAction.REVISE
    # Direct storage activates an injected instruction and a refuted claim.
    assert not run(
        DirectStorageBaseline, "krishna-injection-active-intact-01"
    ).comparison.exact_match
    assert not run(DirectStorageBaseline, "krishna-reject-active-intact-01").comparison.exact_match
    # Static RAG cannot store a new fact.
    assert not run(StaticRagBaseline, "krishna-supported-new-01").comparison.exact_match
    # Where the rule happens to be right it matches: new fact with empty memory.
    assert run(DirectStorageBaseline, "krishna-supported-new-01").comparison.exact_match
