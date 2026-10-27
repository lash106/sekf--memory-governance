import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from tests.conftest import NOW

from sekf.api import Governance, create_app
from sekf.contracts import (
    ClaimStatus,
    DecideMode,
    DecideRequest,
    DecideResponse,
    ExistingClaim,
    FinalAction,
    HistorianAssessment,
    HistorianRequest,
    JudgeAssessment,
    JudgeLabel,
    JudgeRequest,
    PolicyContext,
)
from sekf.errors import CommitRejected, RoleOutputInvalid
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.memory import InMemoryRepository
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def client_for(governance: Governance) -> TestClient:
    return TestClient(create_app(governance))


def test_lifespan_closes_the_live_provider() -> None:
    closed = False

    class _Closable:
        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    class _Judge:
        def __init__(self) -> None:
            self._provider = _Closable()

        async def aclose(self) -> None:
            await self._provider.aclose()

        async def evaluate(self, _request: JudgeRequest) -> JudgeAssessment:
            raise AssertionError("health does not call the judge")

    governance = Governance(
        memory=InMemoryRepository(),
        historian=FakeHistorian(),
        judge=_Judge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
    )
    with TestClient(create_app(governance)) as client:
        assert client.get("/health").json()["status"] == "ok"
    assert closed is True


def test_health() -> None:
    body = TestClient(create_app()).get("/health").json()
    assert body["status"] == "ok"
    assert body["contract_version"] == "v1"


def test_worked_example_revises_via_http(
    governance: Governance, lab_fixture: dict[str, Any]
) -> None:
    resp = client_for(governance).post("/v1/decide", json=lab_fixture["request"])
    assert resp.status_code == 200
    body = DecideResponse.model_validate(resp.json())
    assert body.decision.final_action is FinalAction.REVISE
    assert body.decision.rule_ids_fired == ["R8"]
    assert body.historian is not None and body.judge is not None and body.synthesizer is not None
    assert body.decision.cost.llm_calls == 0  # fakes make no real LLM calls
    assert body.historian.provider == "fake" and body.historian.prompt_hash
    assert body.memory_delta is None  # dry_run


def test_dry_run_mutates_nothing(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    asyncio.run(governance.decide(decide_request))
    assert repo.candidate_count() == 0
    assert repo.decision_count() == 0


def test_commit_records_decision_and_is_idempotent(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    first = asyncio.run(governance.decide(commit))
    second = asyncio.run(governance.decide(commit))
    assert first.decision.decision_id == second.decision.decision_id
    assert repo.decision_count() == 1
    assert repo.candidate_count() == 1
    assert first.memory_delta is not None
    assert first.memory_delta.applied is False  # skeleton repo performs no state transition


def test_invalid_request_is_rejected_by_schema(
    governance: Governance, lab_fixture: dict[str, Any]
) -> None:
    bad = {**lab_fixture["request"], "unexpected": True}
    assert client_for(governance).post("/v1/decide", json=bad).status_code == 422


def test_no_evidence_keeps_claim_episodic(
    governance: Governance, decide_request: DecideRequest
) -> None:
    req = decide_request.model_copy(update={"evidence": []})
    resp = asyncio.run(governance.decide(req))
    assert resp.decision.final_action is FinalAction.KEEP_EPISODIC


class FailingHistorian:
    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        raise RoleOutputInvalid("bad json after 2 repairs")


class SlowHistorian:
    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        await asyncio.sleep(5)
        raise AssertionError("should have timed out")


def governance_with(historian: Any, repo: InMemoryRepository) -> Governance:
    return Governance(
        memory=repo,
        historian=historian,
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )


def test_invalid_role_output_escalates(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    resp = asyncio.run(governance_with(FailingHistorian(), repo).decide(decide_request))
    assert resp.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert resp.decision.rule_ids_fired == ["R1"]
    assert [(e.role, e.error_type, e.message) for e in resp.role_errors] == [
        ("historian", "RoleOutputInvalid", "bad json after 2 repairs")
    ]
    assert resp.synthesizer is None


def test_role_timeout_degrades_gracefully(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    fast = decide_request.model_copy(
        update={"run_config": decide_request.run_config.model_copy(update={"timeout_s": 0.05})}
    )
    resp = asyncio.run(governance_with(SlowHistorian(), repo).decide(fast))
    assert resp.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    failure = resp.role_errors[0]
    assert (failure.role, failure.error_type) == ("historian", "RoleTimeout")


class ExplodingPolicy:
    def identify(self, context: PolicyContext) -> tuple[UUID, str]:
        return PolicyEngine().identify(context)

    def decide(self, context: object) -> object:
        raise RuntimeError("policy bug")


def test_policy_crash_in_commit_mode_leaves_memory_untouched(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    gov = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=ExplodingPolicy(),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    with pytest.raises(RuntimeError):
        asyncio.run(gov.decide(commit))
    assert repo.candidate_count() == 0
    assert repo.evidence_count() == 0
    assert repo.decision_count() == 0


def test_commit_stores_candidate_evidence_and_decision_together(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    asyncio.run(governance.decide(commit))
    assert (repo.candidate_count(), repo.evidence_count(), repo.decision_count()) == (1, 1, 1)


def test_escalated_commit_still_stores_candidate_and_decision(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    resp = asyncio.run(governance_with(FailingHistorian(), repo).decide(commit))
    assert resp.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    assert (repo.candidate_count(), repo.evidence_count(), repo.decision_count()) == (1, 1, 1)


def test_rerun_with_different_evidence_gets_a_different_decision_id(
    governance: Governance, decide_request: DecideRequest
) -> None:
    a = asyncio.run(governance.decide(decide_request))
    b = asyncio.run(governance.decide(decide_request.model_copy(update={"evidence": []})))
    assert a.decision.decision_id != b.decision.decision_id


class NondeterministicHistorian:
    """Mimics a real LLM: same request, but fresh assessment UUID, latency, and wording."""

    def __init__(self) -> None:
        self._inner = FakeHistorian()
        self.calls = 0

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        self.calls += 1
        out = await self._inner.analyze(request)
        return out.model_copy(
            update={
                "assessment_id": uuid4(),
                "latency_ms": 100 * self.calls,
                "rationale": f"wording {self.calls}",
            }
        )


def test_retry_with_nondeterministic_llm_is_an_idempotent_replay(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    historian = NondeterministicHistorian()
    gov = governance_with(historian, repo)
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    first = asyncio.run(gov.decide(commit))
    second = asyncio.run(gov.decide(commit))
    assert first.decision.decision_id == second.decision.decision_id
    assert repo.decision_count() == 1
    assert second.memory_delta is not None and second.memory_delta.replayed
    assert historian.calls == 1  # the retry spent no LLM calls
    # The first committed decision wins: the stored role output is the first run's.
    stored = repo.get_audit_trace(commit.candidate.claim_id).entries[0].bundle
    assert stored.historian is not None and first.historian is not None
    assert stored.historian.assessment_id == first.historian.assessment_id


def test_idempotency_key_makes_a_deliberate_rerun(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    gov = governance_with(NondeterministicHistorian(), repo)
    first = asyncio.run(gov.decide(decide_request.model_copy(update={"mode": DecideMode.COMMIT})))
    again = decide_request.model_copy(
        update={"mode": DecideMode.COMMIT, "idempotency_key": "rerun-after-fix"}
    )
    second = asyncio.run(gov.decide(again))
    assert first.decision.decision_id != second.decision.decision_id
    assert repo.decision_count() == 2


def test_committed_audit_keeps_role_outputs_failures_and_run_config(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    cfg = decide_request.run_config.model_copy(update={"model": "m/x", "seed": 7})
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT, "run_config": cfg})
    asyncio.run(governance_with(FailingHistorian(), repo).decide(commit))
    (entry,) = repo.get_audit_trace(commit.candidate.claim_id).entries
    bundle = entry.bundle
    assert [f.error_type for f in bundle.role_errors] == ["RoleOutputInvalid"]
    assert bundle.run_config.model == "m/x" and bundle.run_config.seed == 7
    assert bundle.historian is None and bundle.judge is not None


def test_full_commit_audit_includes_all_three_role_outputs(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    asyncio.run(governance.decide(commit))
    (entry,) = repo.get_audit_trace(commit.candidate.claim_id).entries
    bundle = entry.bundle
    assert bundle.historian and bundle.judge and bundle.synthesizer


class FlippingJudge:
    """First call supports the claim; any later call would refute it."""

    def __init__(self) -> None:
        self.calls = 0
        self._inner = FakeJudge()

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        self.calls += 1
        out = await self._inner.evaluate(request)
        if self.calls == 1:
            return out
        return out.model_copy(
            update={"label": JudgeLabel.REFUTED, "assessment_id": uuid4(), "confidence": 0.95}
        )


def test_retry_returns_the_stored_decision_not_a_new_one(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    judge = FlippingJudge()
    gov = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=judge,
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    first = asyncio.run(gov.decide(commit))
    retry = asyncio.run(gov.decide(commit))
    assert first.decision.final_action is FinalAction.REVISE
    assert retry.decision == first.decision  # the client sees what was actually committed
    assert retry.judge == first.judge
    assert judge.calls == 1  # the second (refuting) judgment was never even requested


def test_retry_with_larger_timeout_is_a_new_decision(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    short = decide_request.model_copy(
        update={
            "mode": DecideMode.COMMIT,
            "run_config": decide_request.run_config.model_copy(update={"timeout_s": 0.05}),
        }
    )
    gov = governance_with(SlowHistorian(), repo)
    escalated = asyncio.run(gov.decide(short))
    assert escalated.decision.final_action is FinalAction.ESCALATE_FOR_REVIEW
    longer = short.model_copy(update={"run_config": decide_request.run_config})
    ok = asyncio.run(governance_with(FakeHistorian(), repo).decide(longer))
    assert ok.decision.final_action is FinalAction.REVISE
    assert ok.decision.decision_id != escalated.decision.decision_id


def test_retry_is_recognized_after_the_decision_changed_memory(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    """A real REVISE supersedes the old claim; the retry must not re-run and re-commit."""
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    first = asyncio.run(governance.decide(commit))
    (old,) = repo.search_related_claims(commit.candidate)
    repo.seed_claim(old.model_copy(update={"status": ClaimStatus.SUPERSEDED}))  # what REVISE does
    retry = asyncio.run(governance.decide(commit))
    assert retry.decision == first.decision
    assert retry.memory_delta is not None and retry.memory_delta.replayed
    assert repo.decision_count() == 1


def test_commit_stores_the_memory_snapshot_the_roles_saw(
    governance: Governance, repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    asyncio.run(governance.decide(commit))
    (entry,) = repo.get_audit_trace(commit.candidate.claim_id).entries
    bundle = entry.bundle
    assert [c.claim_id for c in bundle.memory_before] == [
        c.claim_id for c in repo.search_related_claims(commit.candidate)
    ]


class MemoryMutatingHistorian:
    """Another request supersedes the claim while this decision is being made."""

    def __init__(self, repo: InMemoryRepository) -> None:
        self._repo = repo
        self._inner = FakeHistorian()

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        out = await self._inner.analyze(request)
        for claim in request.related_claims:
            self._repo.seed_claim(claim.model_copy(update={"status": ClaimStatus.SUPERSEDED}))
        return out


def test_commit_on_a_stale_snapshot_is_rejected_and_writes_nothing(
    repo: InMemoryRepository, decide_request: DecideRequest
) -> None:
    gov = governance_with(MemoryMutatingHistorian(repo), repo)
    commit = decide_request.model_copy(update={"mode": DecideMode.COMMIT})
    with pytest.raises(CommitRejected, match="memory changed"):
        asyncio.run(gov.decide(commit))
    assert (repo.candidate_count(), repo.evidence_count(), repo.decision_count()) == (0, 0, 0)


def test_two_racing_promotions_cannot_both_commit(decide_request: DecideRequest) -> None:
    """Both requests read an empty memory; only the first may commit (phantom protection)."""
    from sekf.contracts import CandidateClaim

    empty = InMemoryRepository()

    class CompetingHistorian:
        """While request B is being decided, request A's promotion lands in memory."""

        def __init__(self) -> None:
            self._inner = FakeHistorian()
            self.armed = True

        async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
            out = await self._inner.analyze(request)
            if self.armed:
                self.armed = False
                winner = ExistingClaim.model_validate(
                    {
                        "claim_id": str(uuid4()),
                        "claim_text": "A won",
                        "subject_key": "laboratory",
                        "predicate_key": "closing_time_friday",
                        "object_text": "18:00",
                        "status": "ACTIVE",
                        "version": 1,
                        "confidence": 0.9,
                        "created_at": NOW,
                        "updated_at": NOW,
                    }
                )
                empty.seed_claim(winner)
            return out

    candidate = CandidateClaim.model_validate(decide_request.candidate.model_dump())
    request = decide_request.model_copy(update={"mode": DecideMode.COMMIT, "candidate": candidate})
    gov = governance_with(CompetingHistorian(), empty)
    with pytest.raises(CommitRejected, match="memory changed"):
        asyncio.run(gov.decide(request))
    assert empty.decision_count() == 0
    # The caller retries with a fresh read and now sees the competing claim.
    retried = asyncio.run(gov.decide(request))
    assert retried.decision.final_action is not FinalAction.PROMOTE
    assert empty.decision_count() == 1
