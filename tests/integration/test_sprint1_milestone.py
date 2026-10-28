"""Sprint 1 milestone (SPEC.md section 18): five governed decisions end to end.

Real: the HTTP layer, orchestration, the deterministic Historian, the LLMJudge's parsing and
validation, the deterministic policy, and the SQLite memory store. Scripted: the language
model's output (a FakeProvider) and the Synthesizer (the fake one, until its real
implementation lands). Source conflicts (cases 6 and 7) additionally use a scripted Historian,
because the rule-based Historian cannot detect them yet (it is a documented v1 limitation).
Every case commits, then checks the resulting memory state, the audit trail, and the cost.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from tests.conftest import NOW, load_fixture

from sekf.api import Governance, create_app
from sekf.contracts import (
    ClaimStatus,
    ConflictType,
    DecideRequest,
    DecideResponse,
    ExistingClaim,
    FinalAction,
    HistorianAssessment,
    HistorianRequest,
    JudgeAssessment,
    JudgeRequest,
    TemporalRelation,
)
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge, LLMJudge
from sekf.memory import SqlMemoryRepository
from sekf.policy import PolicyEngine
from sekf.providers import FakeProvider
from sekf.synthesizer import FakeSynthesizer

EVIDENCE_ID = "33333333-3333-4333-8333-333333333333"
NEW_CLAIM_ID = "44444444-4444-4444-8444-444444444444"
OLD_CLAIM_ID = UUID("11111111-1111-4111-8111-111111111111")


def judge_saying(
    label: str, *, supporting: bool = False, refuting: bool = False, confidence: float = 0.95
) -> LLMJudge:
    """A real LLMJudge whose (fake) model always answers with `label`."""

    def respond(_messages: Any, _model: Any) -> dict[str, Any]:
        return {
            "label": label,
            "confidence": confidence,
            "supporting_evidence_ids": [EVIDENCE_ID] if supporting else [],
            "refuting_evidence_ids": [EVIDENCE_ID] if refuting else [],
            "missing_information": [] if (supporting or refuting) else ["nothing to go on"],
            "rationale": f"scripted: {label}",
        }

    return LLMJudge(FakeProvider(respond))


class Stack:
    def __init__(self, judge: Any, *, seed_old_claim: bool = False, historian: Any = None) -> None:
        self.repo = SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW)
        if seed_old_claim:
            fixture = load_fixture("lab_closing_revise.json")
            self.repo.seed_claims(
                [ExistingClaim.model_validate(c) for c in fixture["existing_claims"]]
            )
        gov = Governance(
            memory=self.repo,
            historian=historian or DeterministicHistorian(),
            judge=judge,
            synthesizer=FakeSynthesizer(),
            policy=PolicyEngine(),
            clock=lambda: NOW,
        )
        self.client = TestClient(create_app(gov))

    def commit(self, request: dict[str, Any]) -> DecideResponse:
        resp = self.client.post("/v1/decide", json={**request, "mode": "commit"})
        assert resp.status_code == 200, resp.text
        return DecideResponse.model_validate(resp.json())


def request_for(
    *,
    subject: str = "laboratory",
    predicate: str = "closing_time_friday",
    value: str = "18:00",
    reliability: float | None = 0.95,
    evidence_text: str | None = None,
) -> dict[str, Any]:
    """A commit request. `reliability=None` means no evidence at all."""
    evidence = []
    if reliability is not None:
        evidence = [
            {
                "evidence_id": EVIDENCE_ID,
                "content": evidence_text or f"Official notice: the value is {value}.",
                "source_id": "official-notice" if reliability >= 0.6 else "hallway-rumor",
                "source_type": "official_document" if reliability >= 0.6 else "chat",
                "published_at": "2026-09-28T00:00:00Z",
                "retrieved_at": "2026-10-01T20:00:00Z",
                "reliability": reliability,
            }
        ]
    return {
        "candidate": {
            "claim_id": NEW_CLAIM_ID,
            "claim_text": f"{subject} {predicate} is {value}.",
            "subject_key": subject,
            "predicate_key": predicate,
            "object_text": value,
            "source_id": "student-chat",
            "observed_at": "2026-10-01T20:00:00Z",
            "valid_from": "2026-10-01T00:00:00Z",
            "valid_to": None,
            "extractor_confidence": 0.9,
        },
        "evidence": evidence,
    }


class SourceConflictHistorian(DeterministicHistorian):
    """Stands in for the LLM Historian layer that will detect SOURCE_CONFLICT."""

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        out = await super().analyze(request)
        return out.model_copy(
            update={
                "temporal_relation": TemporalRelation.OVERLAPPING_VALIDITY,
                "conflict_type": ConflictType.SOURCE_CONFLICT,
                "supersedes_claim_id": None,
                "confidence": 0.85,
                "provider": "scripted",
                "rationale": "Two sources of similar standing give different values.",
            }
        )


def assert_audit_complete(stack: Stack, resp: DecideResponse, *, applied: bool) -> None:
    """The committed audit trail holds the inputs, all role outputs, the decision, the delta."""
    (entry,) = stack.repo.get_audit_trace(UUID(NEW_CLAIM_ID)).entries
    assert entry.bundle.decision == resp.decision
    assert entry.bundle.historian is not None and entry.bundle.judge is not None
    assert entry.bundle.synthesizer is not None
    assert entry.bundle.historian.provider in {"deterministic", "scripted"}
    assert entry.memory_delta is not None and entry.memory_delta.applied is applied
    assert resp.decision.cost.llm_calls == 0  # nothing here touched a real model


# --------------------------------------------------------------------- the five cases


def test_case_1_supported_new_claim_is_promoted() -> None:
    stack = Stack(judge_saying("SUPPORTED", supporting=True), seed_old_claim=False)
    resp = stack.commit(request_for(subject="library", predicate="opening_time", value="08:00"))
    assert resp.decision.final_action is FinalAction.PROMOTE
    assert resp.decision.rule_ids_fired == ["R10"]
    claim = stack.repo.get_claim(UUID(NEW_CLAIM_ID))
    assert claim is not None and claim.status is ClaimStatus.ACTIVE and claim.version == 1
    assert_audit_complete(stack, resp, applied=True)


def test_case_2_refuted_claim_is_rejected_and_kept_for_audit() -> None:
    stack = Stack(judge_saying("REFUTED", refuting=True), seed_old_claim=True)
    resp = stack.commit(request_for(value="23:00", evidence_text="Closing is 17:00, not 23:00."))
    assert resp.decision.final_action is FinalAction.REJECT
    assert resp.decision.rule_ids_fired == ["R2"]
    claim = stack.repo.get_claim(UUID(NEW_CLAIM_ID))
    assert claim is not None and claim.status is ClaimStatus.REJECTED  # kept, never deleted
    old = stack.repo.get_claim(OLD_CLAIM_ID)
    assert old is not None and old.status is ClaimStatus.ACTIVE  # existing knowledge untouched
    assert_audit_complete(stack, resp, applied=True)


def test_case_3_insufficient_evidence_is_retained_as_episodic_only() -> None:
    stack = Stack(judge_saying("INSUFFICIENT_EVIDENCE"), seed_old_claim=False)
    resp = stack.commit(request_for(subject="gym", predicate="closing_time", reliability=None))
    assert resp.decision.final_action is FinalAction.KEEP_EPISODIC
    assert resp.decision.rule_ids_fired == ["R3"]
    assert stack.repo.get_claim(UUID(NEW_CLAIM_ID)) is None  # not validated knowledge
    # The decision is recorded even though no claim state changed (delta.applied is False).
    assert_audit_complete(stack, resp, applied=False)


def test_case_4_newer_but_unreliable_claim_is_quarantined_not_applied() -> None:
    """The core idea: a newer claim is not automatically a true claim."""
    stack = Stack(judge_saying("SUPPORTED", supporting=True), seed_old_claim=True)
    resp = stack.commit(request_for(reliability=0.3, evidence_text="Heard it closes at 18:00."))
    assert resp.historian is not None
    assert resp.historian.conflict_type.value == "TEMPORAL_UPDATE"  # looks like an update...
    assert resp.decision.final_action is FinalAction.QUARANTINE  # ...but the source is weak
    assert resp.decision.rule_ids_fired == ["R9"]
    old = stack.repo.get_claim(OLD_CLAIM_ID)
    assert old is not None and old.status is ClaimStatus.ACTIVE
    claim = stack.repo.get_claim(UUID(NEW_CLAIM_ID))
    assert claim is not None and claim.status is ClaimStatus.QUARANTINED
    assert_audit_complete(stack, resp, applied=True)


def test_case_5_reliable_newer_update_revises_and_keeps_history() -> None:
    stack = Stack(judge_saying("SUPPORTED", supporting=True), seed_old_claim=True)
    resp = stack.commit(request_for(evidence_text="Beginning October 1, closing time is 18:00."))
    assert resp.decision.final_action is FinalAction.REVISE
    assert resp.decision.rule_ids_fired == ["R8"]
    old = stack.repo.get_claim(OLD_CLAIM_ID)
    new = stack.repo.get_claim(UUID(NEW_CLAIM_ID))
    assert old is not None and old.status is ClaimStatus.SUPERSEDED  # kept, not deleted
    assert new is not None and new.status is ClaimStatus.ACTIVE and new.version == 2
    assert new.supersedes_claim_id == OLD_CLAIM_ID
    assert [c.claim_id for c in stack.repo.get_claim_history(UUID(NEW_CLAIM_ID))] == [
        UUID(NEW_CLAIM_ID),
        OLD_CLAIM_ID,
    ]
    assert_audit_complete(stack, resp, applied=True)

    retry = stack.commit(request_for(evidence_text="Beginning October 1, closing time is 18:00."))
    assert retry.decision == resp.decision  # a retry returns what was committed
    assert retry.memory_delta is not None and retry.memory_delta.replayed
    assert stack.repo.decision_count() == 1


# ---------------------------------------------------- concurrency and cost aggregation


class SlowHistorian(DeterministicHistorian):
    async def analyze(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.25)
        return await super().analyze(request)


class SlowJudge(FakeJudge):
    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        await asyncio.sleep(0.25)
        return await super().evaluate(request)


def test_historian_and_judge_run_concurrently() -> None:
    gov = Governance(
        memory=SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW),
        historian=SlowHistorian(),
        judge=SlowJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    request = DecideRequest.model_validate(request_for())
    started = time.perf_counter()
    asyncio.run(gov.decide(request))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.45, f"roles ran one after another ({elapsed:.2f}s)"


class Costly:
    """Wraps a role and reports usage as a real LLM-backed role would."""

    def __init__(self, inner: Any, method: str, calls: int, tokens: int, cost: float) -> None:
        self._inner, self._method = inner, method
        self._usage = {
            "llm_calls": calls,
            "input_tokens": tokens,
            "output_tokens": tokens // 2,
            "latency_ms": 10 * calls,
            "estimated_cost_usd": cost,
        }

    async def _call(self, request: Any) -> Any:
        out = await getattr(self._inner, self._method)(request)
        return out.model_copy(update=self._usage)

    async def analyze(self, request: Any) -> Any:
        return await self._call(request)

    async def evaluate(self, request: Any) -> Any:
        return await self._call(request)

    async def synthesize(self, request: Any) -> Any:
        return await self._call(request)


@pytest.fixture
def costed_governance() -> Callable[[], Governance]:
    def build() -> Governance:
        return Governance(
            memory=SqlMemoryRepository.from_url("sqlite://", clock=lambda: NOW),
            historian=Costly(DeterministicHistorian(), "analyze", 1, 100, 0.01),
            judge=Costly(FakeJudge(), "evaluate", 2, 200, 0.02),  # e.g. one repair attempt
            synthesizer=Costly(FakeSynthesizer(), "synthesize", 1, 50, 0.005),
            policy=PolicyEngine(),
            clock=lambda: NOW,
        )

    return build


def test_cost_is_aggregated_across_all_three_roles(
    costed_governance: Callable[[], Governance],
) -> None:
    request = DecideRequest.model_validate(request_for())
    resp = asyncio.run(costed_governance().decide(request))
    cost = resp.decision.cost
    assert cost.llm_calls == 4  # 1 + 2 (with a repair) + 1
    assert cost.input_tokens == 350
    assert cost.output_tokens == 175
    assert cost.estimated_cost_usd == pytest.approx(0.035)
    assert cost.latency_ms == 10 + 20 + 10


# ------------------------------------------------------ source-conflict quarantine


def test_case_6_source_conflict_is_quarantined_by_the_historians_conflict_type() -> None:
    """Supported by its own evidence, but two sources disagree: do not replace, quarantine."""
    stack = Stack(
        judge_saying("SUPPORTED", supporting=True),
        seed_old_claim=True,
        historian=SourceConflictHistorian(),
    )
    resp = stack.commit(request_for(evidence_text="Beginning October 1, closing time is 18:00."))
    assert resp.historian is not None
    assert resp.historian.conflict_type is ConflictType.SOURCE_CONFLICT
    assert resp.decision.final_action is FinalAction.QUARANTINE
    assert resp.decision.rule_ids_fired == ["R9"]
    old = stack.repo.get_claim(OLD_CLAIM_ID)
    assert old is not None and old.status is ClaimStatus.ACTIVE  # not replaced
    claim = stack.repo.get_claim(UUID(NEW_CLAIM_ID))
    assert claim is not None and claim.status is ClaimStatus.QUARANTINED
    assert_audit_complete(stack, resp, applied=True)


def test_case_7_dataset_source_conflict_case_runs_end_to_end() -> None:
    """datasets/dev/shailen/shailen-source-similar-01: two equally reliable offices disagree."""
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "datasets/dev/shailen/shailen-source-similar-01.json"
    )
    case = json.loads(path.read_text())
    evidence_ids = [e["evidence_id"] for e in case["evidence"]]

    def mixed(_messages: Any, _model: Any) -> dict[str, Any]:
        return {
            "label": "MIXED_EVIDENCE",
            "confidence": 0.9,
            "supporting_evidence_ids": evidence_ids[:1],
            "refuting_evidence_ids": evidence_ids[1:],
            "missing_information": [],
            "rationale": "scripted: one office says $150, the other $120",
        }

    stack = Stack(LLMJudge(FakeProvider(mixed)), historian=SourceConflictHistorian())
    stack.repo.seed_claims([ExistingClaim.model_validate(c) for c in case["initial_memory"]])
    candidate_id = UUID(case["candidate"]["claim_id"])
    resp = stack.commit({"candidate": case["candidate"], "evidence": case["evidence"]})

    gold = case["gold"]
    assert resp.judge is not None and resp.judge.label.value == gold["judge_label"]
    assert resp.decision.final_action.value == gold["final_action"]  # QUARANTINE
    claim = stack.repo.get_claim(candidate_id)
    assert claim is not None and claim.status is ClaimStatus.QUARANTINED
    old_id = case["initial_memory"][0]["claim_id"]
    old = stack.repo.get_claim(UUID(old_id))
    assert old is not None and old.status is ClaimStatus.ACTIVE
