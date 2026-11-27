"""Metrics, retrieval, markdown projection, and the offline experiment API."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from tests.conftest import (
    NOW,
    make_candidate,
    make_context,
    make_existing,
    make_historian,
    make_judge,
)

from sekf.api import Governance, create_app
from sekf.contracts import ClaimStatus, CommitBundle, ExistingClaim
from sekf.errors import CommitRejected
from sekf.evaluation.metrics import (
    false_promotion_rate,
    macro_f1,
    paired_bootstrap_interval,
    unnecessary_quarantine_rate,
)
from sekf.historian import FakeHistorian
from sekf.judge import FakeJudge
from sekf.judge.calibration import apply_temperature, fit_temperature
from sekf.memory import InMemoryRepository
from sekf.memory.markdown_projection import project_bundle
from sekf.memory.retrieval import hybrid_search, retrieval_recall_at_k
from sekf.policy import PolicyEngine
from sekf.synthesizer import FakeSynthesizer


def test_macro_f1_false_promotion_and_bootstrap() -> None:
    gold = ["PROMOTE", "REJECT", "PROMOTE"]
    pred = ["PROMOTE", "PROMOTE", "QUARANTINE"]
    assert macro_f1(gold, pred) < 1
    rate, hits, support = false_promotion_rate(
        gold,
        pred,
        gold_judge=["SUPPORTED", "REFUTED", "SUPPORTED"],
        gold_conflict=["NONE", "NONE", "SOURCE_CONFLICT"],
    )
    assert support == 2
    assert hits == 1
    assert rate == 0.5
    assert unnecessary_quarantine_rate(gold, pred) == pytest.approx(1 / 3)
    interval = paired_bootstrap_interval(gold, pred, macro_f1, n_resamples=20, seed=1)
    assert interval.low <= interval.point <= interval.high


def test_temperature_scaling_moves_an_overconfident_score() -> None:
    scaled = apply_temperature(0.9, 2.0)
    assert 0.5 < scaled < 0.9
    fitted = fit_temperature([0.99, 0.99], [False, False])
    assert fitted > 1


def test_hybrid_search_prefers_the_same_predicate_and_recall_skips_empty() -> None:
    cand = make_candidate()
    same = make_existing()
    other = make_existing().model_copy(
        update={
            "claim_id": uuid4(),
            "predicate_key": "unrelated_preference",
            "claim_text": "The visitor prefers email.",
            "object_text": "email",
        }
    )
    ranked = hybrid_search(cand, [other, same], k=1)
    assert ranked[0].claim_id == same.claim_id
    assert retrieval_recall_at_k([], []) is None
    assert retrieval_recall_at_k([str(same.claim_id)], [str(same.claim_id)]) == 1


def test_markdown_projection_is_idempotent_and_a_failed_write_keeps_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cand = make_candidate()
    historian = make_historian(cand)
    judge = make_judge(cand)
    decision = PolicyEngine().decide(make_context(candidate=cand, historian=historian, judge=judge))
    bundle = CommitBundle(candidate=cand, decision=decision, historian=historian, judge=judge)
    project_bundle(tmp_path, bundle)
    project_bundle(tmp_path, bundle)
    audit = (tmp_path / "L3_AUDIT_LOG.md").read_text(encoding="utf-8")
    assert audit.count(str(decision.decision_id)) == 1
    claims = (tmp_path / "L2_CLAIMS.md").read_text(encoding="utf-8")
    assert claims.count(str(decision.decision_id)) == 1

    repo = InMemoryRepository()
    repo.markdown_root = tmp_path / "broken"

    def _boom(_root, _bundle):  # type: ignore[no-untyped-def]
        raise OSError("disk full")

    monkeypatch.setattr("sekf.memory.in_memory.project_bundle", _boom)
    with pytest.raises(CommitRejected, match="not kept"):
        repo.apply_decision(bundle)
    assert repo.decision_count() == 0


def test_markdown_projection_retry_finishes_after_partial_claims_write(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit is written last. A crash after L2 must still complete on retry."""

    cand = make_candidate()
    historian = make_historian(cand)
    judge = make_judge(cand)
    decision = PolicyEngine().decide(make_context(candidate=cand, historian=historian, judge=judge))
    bundle = CommitBundle(candidate=cand, decision=decision, historian=historian, judge=judge)
    marker = f"decision:{decision.decision_id}"

    real_append = __import__("sekf.memory.markdown_projection", fromlist=["_append"])._append
    calls = {"n": 0}

    def flaky_append(path, text):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        real_append(path, text)
        # Fail once after L2_CLAIMS lands and before L3_AUDIT_LOG.
        if path.name == "L2_CLAIMS.md" and calls["n"] == 1:
            raise OSError("disk full after claims")

    monkeypatch.setattr("sekf.memory.markdown_projection._append", flaky_append)
    with pytest.raises(OSError, match="disk full after claims"):
        project_bundle(tmp_path, bundle)

    claims = tmp_path / "L2_CLAIMS.md"
    audit = tmp_path / "L3_AUDIT_LOG.md"
    assert claims.is_file() and marker in claims.read_text(encoding="utf-8")
    assert not audit.is_file() or marker not in audit.read_text(encoding="utf-8")

    # Retry: skip the claims line already written; write the audit completion marker.
    project_bundle(tmp_path, bundle)
    assert marker in audit.read_text(encoding="utf-8")
    assert claims.read_text(encoding="utf-8").count(marker) == 1
    # A second retry is a no-op.
    project_bundle(tmp_path, bundle)
    assert claims.read_text(encoding="utf-8").count(marker) == 1
    assert audit.read_text(encoding="utf-8").count(marker) == 1


def test_baseline_and_experiment_routes() -> None:
    governance = Governance(
        memory=InMemoryRepository(),
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    client = TestClient(create_app(governance))
    payload = {
        "candidate": make_candidate().model_dump(mode="json"),
        "evidence": [],
        "mode": "dry_run",
    }
    promoted = client.post("/v1/baselines/direct_storage/decide", json=payload)
    assert promoted.status_code == 200
    assert promoted.json()["decision"]["final_action"] == "PROMOTE"
    assert client.post("/v1/baselines/single_judge/decide", json=payload).status_code == 409
    assert client.post("/v1/baselines/not-a-baseline/decide", json=payload).status_code == 404
    missing = client.get("/v1/experiments/does-not-exist")
    assert missing.status_code == 404


def test_brief_omits_an_old_candidate_without_deleting_it() -> None:
    old = make_existing(ClaimStatus.CANDIDATE).model_copy(
        update={
            "subject_key": "customer-1",
            "updated_at": datetime(2020, 1, 1, tzinfo=UTC),
            "created_at": datetime(2020, 1, 1, tzinfo=UTC),
        }
    )
    active = ExistingClaim.model_validate(
        {
            **old.model_dump(mode="json"),
            "claim_id": str(uuid4()),
            "status": "ACTIVE",
            "updated_at": NOW.isoformat(),
        }
    )
    repo = InMemoryRepository([old, active])
    governance = Governance(
        memory=repo,
        historian=FakeHistorian(),
        judge=FakeJudge(),
        synthesizer=FakeSynthesizer(),
        policy=PolicyEngine(),
        clock=lambda: NOW,
    )
    client = TestClient(create_app(governance))
    body = client.get("/v1/customers/customer-1/brief").json()
    assert body["hidden_expired_candidates"] == 1
    assert len(body["claims"]) == 1
    assert body["claims"][0]["status"] == "ACTIVE"
    assert repo.get_claim(old.claim_id) is not None
