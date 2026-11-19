"""Offline baseline actions for the experiment harness.

`direct_storage`, `latest_write_wins`, and `static_rag` follow SPEC.md section 12.1.
Their production packages are Krishna's. These functions exist so the harness can score
the specified rules without a live model. `generic_ensemble` uses FakeJudge.
`single_judge` is refused here: it needs a real provider, and this module will not
substitute a different model.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import HTTPException

from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import (
    BaselineAudit,
    ClaimStatus,
    DecideMode,
    DecideRequest,
    DecideResponse,
    DecisionRecord,
    ExistingClaim,
    FinalAction,
    Thresholds,
    compute_decision_id,
    compute_request_hash,
)
from sekf.judge import FakeJudge
from sekf.memory import MemoryRepository

_OFFLINE = {"direct_storage", "latest_write_wins", "static_rag", "generic_ensemble"}


def direct_storage_action() -> FinalAction:
    return FinalAction.PROMOTE


def static_rag_action() -> FinalAction:
    return FinalAction.KEEP_EPISODIC


def latest_write_wins_action(request: DecideRequest, related: list[ExistingClaim]) -> FinalAction:
    active = [
        claim
        for claim in related
        if claim.status is ClaimStatus.ACTIVE
        and claim.subject_key == request.candidate.subject_key
        and claim.predicate_key == request.candidate.predicate_key
    ]
    if not active:
        return FinalAction.PROMOTE
    if any(claim.updated_at < request.candidate.observed_at for claim in active):
        return FinalAction.REVISE
    return FinalAction.KEEP_EPISODIC


def _record(
    name: str, request: DecideRequest, action: FinalAction, now: datetime
) -> DecisionRecord:
    thresholds = request.thresholds or Thresholds()
    version = f"baseline-{name}"
    digest = compute_request_hash(
        policy_version=version,
        candidate=request.candidate,
        evidence=request.evidence,
        thresholds=thresholds,
        run_config=request.run_config,
        idempotency_key=request.idempotency_key,
    )
    resulting = (
        request.candidate.claim_id
        if action in {FinalAction.PROMOTE, FinalAction.REVISE, FinalAction.REJECT}
        else None
    )
    return DecisionRecord(
        decision_id=compute_decision_id(version, digest),
        candidate_claim_id=request.candidate.claim_id,
        final_action=action,
        rule_action=action,
        policy_version=version,
        request_hash=digest,
        thresholds=thresholds,
        rule_ids_fired=[name],
        resulting_claim_id=resulting,
        explanation=f"{name} baseline rule.",
        created_at=now,
    )


async def baseline_response(
    name: str, request: DecideRequest, memory: MemoryRepository, now: datetime
) -> DecideResponse:
    """One offline baseline decision. Unknown names and single_judge are HTTP errors."""

    if name == "single_judge":
        raise HTTPException(
            status_code=409,
            detail="single_judge needs SEKF_PROVIDER=openai and a key; no substitute model is used",
        )
    if name not in _OFFLINE:
        raise HTTPException(status_code=404, detail=f"unknown baseline {name}")
    if request.mode is DecideMode.COMMIT:
        raise HTTPException(
            status_code=409,
            detail=f"{name} does not write governed memory from this endpoint; use dry_run",
        )
    if name == "direct_storage":
        action = direct_storage_action()
    elif name == "static_rag":
        action = static_rag_action()
    elif name == "latest_write_wins":
        action = latest_write_wins_action(request, memory.search_related_claims(request.candidate))
    else:
        ensemble = GenericEnsembleBaseline(FakeJudge(), memory=memory, clock=lambda: now)
        return await ensemble.decide(request)
    decision = _record(name, request, action, now)
    return DecideResponse(
        mode=request.mode,
        decision=decision,
        baseline_audit=BaselineAudit(
            audit_id=uuid4(),
            baseline_name=name,
            recommended_action=action,
            confidence=1.0,
            rationale=decision.explanation,
            model="baseline/rules",
            provider="rules",
            prompt_version="none",
            prompt_hash="baseline-rules",
            temperature=0.0,
            llm_calls=0,
        ),
    )
