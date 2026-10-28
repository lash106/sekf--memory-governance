"""Generic-ensemble baseline as a DecideResponse producer.

Identical Judge calls vote on a label. A label maps to a final action only when a
majority of the full panel agrees (2 of 3). There is no Historian, no Synthesizer,
and no council PolicyEngine. A tie among a full panel becomes INSUFFICIENT_EVIDENCE.
Fewer agreeing votes than that quorum escalates, and the decision cost includes
every member call, including failed repairs. SUPPORTED promotes; this baseline does
not revise or attach evidence, because those actions need specialized temporal reasoning.

The member calls run concurrently. `run_config.timeout_s` is one budget for the batch.

Owner: Sai Chetan. SPEC.md sections 10.2 and 12.1.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from sekf.baselines.decision_links import action_claim_links
from sekf.baselines.generic_ensemble.ensemble import GenericEnsembleJudge
from sekf.contracts import (
    CostSummary,
    DecideMode,
    DecideRequest,
    DecideResponse,
    DecisionRecord,
    FinalAction,
    JudgeLabel,
    JudgeRequest,
    RoleFailure,
    Thresholds,
    compute_decision_id,
    compute_request_hash,
)
from sekf.errors import CallUsage, CommitRejected, RoleOutputInvalid, RoleTimeout, SekfError
from sekf.judge import Judge
from sekf.memory import MemoryRepository

POLICY_VERSION = "generic-ensemble-v1"

# Label vote only. No temporal or provenance specialization.
LABEL_TO_ACTION: dict[JudgeLabel, FinalAction] = {
    JudgeLabel.SUPPORTED: FinalAction.PROMOTE,
    JudgeLabel.REFUTED: FinalAction.REJECT,
    JudgeLabel.INSUFFICIENT_EVIDENCE: FinalAction.KEEP_EPISODIC,
    JudgeLabel.MIXED_EVIDENCE: FinalAction.QUARANTINE,
    JudgeLabel.INVALID_INPUT: FinalAction.ESCALATE_FOR_REVIEW,
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _cost_from_usage(usage: CallUsage | None) -> CostSummary:
    if usage is None:
        return CostSummary()
    return CostSummary(
        llm_calls=usage.llm_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=usage.latency_ms,
        estimated_cost_usd=usage.estimated_cost_usd,
    )


class GenericEnsembleBaseline:
    """Non-specialized three-judge vote that recommends a final action itself."""

    def __init__(
        self,
        member_judge: Judge,
        *,
        memory: MemoryRepository | None = None,
        n_members: int = 3,
        clock: Callable[[], datetime] = _utcnow,
        policy_version: str = POLICY_VERSION,
    ) -> None:
        self._judge = GenericEnsembleJudge(member_judge, n_members=n_members)
        self._memory = memory
        self._clock = clock
        self._policy_version = policy_version

    async def decide(self, request: DecideRequest) -> DecideResponse:
        """Recommend an action. `commit` is rejected until this baseline writes memory."""

        if request.mode is DecideMode.COMMIT:
            raise CommitRejected(
                "generic_ensemble does not write memory; send mode=dry_run "
                "(commit would report a write that did not happen)"
            )
        started = time.perf_counter()
        thresholds = request.thresholds or Thresholds()
        related = (
            self._memory.search_related_claims(request.candidate)
            if self._memory is not None
            else []
        )
        digest = compute_request_hash(
            policy_version=self._policy_version,
            candidate=request.candidate,
            evidence=request.evidence,
            thresholds=thresholds,
            run_config=request.run_config,
            idempotency_key=request.idempotency_key,
        )
        decision_id = compute_decision_id(self._policy_version, digest)
        role_errors: list[RoleFailure] = []
        judge = None
        try:
            # Timeout lives inside GenericEnsembleJudge: one budget, concurrent members.
            judge = await self._judge.evaluate(
                JudgeRequest(
                    candidate=request.candidate,
                    evidence=request.evidence,
                    run_config=request.run_config,
                )
            )
            action = LABEL_TO_ACTION[judge.label]
            previous, resulting = action_claim_links(
                action, candidate_id=request.candidate.claim_id, related=related
            )
            fired: list[str] = []
            explanation = judge.rationale
            cost = CostSummary(
                llm_calls=judge.llm_calls,
                input_tokens=judge.input_tokens,
                output_tokens=judge.output_tokens,
                latency_ms=judge.latency_ms,
                estimated_cost_usd=judge.estimated_cost_usd,
            )
        except (RoleOutputInvalid, RoleTimeout, SekfError) as exc:
            action = FinalAction.ESCALATE_FOR_REVIEW
            previous, resulting = [], None
            fired = ["R1"]
            explanation = str(exc)
            cost = _cost_from_usage(exc.usage if isinstance(exc.usage, CallUsage) else None)
            judge = None
            role_errors = [
                RoleFailure(
                    role="generic_ensemble",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    llm_calls=cost.llm_calls,
                    input_tokens=cost.input_tokens,
                    output_tokens=cost.output_tokens,
                    latency_ms=cost.latency_ms,
                    estimated_cost_usd=cost.estimated_cost_usd,
                )
            ]

        decision = DecisionRecord(
            decision_id=decision_id,
            candidate_claim_id=request.candidate.claim_id,
            final_action=action,
            rule_action=action,
            policy_version=self._policy_version,
            request_hash=digest,
            thresholds=thresholds,
            rule_ids_fired=fired,
            previous_claim_ids=previous,
            resulting_claim_id=resulting,
            judge_assessment_id=None if judge is None else judge.assessment_id,
            explanation=explanation,
            created_at=self._clock(),
            cost=cost,
        )
        return DecideResponse(
            mode=DecideMode.DRY_RUN,
            decision=decision,
            judge=judge,
            role_errors=role_errors,
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )
