"""Governance orchestration (SPEC.md section 8). Owner: Shailen.

Flow: related-claim lookup -> Historian || Judge -> Synthesizer -> policy -> (commit only) memory.
`dry_run` is read-only: nothing is stored or mutated. Role failures never crash the request;
they become typed `role_errors`, which the policy turns into ESCALATE_FOR_REVIEW (rule R1).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TypeVar

from sekf.contracts import (
    CommitBundle,
    DecideMode,
    DecideRequest,
    DecideResponse,
    HistorianAssessment,
    HistorianRequest,
    JudgeAssessment,
    JudgeRequest,
    MemoryDelta,
    PolicyContext,
    RoleFailure,
    SynthesisRequest,
    SynthesizerRecommendation,
    Thresholds,
)
from sekf.errors import RoleTimeout, SekfError
from sekf.historian import Historian
from sekf.judge import Judge
from sekf.memory import MemoryRepository
from sekf.policy import DecisionPolicy
from sekf.synthesizer import Synthesizer

T = TypeVar("T")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Governance:
    def __init__(
        self,
        *,
        memory: MemoryRepository,
        historian: Historian,
        judge: Judge,
        synthesizer: Synthesizer,
        policy: DecisionPolicy,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._memory = memory
        self._historian = historian
        self._judge = judge
        self._synthesizer = synthesizer
        self._policy = policy
        self._clock = clock

    async def decide(self, request: DecideRequest) -> DecideResponse:
        started = time.perf_counter()
        cand = request.candidate
        timeout = request.run_config.timeout_s
        commit = request.mode is DecideMode.COMMIT

        thresholds = request.thresholds or Thresholds()

        if commit:
            # Retry safety. The identity uses only the submitted request, so it stays stable
            # after a committed decision has changed memory. Checked BEFORE reading memory or
            # calling any role: a retry spends no LLM calls and returns what was committed.
            decision_id, _ = self._policy.identify(
                PolicyContext(
                    candidate=cand,
                    evidence=request.evidence,
                    thresholds=thresholds,
                    run_config=request.run_config,
                    idempotency_key=request.idempotency_key,
                    now=self._clock(),
                )
            )
            stored = self._memory.get_bundle(decision_id)
            if stored is not None:
                return self._replay_response(stored, started)

        related = self._memory.search_related_claims(cand)

        hist_req = HistorianRequest(
            candidate=cand,
            related_claims=related,
            evidence=request.evidence,
            run_config=request.run_config,
        )
        judge_req = JudgeRequest(
            candidate=cand, evidence=request.evidence, run_config=request.run_config
        )
        hist_res, judge_res = await asyncio.gather(
            self._guard("historian", self._historian.analyze(hist_req), timeout),
            self._guard("judge", self._judge.evaluate(judge_req), timeout),
        )

        role_errors: list[RoleFailure] = []
        historian = hist_res if isinstance(hist_res, HistorianAssessment) else None
        judge = judge_res if isinstance(judge_res, JudgeAssessment) else None
        for res in (hist_res, judge_res):
            if isinstance(res, RoleFailure):
                role_errors.append(res)

        synthesizer: SynthesizerRecommendation | None = None
        if historian is not None and judge is not None:
            syn_res = await self._guard(
                "synthesizer",
                self._synthesizer.synthesize(
                    SynthesisRequest(
                        candidate=cand,
                        historian=historian,
                        judge=judge,
                        run_config=request.run_config,
                    )
                ),
                timeout,
            )
            if isinstance(syn_res, SynthesizerRecommendation):
                synthesizer = syn_res
            else:
                role_errors.append(syn_res)

        context = PolicyContext(
            candidate=cand,
            evidence=request.evidence,
            related_claims=related,
            historian=historian,
            judge=judge,
            synthesizer=synthesizer,
            thresholds=thresholds,
            role_errors=role_errors,
            run_config=request.run_config,
            idempotency_key=request.idempotency_key,
            now=self._clock(),
        )
        decision = self._policy.decide(context)

        # Nothing is written before the policy decides; commit is one atomic bundle write.
        memory_delta = None
        if commit:
            memory_delta = self._memory.apply_decision(
                CommitBundle(
                    candidate=cand,
                    evidence=request.evidence,
                    decision=decision,
                    historian=historian,
                    judge=judge,
                    synthesizer=synthesizer,
                    memory_before=related,
                    idempotency_key=request.idempotency_key,
                    role_errors=role_errors,
                    run_config=request.run_config,
                )
            )
        if memory_delta is not None and memory_delta.replayed:
            # Lost a race with an identical request: report what is actually stored.
            stored = self._memory.get_bundle(decision.decision_id)
            if stored is not None:
                return self._replay_response(stored, started)
        return DecideResponse(
            mode=request.mode,
            decision=decision,
            historian=historian,
            judge=judge,
            synthesizer=synthesizer,
            memory_delta=memory_delta,
            role_errors=role_errors,
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _replay_response(stored: CommitBundle, started: float) -> DecideResponse:
        """Answer a retry with the originally committed decision and role outputs."""
        return DecideResponse(
            mode=DecideMode.COMMIT,
            decision=stored.decision,
            historian=stored.historian,
            judge=stored.judge,
            synthesizer=stored.synthesizer,
            memory_delta=MemoryDelta(
                decision_id=stored.decision.decision_id,
                applied=False,
                replayed=True,
                note="replay: returned the originally committed decision",
            ),
            role_errors=stored.role_errors,
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    async def _guard(name: str, call: Awaitable[T], timeout: float) -> T | RoleFailure:
        """Run a role call; return its result or a typed `RoleFailure`."""
        try:
            return await asyncio.wait_for(call, timeout)
        except TimeoutError:
            return RoleFailure(
                role=name, error_type=RoleTimeout.__name__, message=f"exceeded {timeout}s"
            )
        except SekfError as exc:
            return RoleFailure(role=name, error_type=type(exc).__name__, message=str(exc))
