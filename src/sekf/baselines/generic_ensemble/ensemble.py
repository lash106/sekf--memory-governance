"""Generic ensemble: identical Judge calls, majority of the full panel (SPEC 12.1).

Owner: Sai Chetan. No role specialization.

Quorum is a majority of `n_members` (2 of 3), not a majority of the calls that
happened to return. One surviving vote cannot authorize a label. A full panel
that ties still maps to INSUFFICIENT_EVIDENCE. Fewer successful votes than the
quorum, or an overall timeout before a quorum exists, escalates and keeps every
member's call usage, including failed repairs.

Members run concurrently. `JudgeRequest.run_config.timeout_s` is one budget for
the whole batch, not a budget that is spent once per member.
"""

from __future__ import annotations

import asyncio
import time
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.baselines.generic_ensemble.aggregation import label_with_quorum, quorum_size
from sekf.contracts import JudgeAssessment, JudgeLabel, JudgeRequest
from sekf.errors import CallUsage, RoleOutputInvalid, RoleTimeout, SekfError
from sekf.judge import Judge


def _usage_of_assessment(item: JudgeAssessment) -> CallUsage:
    return CallUsage(
        llm_calls=item.llm_calls,
        input_tokens=item.input_tokens,
        output_tokens=item.output_tokens,
        estimated_cost_usd=item.estimated_cost_usd,
    )


def _usage_of_error(exc: SekfError) -> CallUsage:
    if isinstance(exc.usage, CallUsage):
        return exc.usage
    return CallUsage()


def _combine_usage(parts: list[CallUsage], *, latency_ms: int) -> CallUsage:
    """Sum calls, tokens, and cost. Latency is the batch wall time.

    Summing member latencies would count concurrent waits more than once and
    make the ensemble look slower than the council for an implementation reason.
    """

    return CallUsage(
        llm_calls=sum(part.llm_calls for part in parts),
        input_tokens=sum(part.input_tokens for part in parts),
        output_tokens=sum(part.output_tokens for part in parts),
        latency_ms=latency_ms,
        estimated_cost_usd=sum(part.estimated_cost_usd for part in parts),
    )


class GenericEnsembleJudge:
    """Judge Protocol implementation that votes across `n_members` identical calls."""

    def __init__(self, member: Judge, *, n_members: int = 3) -> None:
        if n_members < 1:
            raise ValueError("n_members must be >= 1")
        self._member = member
        self._n_members = n_members

    @property
    def n_members(self) -> int:
        return self._n_members

    async def _vote(self, request: JudgeRequest) -> JudgeAssessment | SekfError:
        try:
            return await self._member.evaluate(request)
        except SekfError as exc:
            return exc
        except TimeoutError:
            return RoleTimeout("generic_ensemble member timed out")
        except Exception as exc:
            return RoleOutputInvalid(f"generic_ensemble member failed: {type(exc).__name__}: {exc}")

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        """Vote concurrently under one overall timeout of `run_config.timeout_s`."""

        started = time.perf_counter()
        tasks = [asyncio.create_task(self._vote(request)) for _ in range(self._n_members)]
        _, pending = await asyncio.wait(tasks, timeout=request.run_config.timeout_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        assessments: list[JudgeAssessment] = []
        failures: list[SekfError] = []
        timed_out = False
        for task in tasks:
            if task.cancelled() or not task.done():
                timed_out = True
                continue
            raised = task.exception()
            if raised is not None:
                failures.append(
                    RoleOutputInvalid(
                        f"generic_ensemble member crashed: {type(raised).__name__}: {raised}"
                    )
                )
                continue
            item = task.result()
            if isinstance(item, JudgeAssessment):
                assessments.append(item)
            else:
                failures.append(item)

        wall_ms = int((time.perf_counter() - started) * 1000)
        usage = _combine_usage(
            [_usage_of_assessment(item) for item in assessments]
            + [_usage_of_error(exc) for exc in failures],
            latency_ms=wall_ms,
        )
        need = quorum_size(self._n_members)
        winner = label_with_quorum([item.label for item in assessments], self._n_members)
        error_notes = [f"{type(exc).__name__}: {exc}" for exc in failures]
        if timed_out:
            error_notes.append(
                f"overall timeout {request.run_config.timeout_s}s "
                f"with {len(assessments)} completed vote(s)"
            )

        if winner is None and (timed_out or len(assessments) < need):
            detail = (
                f"generic_ensemble: no operational quorum "
                f"({len(assessments)}/{self._n_members} votes, need {need} agreeing)"
            )
            if error_notes:
                detail += ": " + "; ".join(error_notes)
            error_type = RoleTimeout if timed_out else RoleOutputInvalid
            raise error_type(detail, usage=usage)

        label = winner if winner is not None else JudgeLabel.INSUFFICIENT_EVIDENCE
        votes = sum(1 for item in assessments if item.label is label)
        confidence = votes / self._n_members
        first = assessments[0]
        supporting = list(
            dict.fromkeys(
                evidence_id for item in assessments for evidence_id in item.supporting_evidence_ids
            )
        )
        refuting = list(
            dict.fromkeys(
                evidence_id for item in assessments for evidence_id in item.refuting_evidence_ids
            )
        )
        missing = list(
            dict.fromkeys(note for item in assessments for note in item.missing_information)
        )
        fingerprint = ":".join(str(item.assessment_id) for item in assessments)
        rationale = (
            f"generic_ensemble label={label.value} "
            f"agreeing={votes}/{self._n_members} quorum={need}; "
            + " | ".join(f"{item.label.value}@{item.confidence:.2f}" for item in assessments)
        )
        if error_notes:
            rationale += " | member_errors=" + "; ".join(error_notes)
        prompt_hash = sha256(
            (
                "|".join(sorted(item.prompt_hash for item in assessments))
                + f"|n={self._n_members}|failures={len(failures)}"
            ).encode()
        ).hexdigest()
        return JudgeAssessment(
            assessment_id=uuid5(NAMESPACE_URL, "generic-ensemble:" + fingerprint),
            candidate_claim_id=request.candidate.claim_id,
            label=label,
            confidence=confidence,
            supporting_evidence_ids=supporting,
            refuting_evidence_ids=refuting,
            missing_information=missing,
            rationale=rationale,
            model=first.model,
            provider=first.provider,
            prompt_version=first.prompt_version,
            prompt_hash=prompt_hash,
            temperature=first.temperature,
            seed=first.seed,
            llm_calls=usage.llm_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=usage.latency_ms,
            estimated_cost_usd=usage.estimated_cost_usd,
        )
