"""Generic ensemble: three identical Judge calls, majority label (SPEC 12.1).

Owner: Sai Chetan. No role specialization. Ties -> INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from sekf.baselines.generic_ensemble.aggregation import majority_judge_label
from sekf.contracts import JudgeAssessment, JudgeRequest
from sekf.errors import RoleOutputInvalid, SekfError
from sekf.judge import Judge


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

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        assessments: list[JudgeAssessment] = []
        errors: list[str] = []
        for _ in range(self._n_members):
            try:
                assessments.append(await self._member.evaluate(request))
            except SekfError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        if not assessments:
            raise RoleOutputInvalid("generic_ensemble: every member failed: " + "; ".join(errors))

        label = majority_judge_label([item.label for item in assessments])
        # Confidence = share of members that voted for the winning label.
        votes = sum(1 for item in assessments if item.label is label)
        confidence = votes / self._n_members

        supporting: list = []
        refuting: list = []
        missing: list[str] = []
        for item in assessments:
            supporting.extend(item.supporting_evidence_ids)
            refuting.extend(item.refuting_evidence_ids)
            missing.extend(item.missing_information)
        # Deduplicate while preserving order.
        supporting = list(dict.fromkeys(supporting))
        refuting = list(dict.fromkeys(refuting))
        missing = list(dict.fromkeys(missing))

        first = assessments[0]
        fingerprint = ":".join(str(item.assessment_id) for item in assessments)
        rationale = (
            f"generic_ensemble majority={label.value} "
            f"votes={votes}/{self._n_members}; "
            + " | ".join(f"{a.label.value}@{a.confidence:.2f}" for a in assessments)
        )
        if errors:
            rationale += " | member_errors=" + "; ".join(errors)

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
            prompt_hash=sha256((first.prompt_hash + f"|n={self._n_members}").encode()).hexdigest(),
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=sum(item.llm_calls for item in assessments),
            input_tokens=sum(item.input_tokens for item in assessments),
            output_tokens=sum(item.output_tokens for item in assessments),
            latency_ms=sum(item.latency_ms for item in assessments),
            estimated_cost_usd=sum(item.estimated_cost_usd for item in assessments),
        )
