"""Single-model judge baseline (SPEC.md section 12.1). Owner: Sai Chetan.

One model sees the candidate, related claims, and evidence and recommends a final action.
It does not run the council policy rules on the happy path. Failures escalate like R1.
This slice never writes memory; commit wiring can wrap `decide` later.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from sekf.baselines.single_judge.constants import (
    DEFAULT_BASELINE_NAME,
    DEFAULT_POLICY_VERSION,
    DEFAULT_PROMPT_VERSION,
    R1_EXPLANATION,
    SYSTEM_MESSAGE,
)
from sekf.baselines.single_judge.schemas import SingleJudgeLLMOutput
from sekf.contracts import (
    CostSummary,
    DecideRequest,
    DecideResponse,
    DecisionRecord,
    ExistingClaim,
    FinalAction,
    RoleFailure,
    Thresholds,
    compute_decision_id,
    compute_request_hash,
)
from sekf.errors import RoleOutputInvalid, RoleTimeout, SekfError
from sekf.memory import MemoryRepository
from sekf.providers.base import StructuredLLMProvider
from sekf.providers.prompts import LoadedPrompt, load_baseline_prompt, render_prompt
from sekf.providers.serialize import dump_contract_json, dump_contract_list_json


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SingleJudgeBaseline:
    """Mem0-style baseline: one structured LLM call, then a `DecideResponse`."""

    def __init__(
        self,
        provider: StructuredLLMProvider,
        memory: MemoryRepository | None = None,
        *,
        clock: Callable[[], datetime] = _utcnow,
        baseline_name: str = DEFAULT_BASELINE_NAME,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        policy_version: str = DEFAULT_POLICY_VERSION,
        prompt: LoadedPrompt | None = None,
        system_message: str = SYSTEM_MESSAGE,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._clock = clock
        self._policy_version = policy_version
        self._prompt = prompt or load_baseline_prompt(baseline_name, prompt_version)
        self._system_message = system_message

    @property
    def policy_version(self) -> str:
        """Policy id baked into decision hashes for this baseline instance."""

        return self._policy_version

    async def decide(self, request: DecideRequest) -> DecideResponse:
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

        try:
            parsed, cost = await self._recommend(request, related)
            action = parsed.recommended_action
            fired: list[str] = []
            role_errors: list[RoleFailure] = []
            explanation = parsed.rationale
        except (RoleOutputInvalid, RoleTimeout, SekfError) as exc:
            action = FinalAction.ESCALATE_FOR_REVIEW
            fired = ["R1"]
            role_errors = [
                RoleFailure(
                    role="single_judge",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            ]
            explanation = R1_EXPLANATION
            cost = CostSummary()

        decision = DecisionRecord(
            decision_id=decision_id,
            candidate_claim_id=request.candidate.claim_id,
            final_action=action,
            rule_action=action,
            policy_version=self._policy_version,
            request_hash=digest,
            thresholds=thresholds,
            rule_ids_fired=fired,
            previous_claim_ids=[claim.claim_id for claim in related],
            resulting_claim_id=request.candidate.claim_id,
            explanation=explanation,
            created_at=self._clock(),
            cost=cost,
        )
        return DecideResponse(
            mode=request.mode,
            decision=decision,
            role_errors=role_errors,
            wall_latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _recommend(
        self, request: DecideRequest, related: list[ExistingClaim]
    ) -> tuple[SingleJudgeLLMOutput, CostSummary]:
        user_content = render_prompt(
            self._prompt.text,
            candidate_json=dump_contract_json(request.candidate),
            existing_claims_json=dump_contract_list_json(related),
            evidence_json=dump_contract_list_json(request.evidence),
        )
        result = await self._provider.complete_json(
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_content},
            ],
            response_model=SingleJudgeLLMOutput,
            model=request.run_config.model,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
        )
        parsed = result.parsed
        if not isinstance(parsed, SingleJudgeLLMOutput):
            raise RoleOutputInvalid("provider returned a non-SingleJudgeLLMOutput payload")
        cost = CostSummary(
            llm_calls=result.llm_calls,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
        )
        return parsed, cost
