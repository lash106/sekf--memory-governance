"""LLM Judge implementation (SPEC.md sections 9.2, 10.2, 11). Owner: Sai Chetan.

Assesses evidence support only. Never receives gold labels or a final action.
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from sekf.contracts import JudgeAssessment, JudgeRequest
from sekf.errors import RoleOutputInvalid
from sekf.judge.llm.constants import (
    DEFAULT_PROMPT_ROLE,
    DEFAULT_PROMPT_VERSION,
    SYSTEM_MESSAGE,
)
from sekf.judge.llm.schemas import JudgeLLMOutput
from sekf.providers.base import StructuredLLMProvider
from sekf.providers.prompts import LoadedPrompt, load_role_prompt, render_prompt
from sekf.providers.serialize import dump_contract_json, dump_contract_list_json


class LLMJudge:
    """Judge that calls `StructuredLLMProvider.complete_json`.

    Prompt role/version are constructor knobs so `prompts/judge/v2.txt` can be tried
    without editing call sites. Evidence-ID validation stays mandatory.
    """

    def __init__(
        self,
        provider: StructuredLLMProvider,
        *,
        prompt_role: str = DEFAULT_PROMPT_ROLE,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        prompt: LoadedPrompt | None = None,
        system_message: str = SYSTEM_MESSAGE,
    ) -> None:
        self._provider = provider
        self._prompt = prompt or load_role_prompt(prompt_role, prompt_version)
        self._system_message = system_message

    async def evaluate(self, request: JudgeRequest) -> JudgeAssessment:
        user_content = render_prompt(
            self._prompt.text,
            candidate_json=dump_contract_json(request.candidate),
            evidence_json=dump_contract_list_json(request.evidence),
        )
        result = await self._provider.complete_json(
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_content},
            ],
            response_model=JudgeLLMOutput,
            model=request.run_config.model,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
        )
        parsed = result.parsed
        if not isinstance(parsed, JudgeLLMOutput):
            raise RoleOutputInvalid("provider returned a non-JudgeLLMOutput payload")
        self._validate_cited_ids(parsed, request)

        evidence_fingerprint = "".join(
            str(item.evidence_id) + item.content_hash for item in request.evidence
        )
        return JudgeAssessment(
            assessment_id=uuid5(
                NAMESPACE_URL,
                "llm-judge:"
                + str(request.candidate.claim_id)
                + ":"
                + evidence_fingerprint
                + ":"
                + self._prompt.prompt_hash,
            ),
            candidate_claim_id=request.candidate.claim_id,
            label=parsed.label,
            confidence=parsed.confidence,
            supporting_evidence_ids=parsed.supporting_evidence_ids,
            refuting_evidence_ids=parsed.refuting_evidence_ids,
            missing_information=parsed.missing_information,
            rationale=parsed.rationale,
            model=result.model,
            provider=result.provider,
            prompt_version=self._prompt.version,
            prompt_hash=self._prompt.prompt_hash,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=result.llm_calls,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
        )

    @staticmethod
    def _validate_cited_ids(parsed: JudgeLLMOutput, request: JudgeRequest) -> None:
        known = {item.evidence_id for item in request.evidence}
        cited = set(parsed.supporting_evidence_ids) | set(parsed.refuting_evidence_ids)
        unknown = cited - known
        if unknown:
            raise RoleOutputInvalid(
                "judge cited evidence ids that are not in the request: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
