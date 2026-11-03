"""LLM layer for the Synthesizer (SPEC.md section 18, Sprint 1). Owner: Mahalakshmi.

Runs the rule-based Synthesizer first, then asks a model for a second opinion on what rules
cannot judge (do the two rationales describe the same fact?). The Synthesizer is advisory and
the policy only lets it make an outcome *more* cautious, so the merge is one-directional too:

- Same action: keep it; confidence is the lower of the two.
- The model is more cautious than a mutating rule-based action (PROMOTE, REVISE,
  ATTACH_EVIDENCE): take the model's action and confidence, so the policy veto can fire.
- The model would mutate where the rules would not: ignore it, record it as an open question.
- Both mutate but disagree on which change: that is ambiguity, so recommend QUARANTINE.
- Other disagreements keep the rule-based action at reduced confidence (never a veto).
"""

from __future__ import annotations

import json
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from sekf.contracts import FinalAction, SynthesisRequest, SynthesizerRecommendation
from sekf.errors import RoleOutputInvalid
from sekf.providers.base import StructuredLLMProvider
from sekf.providers.prompts import LoadedPrompt, load_role_prompt, render_prompt
from sekf.providers.serialize import dump_contract_json
from sekf.synthesizer.deterministic import MUTATING, UNSURE, VETO, DeterministicSynthesizer

DEFAULT_PROMPT_ROLE = "synthesizer"
DEFAULT_PROMPT_VERSION = "v1"
SYSTEM_MESSAGE = "You are the SEKF Synthesizer. Reply with valid JSON only."
# Call bookkeeping is noise to the model; it only needs what each role concluded.
_METADATA_FIELDS = {
    "model",
    "provider",
    "prompt_version",
    "prompt_hash",
    "temperature",
    "seed",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "estimated_cost_usd",
}


class SynthesizerLLMOutput(BaseModel):
    """Structured model output before call metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    recommended_action: FinalAction
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_questions: list[str] = Field(default_factory=list)
    rationale: str


def merge_recommendations(
    base: SynthesizerRecommendation, llm: SynthesizerLLMOutput
) -> tuple[FinalAction, float, list[str], str]:
    """Merge rules and model; returns (action, confidence, questions, note)."""
    cap = min(base.confidence, llm.confidence)
    ours, theirs = base.recommended_action, llm.recommended_action
    if ours is theirs:
        return ours, cap, [], "Model agrees with the rules."
    if ours in MUTATING and theirs not in MUTATING:
        return (
            theirs,
            llm.confidence,
            [f"Model advised {theirs.value} instead of {ours.value}."],
            f"Model is more cautious than the rules ({ours.value} -> {theirs.value}).",
        )
    if ours in MUTATING and theirs in MUTATING:
        return (
            FinalAction.QUARANTINE,
            VETO,
            [f"Rules say {ours.value} but the model says {theirs.value}; unclear which change."],
            "Rules and model disagree on which change to make; quarantining.",
        )
    if theirs in MUTATING:
        return (
            ours,
            min(cap, UNSURE),
            [f"Model suggested {theirs.value}; ignored (the Synthesizer may only add caution)."],
            "Model would change memory where the rules would not; ignored.",
        )
    return (
        ours,
        min(cap, UNSURE),
        [f"Model advised {theirs.value} instead of {ours.value}."],
        "Rules and model differ on a non-mutating action; kept the rules' action.",
    )


class LLMSynthesizer:
    """Rule-based Synthesizer plus a model second opinion, behind ``StructuredLLMProvider``."""

    def __init__(
        self,
        provider: StructuredLLMProvider,
        *,
        deterministic: DeterministicSynthesizer | None = None,
        prompt: LoadedPrompt | None = None,
        prompt_role: str = DEFAULT_PROMPT_ROLE,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        system_message: str = SYSTEM_MESSAGE,
    ) -> None:
        self._provider = provider
        self._deterministic = deterministic or DeterministicSynthesizer()
        self._prompt = prompt or load_role_prompt(prompt_role, prompt_version)
        self._system_message = system_message

    async def synthesize(self, request: SynthesisRequest) -> SynthesizerRecommendation:
        base = await self._deterministic.synthesize(request)
        user_content = render_prompt(
            self._prompt.text,
            candidate_json=dump_contract_json(request.candidate),
            historian_json=_dump_without_metadata(request.historian),
            judge_json=_dump_without_metadata(request.judge),
        )
        result = await self._provider.complete_json(
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_content},
            ],
            response_model=SynthesizerLLMOutput,
            model=request.run_config.model,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
        )
        parsed = result.parsed
        if not isinstance(parsed, SynthesizerLLMOutput):
            raise RoleOutputInvalid("provider returned a non-SynthesizerLLMOutput payload")

        action, confidence, questions, note = merge_recommendations(base, parsed)
        return SynthesizerRecommendation(
            recommendation_id=uuid4(),
            candidate_claim_id=base.candidate_claim_id,
            recommended_action=action,
            confidence=confidence,
            historian_assessment_id=base.historian_assessment_id,
            judge_assessment_id=base.judge_assessment_id,
            rationale=f"{base.rationale} {note} Model: {parsed.rationale}",
            unresolved_questions=[
                *base.unresolved_questions,
                *parsed.unresolved_questions,
                *questions,
            ],
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

    async def aclose(self) -> None:
        """Close the underlying provider when it owns a network client."""
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()


def _dump_without_metadata(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude=_METADATA_FIELDS)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
