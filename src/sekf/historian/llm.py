"""LLM layer for the Historian (SPEC.md section 18, Sprint 1). Owner: Mahalakshmi.

Runs the rule-based Historian first, then asks a model to look for what rules cannot see:
chiefly ``SOURCE_CONFLICT`` (the sources disagree with each other) and paraphrase detection.
The two answers are merged conservatively:

- The rules stay authoritative for time arithmetic and for which claims exist.
- The model may *upgrade* a finding to ``SOURCE_CONFLICT`` (that only ever makes the policy
  more cautious) and may swap ``DUPLICATE`` for ``PARAPHRASE``. A ``SOURCE_CONFLICT`` is only
  believed when it cites at least two evidence records that exist, come from different
  sources, and have similar reliability (within ``MAX_RELIABILITY_SPREAD``). Otherwise the
  model is treated as disagreeing with the rules, so a mistaken or manipulated model cannot
  force a quarantine by inventing a conflict.
- Any other disagreement keeps the rule-based fields but drops confidence below the policy's
  Historian threshold, so the policy escalates for review instead of guessing.
- The model's confidence can only lower the result, never raise it.

Cited claim IDs are validated: a hallucinated or non-active id is ``RoleOutputInvalid``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from sekf.contracts import (
    ConflictType,
    EvidenceRecord,
    HistorianAssessment,
    HistorianRequest,
    TemporalRelation,
)
from sekf.errors import RoleOutputInvalid
from sekf.historian.deterministic import DeterministicHistorian, validate_citations
from sekf.providers.base import StructuredLLMProvider
from sekf.providers.prompts import LoadedPrompt, load_role_prompt, render_prompt
from sekf.providers.serialize import dump_contract_json, dump_contract_list_json

DEFAULT_PROMPT_ROLE = "historian"
DEFAULT_PROMPT_VERSION = "v1"
SYSTEM_MESSAGE = "You are the SEKF Historian. Reply with valid JSON only."
# Below the policy's default historian_confidence_threshold (0.70): forces ESCALATE_FOR_REVIEW.
DISAGREEMENT_CONFIDENCE = 0.50
# Two sources are "of similar standing" when their reliabilities are within this distance.
MAX_RELIABILITY_SPREAD = 0.30
_SAME_VALUE = {ConflictType.DUPLICATE, ConflictType.PARAPHRASE}


class HistorianLLMOutput(BaseModel):
    """Structured model output before call metadata is attached."""

    model_config = ConfigDict(extra="forbid")

    temporal_relation: TemporalRelation
    conflict_type: ConflictType
    supersedes_claim_id: UUID | None = None
    related_claim_ids: list[UUID] = Field(default_factory=list)
    # The evidence records that disagree with each other (SOURCE_CONFLICT only).
    conflicting_evidence_ids: list[UUID] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def source_conflict_problem(llm: HistorianLLMOutput, evidence: list[EvidenceRecord]) -> str | None:
    """Why a claimed SOURCE_CONFLICT cannot be believed, or None if the evidence backs it."""
    known = {e.evidence_id: e for e in evidence}
    cited = list(dict.fromkeys(llm.conflicting_evidence_ids))
    if len(cited) < 2:
        return "it cites fewer than two evidence records"
    if any(cid not in known for cid in cited):
        return "it cites evidence that is not in the request"
    records = [known[cid] for cid in cited]
    if len({r.source_id for r in records}) < 2:
        return "the cited evidence does not come from at least two different sources"
    spread = max(r.reliability for r in records) - min(r.reliability for r in records)
    if spread > MAX_RELIABILITY_SPREAD:
        return (
            f"the sources differ too much in reliability ({spread:.2f} > "
            f"{MAX_RELIABILITY_SPREAD}); the more reliable one simply wins"
        )
    return None


def merge_assessments(
    base: HistorianAssessment, llm: HistorianLLMOutput, evidence: list[EvidenceRecord]
) -> tuple[ConflictType, UUID | None, float, str]:
    """Merge rules and model; returns (conflict, supersedes, confidence, note)."""
    cap = min(base.confidence, llm.confidence)
    if llm.conflict_type is ConflictType.SOURCE_CONFLICT and base.conflict_type not in _SAME_VALUE:
        problem = source_conflict_problem(llm, evidence)
        if problem is None:
            return (
                ConflictType.SOURCE_CONFLICT,
                None,  # sources disagree: nothing is clearly superseded
                cap,
                "Model found the sources disagree with each other.",
            )
        return (
            base.conflict_type,
            base.supersedes_claim_id,
            min(cap, DISAGREEMENT_CONFIDENCE),
            f"Model claimed a source conflict but {problem}; not accepted, "
            "confidence lowered so the policy escalates.",
        )
    if llm.conflict_type is base.conflict_type or (
        llm.conflict_type in _SAME_VALUE and base.conflict_type in _SAME_VALUE
    ):
        return base.conflict_type, base.supersedes_claim_id, cap, "Model agrees with the rules."
    return (
        base.conflict_type,
        base.supersedes_claim_id,
        min(cap, DISAGREEMENT_CONFIDENCE),
        f"Model disagrees with the rules (rules: {base.conflict_type.value}, "
        f"model: {llm.conflict_type.value}); confidence lowered so the policy escalates.",
    )


class LLMHistorian:
    """Rule-based Historian plus a model check, behind ``StructuredLLMProvider``."""

    def __init__(
        self,
        provider: StructuredLLMProvider,
        *,
        deterministic: DeterministicHistorian | None = None,
        prompt: LoadedPrompt | None = None,
        prompt_role: str = DEFAULT_PROMPT_ROLE,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        system_message: str = SYSTEM_MESSAGE,
    ) -> None:
        self._provider = provider
        self._deterministic = deterministic or DeterministicHistorian()
        self._prompt = prompt or load_role_prompt(prompt_role, prompt_version)
        self._system_message = system_message

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        base = await self._deterministic.analyze(request)
        user_content = render_prompt(
            self._prompt.text,
            candidate_json=dump_contract_json(request.candidate),
            related_claims_json=dump_contract_list_json(request.related_claims),
            evidence_json=dump_contract_list_json(request.evidence),
        )
        result = await self._provider.complete_json(
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": user_content},
            ],
            response_model=HistorianLLMOutput,
            model=request.run_config.model,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
        )
        parsed = result.parsed
        if not isinstance(parsed, HistorianLLMOutput):
            raise RoleOutputInvalid("provider returned a non-HistorianLLMOutput payload")
        self._validate_model_citations(parsed, request)

        conflict, supersedes, confidence, note = merge_assessments(base, parsed, request.evidence)
        assessment = HistorianAssessment(
            assessment_id=uuid4(),
            candidate_claim_id=base.candidate_claim_id,
            related_claim_ids=base.related_claim_ids,
            provenance_quality=base.provenance_quality,
            temporal_relation=base.temporal_relation,
            conflict_type=conflict,
            supersedes_claim_id=supersedes,
            source_reliability_score=base.source_reliability_score,
            confidence=confidence,
            rationale=f"{base.rationale} {note} Model: {parsed.rationale}",
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
        validate_citations(assessment, request)
        return assessment

    async def aclose(self) -> None:
        """Close the underlying provider when it owns a network client."""
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()

    @staticmethod
    def _validate_model_citations(parsed: HistorianLLMOutput, request: HistorianRequest) -> None:
        known = {c.claim_id: c for c in request.related_claims}
        unknown = [cid for cid in parsed.related_claim_ids if cid not in known]
        if unknown:
            raise RoleOutputInvalid(f"Historian model cited unknown claim ids: {unknown}")
        sup = parsed.supersedes_claim_id
        if sup is not None and sup not in known:
            raise RoleOutputInvalid(f"Historian model proposes superseding unknown claim {sup}")
