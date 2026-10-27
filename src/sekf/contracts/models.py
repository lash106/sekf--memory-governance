"""Canonical Pydantic contracts, v1 (SPEC.md section 9.1).

These models are the single source of truth. `contracts/v1/*.json` is generated from them
by `make contracts`. Unknown fields are rejected. Changing a model requires an ADR and
approval from every affected module owner.
"""

import hashlib
import json
from typing import Annotated, Any, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sekf.contracts.enums import (
    ClaimStatus,
    ConflictType,
    DecideMode,
    FinalAction,
    JudgeLabel,
    ProvenanceQuality,
    TemporalRelation,
)

CONTRACT_VERSION = "v1"

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class Contract(BaseModel):
    """Base for every contract: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- inputs


class CandidateClaim(Contract):
    claim_id: UUID
    claim_text: str = Field(min_length=1)
    subject_key: str = Field(min_length=1)
    predicate_key: str = Field(min_length=1)
    object_text: str = Field(min_length=1)
    source_id: str
    observed_at: AwareDatetime
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    extractor_confidence: UnitFloat = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validity_window_ordered(self) -> Self:
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must not precede valid_from")
        return self


class EvidenceRecord(Contract):
    evidence_id: UUID
    content: str = Field(min_length=1)
    source_id: str
    source_type: str
    published_at: AwareDatetime | None = None
    retrieved_at: AwareDatetime
    reliability: UnitFloat
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_hash(self) -> Self:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not self.content_hash:
            self.content_hash = expected
        elif self.content_hash != expected:
            raise ValueError("content_hash does not match sha256 of content")
        return self


class EpisodeRecord(Contract):
    """Raw experience, preserved before any claim extraction."""

    episode_id: UUID
    content: str
    source_id: str
    observed_at: AwareDatetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExistingClaim(Contract):
    claim_id: UUID
    claim_text: str
    subject_key: str
    predicate_key: str
    object_text: str
    status: ClaimStatus
    version: int = Field(ge=1)
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    confidence: UnitFloat
    created_at: AwareDatetime
    updated_at: AwareDatetime
    supersedes_claim_id: UUID | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)


# ------------------------------------------------------------------- role outputs


class CallMetadata(Contract):
    """Model-call bookkeeping shared by every role output."""

    model: str
    prompt_version: str
    # Reproducibility metadata (SPEC.md sections 11 and 15).
    # Required on purpose: a role that forgets them must fail validation, not pass silently.
    provider: str = Field(min_length=1)
    temperature: float = Field(ge=0.0)
    seed: int | None = None  # None is legitimate (provider-chosen / unsupported)
    prompt_hash: str = Field(min_length=1)
    # Real provider calls made, including repair attempts. Deterministic/fake roles report 0.
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class HistorianAssessment(CallMetadata):
    assessment_id: UUID
    candidate_claim_id: UUID
    related_claim_ids: list[UUID] = Field(default_factory=list)
    provenance_quality: ProvenanceQuality
    temporal_relation: TemporalRelation
    conflict_type: ConflictType
    supersedes_claim_id: UUID | None = None
    source_reliability_score: UnitFloat
    confidence: UnitFloat
    rationale: str


class JudgeAssessment(CallMetadata):
    assessment_id: UUID
    candidate_claim_id: UUID
    label: JudgeLabel
    confidence: UnitFloat
    supporting_evidence_ids: list[UUID] = Field(default_factory=list)
    refuting_evidence_ids: list[UUID] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rationale: str


class SynthesizerRecommendation(CallMetadata):
    """Advisory only: it may make an outcome more conservative, never less."""

    recommendation_id: UUID
    candidate_claim_id: UUID
    recommended_action: FinalAction
    confidence: UnitFloat
    historian_assessment_id: UUID
    judge_assessment_id: UUID
    rationale: str
    unresolved_questions: list[str] = Field(default_factory=list)


class BaselineAudit(CallMetadata):
    """Reproducibility record for a baseline that is not a council role.

    SPEC.md section 15. `model` and `provider` are the values the call actually
    used. `temperature` and `seed` are the values that call observed. `raw_output`
    is the validated structured payload. `RunConfig.model` is not a substitute:
    `fake/none` resolves to the provider's default model.
    """

    audit_id: UUID
    baseline_name: str = Field(min_length=1)
    recommended_action: FinalAction
    confidence: UnitFloat
    rationale: str
    target_claim_id: UUID | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------- policy and decision


class RunConfig(Contract):
    model: str = "fake/none"
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int | None = None
    timeout_s: float = Field(default=30.0, gt=0.0)


class Thresholds(Contract):
    support_threshold: UnitFloat = 0.80
    refute_threshold: UnitFloat = 0.80
    historian_confidence_threshold: UnitFloat = 0.70
    source_reliability_threshold: UnitFloat = 0.60
    synthesizer_veto_threshold: UnitFloat = 0.80


class CostSummary(Contract):
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # Sum of per-role latencies (not wall-clock; Historian and Judge run concurrently).
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class RoleFailure(Contract):
    """A typed role failure (timeout, invalid output, provider down).

    Token and call totals cover attempts that failed validation, so a repaired-then-rejected
    role is not recorded as free. Defaults stay zero for failures that never reached a model.
    """

    role: str
    error_type: str
    message: str
    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class PolicyContext(Contract):
    candidate: CandidateClaim
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    related_claims: list[ExistingClaim] = Field(default_factory=list)
    historian: HistorianAssessment | None = None
    judge: JudgeAssessment | None = None
    synthesizer: SynthesizerRecommendation | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # Any entry forces escalation (R1).
    role_errors: list[RoleFailure] = Field(default_factory=list)
    run_config: RunConfig = Field(default_factory=RunConfig)
    # Caller-chosen key to force a deliberate re-run (a new decision) of identical inputs.
    idempotency_key: str | None = None
    now: AwareDatetime


class DecisionRecord(Contract):
    decision_id: UUID
    candidate_claim_id: UUID
    final_action: FinalAction
    # Action chosen by the rules before the Synthesizer veto step.
    rule_action: FinalAction
    policy_version: str
    # SHA-256 of the stable request inputs; `decision_id` is derived from it. A repository
    # compares it to tell a true replay from a reused ID with different inputs.
    request_hash: str = Field(min_length=64, max_length=64)
    thresholds: Thresholds
    rule_ids_fired: list[str]
    previous_claim_ids: list[UUID] = Field(default_factory=list)
    resulting_claim_id: UUID | None = None
    historian_assessment_id: UUID | None = None
    judge_assessment_id: UUID | None = None
    synthesizer_recommendation_id: UUID | None = None
    explanation: str
    created_at: AwareDatetime
    cost: CostSummary = Field(default_factory=CostSummary)


class StatusChange(Contract):
    claim_id: UUID
    new_status: ClaimStatus


class MemoryDelta(Contract):
    """What a committed decision changed in memory."""

    decision_id: UUID
    applied: bool
    # True when this decision_id was already committed and nothing new was written.
    replayed: bool = False
    created_claim_ids: list[UUID] = Field(default_factory=list)
    status_changes: list[StatusChange] = Field(default_factory=list)
    attached_evidence_ids: list[UUID] = Field(default_factory=list)
    note: str = ""


class CommitBundle(Contract):
    """Everything persisted for one decision, written atomically by `apply_decision`.

    Nothing is written to memory before the policy has decided; a commit stores the raw
    inputs, the role outputs, and the decision together, or not at all.
    """

    episode: EpisodeRecord | None = None
    candidate: CandidateClaim
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    decision: DecisionRecord
    historian: HistorianAssessment | None = None
    judge: JudgeAssessment | None = None
    synthesizer: SynthesizerRecommendation | None = None
    # The related claims the Historian and policy saw (memory "before"). Stored so reviewers and
    # experiment manifests can reconstruct the decision, and so a repository can detect that
    # memory changed underneath a decision before applying it.
    memory_before: list[ExistingClaim] = Field(default_factory=list)
    # Part of the request identity (see `compute_request_hash`); stored so it can be recomputed.
    idempotency_key: str | None = None
    # Why a role output is missing, and the exact configuration of the run.
    role_errors: list[RoleFailure] = Field(default_factory=list)
    run_config: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def _internally_consistent(self) -> Self:
        check_bundle_consistency(self)
        return self


def compute_request_hash(
    *,
    policy_version: str,
    candidate: CandidateClaim,
    evidence: list[EvidenceRecord],
    thresholds: Thresholds,
    run_config: RunConfig,
    idempotency_key: str | None,
) -> str:
    """SHA-256 of the STABLE inputs of a decision: what was asked, not what the roles answered.

    Included in full: the candidate, every evidence record (content, reliability, source,
    timestamps, metadata), thresholds, the whole run configuration (model, temperature, seed,
    timeout), the policy version, and the optional `idempotency_key`. Role outputs are excluded:
    assessment UUIDs, wording, latency, and cost differ between two runs of the same request
    against a real LLM, and a retry must still map to the same `decision_id`. The memory
    snapshot is excluded too: a committed decision changes memory (REVISE supersedes a claim)
    and the retry must still be recognized afterwards; the snapshot is stored separately as
    `CommitBundle.memory_before`. To re-evaluate the same candidate against newer memory, use
    a new `idempotency_key`. The timeout is included so a retry with a larger timeout can
    succeed instead of replaying a timeout escalation.
    """
    payload = {
        "policy_version": policy_version,
        "candidate": candidate.model_dump(mode="json"),
        "evidence": sorted(
            (e.model_dump(mode="json") for e in evidence), key=lambda e: e["evidence_id"]
        ),
        "thresholds": thresholds.model_dump(mode="json"),
        "run_config": run_config.model_dump(mode="json"),
        "idempotency_key": idempotency_key,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_decision_id(policy_version: str, request_hash: str) -> UUID:
    """Deterministic decision ID for a request hash under a policy version."""
    return uuid5(NAMESPACE_URL, f"{policy_version}:{request_hash}")


def check_bundle_consistency(bundle: CommitBundle) -> None:
    """Raise ValueError if the bundle contradicts itself.

    Pure and state-free, so every repository implementation can reuse it. Checks against the
    stored memory state (for example that related claim IDs exist) are the repository's job.
    """
    d, cand = bundle.decision, bundle.candidate
    if d.candidate_claim_id != cand.claim_id:
        raise ValueError("decision does not belong to the candidate")
    expected_hash = compute_request_hash(
        policy_version=d.policy_version,
        candidate=cand,
        evidence=bundle.evidence,
        thresholds=d.thresholds,
        run_config=bundle.run_config,
        idempotency_key=bundle.idempotency_key,
    )
    if d.request_hash != expected_hash:
        raise ValueError("decision.request_hash does not match the bundle's request fields")
    if d.decision_id != compute_decision_id(d.policy_version, d.request_hash):
        raise ValueError("decision.decision_id does not match its request_hash")
    outputs = (
        ("historian", bundle.historian, d.historian_assessment_id),
        ("judge", bundle.judge, d.judge_assessment_id),
    )
    for name, out, ref in outputs:
        if (out.assessment_id if out else None) != ref:
            raise ValueError(f"{name} output does not match the decision")
        if out is not None and out.candidate_claim_id != cand.claim_id:
            raise ValueError(f"{name} output belongs to a different candidate")
    syn = bundle.synthesizer
    if (syn.recommendation_id if syn else None) != d.synthesizer_recommendation_id:
        raise ValueError("synthesizer output does not match the decision")
    if syn is not None:
        if syn.candidate_claim_id != cand.claim_id:
            raise ValueError("synthesizer output belongs to a different candidate")
        if bundle.historian is None or bundle.judge is None:
            raise ValueError("synthesizer output present without historian and judge outputs")
        if syn.historian_assessment_id != bundle.historian.assessment_id:
            raise ValueError("synthesizer references a different historian assessment")
        if syn.judge_assessment_id != bundle.judge.assessment_id:
            raise ValueError("synthesizer references a different judge assessment")
    ids = [e.evidence_id for e in bundle.evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence_id in bundle")
    snapshot = {c.claim_id for c in bundle.memory_before}
    if len(snapshot) != len(bundle.memory_before):
        raise ValueError("duplicate claim in memory_before")
    if bundle.historian is not None:
        referenced = set(bundle.historian.related_claim_ids)
        if bundle.historian.supersedes_claim_id is not None:
            referenced.add(bundle.historian.supersedes_claim_id)
        if not referenced <= snapshot:
            raise ValueError("historian references claims that are not in memory_before")
    if bundle.judge is not None:
        known = {e.evidence_id for e in bundle.evidence}
        cited = set(bundle.judge.supporting_evidence_ids) | set(bundle.judge.refuting_evidence_ids)
        if not cited <= known:
            raise ValueError("judge cites evidence that is not in the bundle")


class AuditEntry(Contract):
    """One committed decision: the bundle (inputs, memory before, roles, decision) and what it
    actually changed in memory."""

    bundle: CommitBundle
    memory_delta: MemoryDelta | None = None


class AuditTrace(Contract):
    """Everything stored for one candidate."""

    candidate_claim_id: UUID
    entries: list[AuditEntry] = Field(default_factory=list)


# ------------------------------------------------------------------- API / requests


class HistorianRequest(Contract):
    candidate: CandidateClaim
    related_claims: list[ExistingClaim] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    run_config: RunConfig = Field(default_factory=RunConfig)


class JudgeRequest(Contract):
    """The Judge sees the claim and evidence only: no memory state, no labels, no action."""

    candidate: CandidateClaim
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    run_config: RunConfig = Field(default_factory=RunConfig)


class SynthesisRequest(Contract):
    candidate: CandidateClaim
    historian: HistorianAssessment
    judge: JudgeAssessment
    run_config: RunConfig = Field(default_factory=RunConfig)


class DecideRequest(Contract):
    candidate: CandidateClaim
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    mode: DecideMode = DecideMode.DRY_RUN
    run_config: RunConfig = Field(default_factory=RunConfig)
    thresholds: Thresholds | None = None
    # Same request and key -> same decision_id (idempotent retry), even after memory changed.
    # Pass a new key to deliberately re-evaluate the same candidate.
    idempotency_key: str | None = None

    @model_validator(mode="after")
    def _evidence_ids_unique(self) -> Self:
        ids = [e.evidence_id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence_id in request")
        return self


class DecideResponse(Contract):
    mode: DecideMode
    decision: DecisionRecord
    historian: HistorianAssessment | None = None
    judge: JudgeAssessment | None = None
    synthesizer: SynthesizerRecommendation | None = None
    memory_delta: MemoryDelta | None = None
    # Present for baselines that recommend an action directly (single_judge).
    # Absent for the council, which records Historian, Judge, and Synthesizer instead.
    baseline_audit: BaselineAudit | None = None
    role_errors: list[RoleFailure] = Field(default_factory=list)
    wall_latency_ms: int = Field(default=0, ge=0)
