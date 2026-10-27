"""Deterministic Historian v1 (SPEC.md section 18, Sprint 1). Owner: Mahalakshmi.

Rule-based provenance, temporal, and conflict analysis with no LLM calls. It decides
*when* and *who*, never whether the evidence supports the claim (that is the Judge's job).
The LLM assessment layer will be added on top of this in a later PR.

Decision outline for a candidate claim:

1. Related claims are the request's claims with the same ``subject_key`` and
   ``predicate_key``; only ``ACTIVE`` ones can be duplicated or superseded.
2. No active related claim -> ``NO_RELATED_CLAIM`` / ``NONE``.
3. Active claims whose validity window does not overlap the candidate's are true in their
   own period. If none overlaps -> ``NON_OVERLAPPING_VALIDITY`` / ``NONE``.
4. More than one overlapping active claim -> ambiguous, low confidence (policy escalates).
5. Against the single overlapping claim:
   - same value -> ``SAME_PERIOD`` + ``DUPLICATE`` (identical text) or ``PARAPHRASE``;
   - different value, candidate starts later and the old claim is open-ended
     -> ``NEWER_UPDATE`` / ``TEMPORAL_UPDATE`` and proposes superseding it;
   - different value, no validity bounds on either side -> order by observation time;
   - otherwise -> ``OVERLAPPING_VALIDITY`` or ``OLDER_INFORMATION`` / ``VALUE_CONFLICT``.

Known v1 limitation: ``SOURCE_CONFLICT`` needs reading what each source says, so it is not
detected here; the LLM layer will add it.
"""

from datetime import timedelta
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    ConflictType,
    ExistingClaim,
    HistorianAssessment,
    HistorianRequest,
    ProvenanceQuality,
    TemporalRelation,
)
from sekf.errors import RoleOutputInvalid
from sekf.historian.temporal import (
    IntervalRelation,
    TemporalOrder,
    classify_validity,
    provenance_band,
    provenance_score,
    temporal_order,
)

PROMPT_VERSION = "deterministic-historian-v1"
PROMPT_HASH = sha256(PROMPT_VERSION.encode()).hexdigest()

CONFIDENT = 0.90
NO_EVIDENCE = 0.80
AMBIGUOUS = 0.50  # below historian_confidence_threshold (0.70) on purpose -> R6 escalates
SAME_TIME_TOLERANCE = timedelta(minutes=1)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


class DeterministicHistorian:
    """Rule-based Historian. Reports ``llm_calls=0`` and provider ``deterministic``."""

    async def analyze(self, request: HistorianRequest) -> HistorianAssessment:
        """Analyze provenance, time, and conflict for ``request.candidate``.

        Example (SPEC.md section 4): an ACTIVE "closes at 17:00" claim valid from Jan 1 with
        no end, and a candidate "closes at 18:00" valid from Oct 1, give
        ``NEWER_UPDATE`` / ``TEMPORAL_UPDATE`` with ``supersedes_claim_id`` set to the
        17:00 claim. See ``tests/unit/test_deterministic_historian.py``.
        """
        cand = request.candidate
        related = [c for c in request.related_claims if self._same_key(cand, c)]
        active = [c for c in related if c.status is ClaimStatus.ACTIVE]

        score = provenance_score(e.reliability for e in request.evidence)
        quality = ProvenanceQuality(provenance_band(score).value)
        base_confidence = CONFIDENT if score is not None else NO_EVIDENCE

        overlapping = [c for c in active if self._overlaps(cand, c)]

        relation: TemporalRelation
        conflict: ConflictType
        supersedes: UUID | None = None
        confidence = base_confidence

        if not active:
            relation, conflict = TemporalRelation.NO_RELATED_CLAIM, ConflictType.NONE
            why = "No active claim with the same subject and predicate."
        elif not overlapping:
            relation, conflict = TemporalRelation.NON_OVERLAPPING_VALIDITY, ConflictType.NONE
            why = "Active related claims cover different periods; both can hold."
        elif len(overlapping) > 1:
            relation, conflict = TemporalRelation.UNKNOWN, ConflictType.VALUE_CONFLICT
            confidence = AMBIGUOUS
            why = f"{len(overlapping)} active claims overlap the candidate's period; ambiguous."
        else:
            prior = overlapping[0]
            relation, conflict, supersedes, why = self._compare(cand, prior)

        assessment = HistorianAssessment(
            assessment_id=self._assessment_id(request),
            candidate_claim_id=cand.claim_id,
            related_claim_ids=[c.claim_id for c in related],
            provenance_quality=quality,
            temporal_relation=relation,
            conflict_type=conflict,
            supersedes_claim_id=supersedes,
            source_reliability_score=score if score is not None else 0.0,
            confidence=confidence,
            rationale=f"{why} Provenance {quality.value} (best source {score}).",
            model="deterministic/historian",
            provider="deterministic",
            prompt_version=PROMPT_VERSION,
            prompt_hash=PROMPT_HASH,
            temperature=request.run_config.temperature,
            seed=request.run_config.seed,
            llm_calls=0,
        )
        validate_citations(assessment, request)
        return assessment

    @staticmethod
    def _same_key(cand: CandidateClaim, claim: ExistingClaim) -> bool:
        return (cand.subject_key, cand.predicate_key) == (claim.subject_key, claim.predicate_key)

    @staticmethod
    def _overlaps(cand: CandidateClaim, claim: ExistingClaim) -> bool:
        rel = classify_validity(cand.valid_from, cand.valid_to, claim.valid_from, claim.valid_to)
        # No bounds anywhere means "no stated period", which we treat as the same period.
        return rel is None or rel is IntervalRelation.OVERLAPPING

    @staticmethod
    def _compare(
        cand: CandidateClaim, prior: ExistingClaim
    ) -> tuple[TemporalRelation, ConflictType, UUID | None, str]:
        if _normalize(cand.object_text) == _normalize(prior.object_text):
            if _normalize(cand.claim_text) == _normalize(prior.claim_text):
                return (
                    TemporalRelation.SAME_PERIOD,
                    ConflictType.DUPLICATE,
                    None,
                    "Same value and wording as an active claim.",
                )
            return (
                TemporalRelation.SAME_PERIOD,
                ConflictType.PARAPHRASE,
                None,
                "Same value as an active claim, different wording.",
            )

        bounded = any(v is not None for v in (cand.valid_from, cand.valid_to))
        bounded = bounded or any(v is not None for v in (prior.valid_from, prior.valid_to))
        if not bounded:
            order = temporal_order(cand.observed_at, prior.updated_at, SAME_TIME_TOLERANCE)
            if order is TemporalOrder.NEWER:
                return (
                    TemporalRelation.NEWER_UPDATE,
                    ConflictType.TEMPORAL_UPDATE,
                    prior.claim_id,
                    "Different value observed after the active claim; no validity dates given.",
                )
            if order is TemporalOrder.OLDER:
                return (
                    TemporalRelation.OLDER_INFORMATION,
                    ConflictType.VALUE_CONFLICT,
                    None,
                    "Different value observed before the active claim was last updated.",
                )
            return (
                TemporalRelation.SAME_PERIOD,
                ConflictType.VALUE_CONFLICT,
                None,
                "Different value reported at the same time as the active claim.",
            )

        takes_over = (
            prior.valid_to is None
            and cand.valid_from is not None
            and (prior.valid_from is None or cand.valid_from > prior.valid_from)
        )
        if takes_over:
            return (
                TemporalRelation.NEWER_UPDATE,
                ConflictType.TEMPORAL_UPDATE,
                prior.claim_id,
                "Different value starting after the open-ended active claim; proposes closing it.",
            )
        starts_earlier = (
            cand.valid_from is not None
            and prior.valid_from is not None
            and cand.valid_from < prior.valid_from
        )
        if starts_earlier:
            return (
                TemporalRelation.OLDER_INFORMATION,
                ConflictType.VALUE_CONFLICT,
                None,
                "Different value for a period starting before the active claim's.",
            )
        return (
            TemporalRelation.OVERLAPPING_VALIDITY,
            ConflictType.VALUE_CONFLICT,
            None,
            "Different value claimed for an overlapping period; unresolved conflict.",
        )

    @staticmethod
    def _assessment_id(request: HistorianRequest) -> UUID:
        parts = [PROMPT_VERSION, str(request.candidate.claim_id)]
        parts += [str(e.evidence_id) + e.content_hash for e in request.evidence]
        parts += [str(c.claim_id) + c.updated_at.isoformat() for c in request.related_claims]
        return uuid5(NAMESPACE_URL, "historian:" + ":".join(parts))


def validate_citations(assessment: HistorianAssessment, request: HistorianRequest) -> None:
    """Raise ``RoleOutputInvalid`` if the assessment cites claims the request did not contain.

    Shared with the future LLM Historian: a model may hallucinate claim IDs, and the policy
    must never supersede a claim the Historian was not shown.
    """
    known = {c.claim_id: c for c in request.related_claims}
    unknown = [cid for cid in assessment.related_claim_ids if cid not in known]
    if unknown:
        raise RoleOutputInvalid(f"Historian cited unknown related claim IDs: {unknown}")
    if assessment.candidate_claim_id != request.candidate.claim_id:
        raise RoleOutputInvalid("Historian assessed a different candidate than requested")
    sup = assessment.supersedes_claim_id
    if sup is not None:
        if sup not in known:
            raise RoleOutputInvalid(f"Historian proposes superseding unknown claim {sup}")
        if known[sup].status is not ClaimStatus.ACTIVE:
            raise RoleOutputInvalid(f"Historian proposes superseding non-active claim {sup}")
