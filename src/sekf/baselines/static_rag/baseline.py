"""`static_rag` baseline (SPEC.md section 12). Owner: Krishna.

Retrieval-augmented answering over a memory that is never governed: related claims and the
submitted evidence are ranked by word overlap with the candidate and can be read back, but
no claim is ever promoted, revised or rejected. As a memory-action system it therefore
always returns KEEP_EPISODIC, so it can never correct a stale claim or store a new fact.
Retrieval is lexical and deterministic (no embeddings, no LLM) so runs are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sekf.baselines.storage_common import StorageBaseline, StorageChoice
from sekf.contracts import CandidateClaim, DecideRequest, EvidenceRecord, ExistingClaim, FinalAction

POLICY_VERSION = "static-rag-v1"
DEFAULT_TOP_K = 3
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Retrieved:
    """One retrieved item: a stored claim or a submitted evidence passage."""

    kind: str  # "claim" or "evidence"
    item_id: UUID
    score: float
    text: str


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(query: set[str], text: str) -> float:
    words = _words(text)
    if not query or not words:
        return 0.0
    return len(query & words) / len(query | words)


def retrieve(
    candidate: CandidateClaim,
    claims: list[ExistingClaim],
    evidence: list[EvidenceRecord],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Retrieved]:
    """Rank claims and evidence by Jaccard word overlap with the candidate (ties by id)."""

    query = _words(candidate.claim_text)
    items = [
        Retrieved("claim", c.claim_id, _overlap(query, c.claim_text), c.claim_text) for c in claims
    ]
    items += [
        Retrieved("evidence", e.evidence_id, _overlap(query, e.content), e.content)
        for e in evidence
    ]
    items.sort(key=lambda item: (-item.score, item.kind, str(item.item_id)))
    return [item for item in items[:top_k] if item.score > 0.0]


class StaticRagBaseline(StorageBaseline):
    baseline_name = "static_rag"
    policy_version = POLICY_VERSION

    def __init__(self, *args: object, top_k: int = DEFAULT_TOP_K, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._top_k = top_k

    def retrieve(self, request: DecideRequest, related: list[ExistingClaim]) -> list[Retrieved]:
        return retrieve(request.candidate, related, request.evidence, top_k=self._top_k)

    def choose(self, request: DecideRequest, related: list[ExistingClaim]) -> StorageChoice:
        hits = self.retrieve(request, related)
        shown = ", ".join(f"{h.kind}:{h.item_id}" for h in hits) or "nothing"
        return StorageChoice(
            action=FinalAction.KEEP_EPISODIC,
            rule_id="STATIC_RAG",
            explanation=(
                f"static_rag retrieved {shown} but never writes governed memory; "
                "the candidate is kept as an episode only."
            ),
        )
