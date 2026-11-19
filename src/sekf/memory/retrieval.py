"""Entity-scoped BM25 retrieval. Deterministic. No embeddings and no network.

The repository `search_related_claims` method stays an exact subject+predicate match.
This helper is for the separate Retrieval Recall@K metric, so a missed neighbor is not
counted as a Historian or policy error.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from sekf.contracts import CandidateClaim, ExistingClaim

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25(
    query: list[str], document: list[str], *, avg_len: float, df: Counter[str], n_docs: int
) -> float:
    if not query or not document:
        return 0.0
    tf = Counter(document)
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term in set(query):
        if term not in tf:
            continue
        idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        freq = tf[term]
        denom = freq + k1 * (1 - b + b * len(document) / avg_len)
        score += idf * (freq * (k1 + 1)) / denom
    return score


def hybrid_search(
    candidate: CandidateClaim, claims: list[ExistingClaim], *, k: int = 5
) -> list[ExistingClaim]:
    """Same-subject claims, exact predicate matches first, then BM25, truncated to `k`."""

    if k < 1:
        raise ValueError("k must be >= 1")
    pool = [
        claim
        for claim in claims
        if claim.subject_key == candidate.subject_key and claim.claim_id != candidate.claim_id
    ]
    exact = [claim for claim in pool if claim.predicate_key == candidate.predicate_key]
    rest = [claim for claim in pool if claim.predicate_key != candidate.predicate_key]
    docs = [_tokens(f"{claim.claim_text} {claim.object_text}") for claim in rest]
    query = _tokens(f"{candidate.claim_text} {candidate.object_text} {candidate.predicate_key}")
    avg_len = (sum(len(doc) for doc in docs) / len(docs)) if docs else 1.0
    df: Counter[str] = Counter(term for doc in docs for term in set(doc))
    ranked = sorted(
        rest,
        key=lambda claim: _bm25(
            query,
            _tokens(f"{claim.claim_text} {claim.object_text}"),
            avg_len=avg_len,
            df=df,
            n_docs=len(docs),
        ),
        reverse=True,
    )
    ordered = exact + ranked
    return ordered[:k]


def retrieval_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str]) -> float | None:
    """Fraction of relevant ids present in the retrieved list. None when nothing is relevant."""

    relevant = [item for item in relevant_ids if item]
    if not relevant:
        return None
    found = set(retrieved_ids)
    return sum(1 for item in relevant if item in found) / len(relevant)
