"""Episodic visibility. Claims are never deleted.

Candidate-status rows older than the TTL drop out of a customer brief only.
`get_claim`, audit, and `search_related_claims` still return them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sekf.contracts import ClaimStatus, ExistingClaim


def claim_visible_in_brief(claim: ExistingClaim, now: datetime, ttl_days: int) -> bool:
    """True when a brief should still show this claim."""

    if ttl_days < 1:
        raise ValueError("ttl_days must be >= 1")
    if claim.status is not ClaimStatus.CANDIDATE:
        return True
    return claim.updated_at + timedelta(days=ttl_days) > now
