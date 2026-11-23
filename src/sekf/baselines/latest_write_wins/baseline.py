"""`latest_write_wins` baseline (SPEC.md section 12). Owner: Krishna.

The newest value replaces the current one. A candidate observed after the current ACTIVE
claim was last updated REVISEs it; an older one is kept as an episode only. It checks time
and nothing else, so a newer but unreliable or poisoned claim still overwrites a good one.
"""

from __future__ import annotations

from sekf.baselines.storage_common import StorageBaseline, StorageChoice, active_related
from sekf.contracts import DecideRequest, ExistingClaim, FinalAction

POLICY_VERSION = "latest-write-wins-v1"


class LatestWriteWinsBaseline(StorageBaseline):
    baseline_name = "latest_write_wins"
    policy_version = POLICY_VERSION

    def choose(self, request: DecideRequest, related: list[ExistingClaim]) -> StorageChoice:
        active = active_related(related)
        if not active:
            return StorageChoice(
                action=FinalAction.PROMOTE,
                rule_id="LWW_NO_ACTIVE",
                explanation="No ACTIVE claim for this subject and predicate; storing the candidate.",  # noqa: E501
            )
        # With several ACTIVE claims, compare against the most recently updated one.
        current = max(active, key=lambda claim: (claim.updated_at, str(claim.claim_id)))
        if request.candidate.observed_at > current.updated_at:
            return StorageChoice(
                action=FinalAction.REVISE,
                rule_id="LWW_NEWER",
                target_claim_id=current.claim_id,
                explanation=(
                    "Candidate was observed after the current ACTIVE claim was updated; "
                    "the newest write wins."
                ),
            )
        return StorageChoice(
            action=FinalAction.KEEP_EPISODIC,
            rule_id="LWW_NOT_NEWER",
            explanation=(
                "Candidate is not newer than the current ACTIVE claim; kept as an episode only."
            ),
        )
