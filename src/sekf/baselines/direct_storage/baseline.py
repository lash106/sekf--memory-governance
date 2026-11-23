"""`direct_storage` baseline (SPEC.md section 12). Owner: Krishna.

Every candidate becomes ACTIVE memory as it arrives: no time check, no source check, no
evidence check. It is the "just save it" lower bound, and it shows how memory goes wrong
(poisoning, stale values, duplicate claims) when nothing governs writes.
"""

from __future__ import annotations

from sekf.baselines.storage_common import StorageBaseline, StorageChoice
from sekf.contracts import DecideRequest, ExistingClaim, FinalAction

POLICY_VERSION = "direct-storage-v1"


class DirectStorageBaseline(StorageBaseline):
    baseline_name = "direct_storage"
    policy_version = POLICY_VERSION

    def choose(self, request: DecideRequest, related: list[ExistingClaim]) -> StorageChoice:
        return StorageChoice(
            action=FinalAction.PROMOTE,
            rule_id="DIRECT_STORAGE",
            explanation=(
                "direct_storage promotes every candidate; related claims, evidence and "
                "source reliability are not consulted."
            ),
        )
