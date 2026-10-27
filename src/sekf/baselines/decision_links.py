"""Which claim ids a baseline action would touch. Deterministic; no LLM calls.

Owner: Sai Chetan. Used so dry-run decisions name the same claims a later commit would.
"""

from __future__ import annotations

from uuid import UUID

from sekf.contracts import ClaimStatus, ExistingClaim, FinalAction
from sekf.errors import RoleOutputInvalid


def action_claim_links(
    action: FinalAction,
    *,
    candidate_id: UUID,
    related: list[ExistingClaim],
    target_claim_id: UUID | None = None,
) -> tuple[list[UUID], UUID | None]:
    """Return `(previous_claim_ids, resulting_claim_id)` for `action`.

    `previous_claim_ids` lists only claims the action changes. `resulting_claim_id` is the
    claim the action stores or updates, or None when nothing becomes governed knowledge.
    """

    if action in (FinalAction.KEEP_EPISODIC, FinalAction.ESCALATE_FOR_REVIEW):
        return [], None
    if action is FinalAction.PROMOTE:
        return [], candidate_id
    if action in (FinalAction.REJECT, FinalAction.QUARANTINE):
        return [], candidate_id
    if action is FinalAction.REVISE:
        target = _active_target(related, target_claim_id, action)
        return [target], candidate_id
    if action is FinalAction.ATTACH_EVIDENCE:
        target = _active_target(related, target_claim_id, action)
        return [target], target
    return [], None


def _active_target(
    related: list[ExistingClaim], target_claim_id: UUID | None, action: FinalAction
) -> UUID:
    active = [claim.claim_id for claim in related if claim.status is ClaimStatus.ACTIVE]
    if target_claim_id is not None:
        if target_claim_id not in active:
            raise RoleOutputInvalid(
                f"{action.value} target_claim_id {target_claim_id} is not an ACTIVE related claim"
            )
        return target_claim_id
    if len(active) == 1:
        return active[0]
    raise RoleOutputInvalid(
        f"{action.value} needs target_claim_id when related ACTIVE claims are not exactly one "
        f"(found {len(active)})"
    )
