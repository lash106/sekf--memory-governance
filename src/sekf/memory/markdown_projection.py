"""Append-only Markdown projection of a committed decision.

SQLite remains the source of truth. These files are a readable copy for a demo brief:
L0 episodes, L1 evidence, L2 claims or quarantine, L3 audit.

Idempotency: the audit file is the *completion* marker and is written last. Each
earlier file also carries the same `decision:{id}` token so a retry after a partial
write skips layers that already landed and finishes the ones that did not.
"""

from __future__ import annotations

from pathlib import Path

from sekf.contracts import CommitBundle, FinalAction

_AUDIT = "L3_AUDIT_LOG.md"
_CLAIMS = "L2_CLAIMS.md"
_QUARANTINE = "L2_QUARANTINE.md"
_EVIDENCE = "L1_EVIDENCE.md"
_EPISODES = "L0_EPISODES.md"

_CLAIM_ACTIONS = frozenset(
    {
        FinalAction.PROMOTE,
        FinalAction.REVISE,
        FinalAction.ATTACH_EVIDENCE,
        FinalAction.QUARANTINE,
        FinalAction.REJECT,
    }
)


def project_bundle(root: Path, bundle: CommitBundle) -> None:
    """Append one decision. Raises OSError when the files cannot be written.

    Non-audit layers are written first. The L3 audit entry is written last and is
    the only signal that the projection finished. A retry after a crash mid-write
    therefore still attempts the missing layers.
    """

    root.mkdir(parents=True, exist_ok=True)
    marker = f"decision:{bundle.decision.decision_id}"
    if _projection_complete(root, bundle, marker):
        return

    decision = bundle.decision
    pending: list[tuple[Path, str]] = []

    if bundle.episode is not None:
        pending.append(
            (
                root / _EPISODES,
                f"- {marker} {bundle.episode.episode_id}: {bundle.episode.content}\n",
            )
        )
    for item in bundle.evidence:
        pending.append(
            (
                root / _EVIDENCE,
                f"- {marker} {item.evidence_id} ({item.source_type}): {item.content}\n",
            )
        )
    if decision.final_action in _CLAIM_ACTIONS:
        target = _QUARANTINE if decision.final_action is FinalAction.QUARANTINE else _CLAIMS
        pending.append(
            (
                root / target,
                f"- {marker} {bundle.candidate.claim_id} {decision.final_action.value}: "
                f"{bundle.candidate.claim_text}\n",
            )
        )

    for path, text in pending:
        if not _file_has_marker(path, marker):
            _append(path, text)

    audit_body = "\n".join(
        [
            f"## {marker}",
            f"- action: {decision.final_action.value}",
            f"- policy: {decision.policy_version}",
            f"- rules: {', '.join(decision.rule_ids_fired)}",
            f"- candidate: {bundle.candidate.claim_text}",
            f"- explanation: {decision.explanation}",
            "",
        ]
    )
    # Completion marker last: only after L0/L1/L2 succeeded.
    if not _file_has_marker(root / _AUDIT, marker):
        _append(root / _AUDIT, audit_body)


def _projection_complete(root: Path, bundle: CommitBundle, marker: str) -> bool:
    """True only when the audit marker and every required layer marker are present."""

    if not _file_has_marker(root / _AUDIT, marker):
        return False
    if bundle.episode is not None and not _file_has_marker(root / _EPISODES, marker):
        return False
    if bundle.evidence and not _file_has_marker(root / _EVIDENCE, marker):
        return False
    action = bundle.decision.final_action
    if action in _CLAIM_ACTIONS:
        target = root / (_QUARANTINE if action is FinalAction.QUARANTINE else _CLAIMS)
        if not _file_has_marker(target, marker):
            return False
    return True


def _file_has_marker(path: Path, marker: str) -> bool:
    return path.is_file() and marker in path.read_text(encoding="utf-8")


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
