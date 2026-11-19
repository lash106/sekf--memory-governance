"""Append-only Markdown projection of a committed decision.

SQLite remains the source of truth. These files are a readable copy for a demo brief:
L0 episodes, L1 evidence, L2 claims or quarantine, L3 audit. A decision id already
present in the audit file is not written again.
"""

from __future__ import annotations

from pathlib import Path

from sekf.contracts import CommitBundle, FinalAction

_AUDIT = "L3_AUDIT_LOG.md"
_CLAIMS = "L2_CLAIMS.md"
_QUARANTINE = "L2_QUARANTINE.md"
_EVIDENCE = "L1_EVIDENCE.md"
_EPISODES = "L0_EPISODES.md"


def project_bundle(root: Path, bundle: CommitBundle) -> None:
    """Append one decision. Raises OSError when the files cannot be written."""

    root.mkdir(parents=True, exist_ok=True)
    audit = root / _AUDIT
    marker = f"decision:{bundle.decision.decision_id}"
    if audit.is_file() and marker in audit.read_text(encoding="utf-8"):
        return
    decision = bundle.decision
    lines = [
        f"## {marker}",
        f"- action: {decision.final_action.value}",
        f"- policy: {decision.policy_version}",
        f"- rules: {', '.join(decision.rule_ids_fired)}",
        f"- candidate: {bundle.candidate.claim_text}",
        f"- explanation: {decision.explanation}",
        "",
    ]
    _append(audit, "\n".join(lines))
    if bundle.episode is not None:
        _append(
            root / _EPISODES,
            f"- {bundle.episode.episode_id}: {bundle.episode.content}\n",
        )
    for item in bundle.evidence:
        _append(
            root / _EVIDENCE,
            f"- {item.evidence_id} ({item.source_type}): {item.content}\n",
        )
    claim_line = (
        f"- {bundle.candidate.claim_id} {decision.final_action.value}: "
        f"{bundle.candidate.claim_text}\n"
    )
    held = decision.final_action in {FinalAction.QUARANTINE, FinalAction.ESCALATE_FOR_REVIEW}
    if held:
        clause = decision.compliance_outcome or "no R_COMPLIANCE clause fired"
        needed = decision.next_evidence_required or (
            "Evidence that clears the quarantine reason in this decision."
        )
        rules = ", ".join(decision.rule_ids_fired) or "none"
        _append(
            root / _QUARANTINE,
            (
                f"{claim_line}"
                f"  - failing_clause: {clause} (rules: {rules})\n"
                f"  - next_evidence_required: {needed}\n"
            ),
        )
    elif decision.final_action in {
        FinalAction.PROMOTE,
        FinalAction.REVISE,
        FinalAction.ATTACH_EVIDENCE,
        FinalAction.REJECT,
    }:
        extra = ""
        if decision.compliance_outcome:
            extra = f"  - compliance_clause: {decision.compliance_outcome}\n"
        _append(root / _CLAIMS, claim_line + extra)


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
