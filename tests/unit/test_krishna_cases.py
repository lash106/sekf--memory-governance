"""Krishna's Sprint 0 `datasets/dev/krishna` cases load and have coherent memory deltas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    ConflictType,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeLabel,
    TemporalRelation,
)

CASES = sorted(
    (Path(__file__).resolve().parents[2] / "datasets" / "dev" / "krishna").glob("*.json")
)


def test_five_cases_present() -> None:
    assert len(CASES) >= 5


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_is_valid_and_delta_is_coherent(path: Path) -> None:
    case: dict[str, Any] = json.loads(path.read_text())
    assert case["case_id"] == path.stem and case["split"] == "dev"
    memory = [ExistingClaim.model_validate(c) for c in case["initial_memory"]]
    cand = CandidateClaim.model_validate(case["candidate"])
    evidence = [EvidenceRecord.model_validate(e) for e in case["evidence"]]
    gold = case["gold"]
    JudgeLabel(gold["judge_label"])
    TemporalRelation(gold["temporal_relation"])
    ConflictType(gold["conflict_type"])
    action = FinalAction(gold["final_action"])

    active = {c.claim_id for c in memory if c.status is ClaimStatus.ACTIVE}
    delta = {k: {str(v) for v in vs} for k, vs in gold["expected_memory_delta"].items()}
    assert {str(a) for a in active} >= delta["superseded_claim_ids"]
    assert {str(e.evidence_id) for e in evidence} >= delta["attached_evidence_ids"]
    expected_status = {
        FinalAction.PROMOTE: "new_active_claim_ids",
        FinalAction.REVISE: "new_active_claim_ids",
        FinalAction.QUARANTINE: "quarantined_claim_ids",
        FinalAction.REJECT: "rejected_claim_ids",
    }.get(action)
    if expected_status:
        assert delta[expected_status] == {str(cand.claim_id)}
    if action is FinalAction.REVISE:
        assert len(delta["superseded_claim_ids"]) == 1
    if action is FinalAction.ATTACH_EVIDENCE:
        assert delta["attached_evidence_ids"] and not delta["new_active_claim_ids"]
    for c in memory:  # every key in initial memory matches the candidate's
        assert (c.subject_key, c.predicate_key) == (cand.subject_key, cand.predicate_key)
