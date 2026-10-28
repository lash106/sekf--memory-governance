"""Shailen's Sprint 0 `datasets/dev/shailen` cases load and are internally coherent."""

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
    (Path(__file__).resolve().parents[2] / "datasets" / "dev" / "shailen").glob("*.json")
)


def test_five_cases_with_distinct_gold_actions() -> None:
    assert len(CASES) == 5
    actions = {json.loads(p.read_text())["gold"]["final_action"] for p in CASES}
    assert actions == {"PROMOTE", "REJECT", "REVISE", "QUARANTINE", "KEEP_EPISODIC"}


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_case_is_valid_and_delta_is_coherent(path: Path) -> None:
    case: dict[str, Any] = json.loads(path.read_text())
    assert case["case_id"] == path.stem and case["split"] == "dev"
    assert case["annotators"] and case["annotation_notes"]
    memory = [ExistingClaim.model_validate(c) for c in case["initial_memory"]]
    cand = CandidateClaim.model_validate(case["candidate"])
    evidence = [EvidenceRecord.model_validate(e) for e in case["evidence"]]
    gold = case["gold"]
    JudgeLabel(gold["judge_label"])
    TemporalRelation(gold["temporal_relation"])
    ConflictType(gold["conflict_type"])
    action = FinalAction(gold["final_action"])

    assert len({e.evidence_id for e in evidence}) == len(evidence)
    for claim in memory:  # every remembered claim is about the same thing as the candidate
        assert (claim.subject_key, claim.predicate_key) == (cand.subject_key, cand.predicate_key)

    delta = {k: {str(v) for v in vs} for k, vs in gold["expected_memory_delta"].items()}
    active = {str(c.claim_id) for c in memory if c.status is ClaimStatus.ACTIVE}
    assert active >= delta["superseded_claim_ids"]
    assert {str(e.evidence_id) for e in evidence} >= delta["attached_evidence_ids"]

    target = {
        FinalAction.PROMOTE: "new_active_claim_ids",
        FinalAction.REVISE: "new_active_claim_ids",
        FinalAction.QUARANTINE: "quarantined_claim_ids",
        FinalAction.REJECT: "rejected_claim_ids",
    }.get(action)
    if target:
        assert delta[target] == {str(cand.claim_id)}
    else:  # KEEP_EPISODIC / ESCALATE: no claim state changes at all
        assert not any(delta.values())
    if action is FinalAction.REVISE:
        assert len(delta["superseded_claim_ids"]) == 1
    else:
        assert not delta["superseded_claim_ids"]
