"""Every dev case is well formed and obeys datasets/ANNOTATION_GUIDE.md.

These are rules about *labels agreeing with each other* (for example hedged evidence cannot be
labeled SUPPORTED, and a revision needs exactly one active claim to replace). They catch
annotator slips. They do not decide a gold label, and the policy code is never consulted.
"""

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

DEV = Path(__file__).resolve().parents[2] / "datasets" / "dev"
PATHS = sorted(p for p in DEV.rglob("*.json"))
A = FinalAction


def load(path: Path) -> dict[str, Any]:
    case: dict[str, Any] = json.loads(path.read_text())
    return case


def test_there_are_enough_cases_and_ids_are_unique() -> None:
    assert len(PATHS) >= 36
    ids = [load(p)["case_id"] for p in PATHS]
    assert len(ids) == len(set(ids))
    claim_ids = [load(p)["candidate"]["claim_id"] for p in PATHS]
    assert len(claim_ids) == len(set(claim_ids)), "two cases share a candidate claim_id"


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.stem)
def test_case_is_well_formed(path: Path) -> None:
    case = load(path)
    assert case["case_id"] == path.stem and case["split"] == "dev"
    assert case["annotators"] and case["annotation_notes"].strip()
    memory = [ExistingClaim.model_validate(c) for c in case["initial_memory"]]
    cand = CandidateClaim.model_validate(case["candidate"])
    evidence = [EvidenceRecord.model_validate(e) for e in case["evidence"]]
    assert len({e.evidence_id for e in evidence}) == len(evidence)
    assert len({c.claim_id for c in memory}) == len(memory)
    assert cand.claim_id not in {c.claim_id for c in memory}
    for claim in memory:
        assert (claim.subject_key, claim.predicate_key) == (cand.subject_key, cand.predicate_key)


@pytest.mark.parametrize("path", PATHS, ids=lambda p: p.stem)
def test_labels_agree_with_each_other_and_with_the_guide(path: Path) -> None:
    case = load(path)
    gold = case["gold"]
    judge = JudgeLabel(gold["judge_label"])
    relation = TemporalRelation(gold["temporal_relation"])
    conflict = ConflictType(gold["conflict_type"])
    action = FinalAction(gold["final_action"])
    memory = [ExistingClaim.model_validate(c) for c in case["initial_memory"]]
    evidence = [EvidenceRecord.model_validate(e) for e in case["evidence"]]
    active = [c for c in memory if c.status is ClaimStatus.ACTIVE]

    # Step 1 -> step 3 of the guide: what the evidence says limits what can happen.
    assert (judge is JudgeLabel.REFUTED) == (action is A.REJECT)
    assert (judge is JudgeLabel.INSUFFICIENT_EVIDENCE) == (action is A.KEEP_EPISODIC)
    assert (judge is JudgeLabel.INVALID_INPUT) == (action is A.ESCALATE_FOR_REVIEW)
    if judge is JudgeLabel.MIXED_EVIDENCE:
        assert action is A.QUARANTINE
    if action in (A.PROMOTE, A.REVISE, A.ATTACH_EVIDENCE):
        assert judge is JudgeLabel.SUPPORTED
    if action is A.QUARANTINE:
        assert judge in (JudgeLabel.SUPPORTED, JudgeLabel.MIXED_EVIDENCE)

    # Historian labels.
    assert (action is A.ATTACH_EVIDENCE) == (
        conflict in (ConflictType.DUPLICATE, ConflictType.PARAPHRASE)
    )
    if action is A.REVISE:
        assert conflict is ConflictType.TEMPORAL_UPDATE and len(active) == 1
    if relation is TemporalRelation.NON_OVERLAPPING_VALIDITY:
        assert conflict is ConflictType.NONE  # guide rule B1
    if relation is TemporalRelation.NO_RELATED_CLAIM:
        assert not active
    if conflict is ConflictType.SOURCE_CONFLICT:  # sources of similar standing disagree
        assert len({e.source_id for e in evidence}) >= 2

    # expected_memory_delta, exactly as the guide's table.
    cand_id = case["candidate"]["claim_id"]
    expected = {
        A.PROMOTE: {"new_active_claim_ids": {cand_id}},
        A.REVISE: {
            "new_active_claim_ids": {cand_id},
            "superseded_claim_ids": {str(c.claim_id) for c in active},
        },
        A.QUARANTINE: {"quarantined_claim_ids": {cand_id}},
        A.REJECT: {"rejected_claim_ids": {cand_id}},
        A.ATTACH_EVIDENCE: {"attached_evidence_ids": {str(e.evidence_id) for e in evidence}},
    }.get(action, {})
    delta = {k: set(v) for k, v in gold["expected_memory_delta"].items()}
    for name, got in delta.items():
        assert got == expected.get(name, set()), f"{name}: {got} != {expected.get(name, set())}"
