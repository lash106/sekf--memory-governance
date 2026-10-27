"""Sanity-check Sai Chetan `datasets/dev` cases load and cover required categories."""

from __future__ import annotations

import json
from pathlib import Path

from sekf.contracts import CandidateClaim, EvidenceRecord, ExistingClaim, JudgeLabel

DEV = Path(__file__).resolve().parents[2] / "datasets" / "dev"

REQUIRED_CATEGORIES = {
    "supported_new_claim",
    "unsupported_claim",
    "refuted_claim",
    "duplicate_paraphrase",
    "reliable_newer_correction",
    "unreliable_newer_contradiction",
    "non_overlapping_temporal_validity",
    "overlapping_temporal_conflict",
    "conflicting_sources_unequal_reliability",
    "conflicting_sources_similar_reliability",
    "insufficient_evidence",
    "malformed_input",
    "prompt_injection_memory_poisoning",
}


def test_at_least_twenty_dev_cases() -> None:
    files = sorted(DEV.glob("*.json"))
    assert len(files) >= 20


def test_cases_match_generator_and_memory_deltas() -> None:
    from generate_sai_chetan_cases import build_cases

    generated = {case["case_id"]: case for case in build_cases()}
    assert generated == {case["case_id"]: case for case in build_cases()}
    on_disk = {}
    for path in sorted(DEV.glob("*.json")):
        on_disk[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == set(generated)
    for case_id, case in generated.items():
        assert on_disk[case_id] == case
        action = case["gold"]["final_action"]
        delta = case["gold"]["expected_memory_delta"]
        candidate_id = case["candidate"]["claim_id"]
        active_ids = [
            item["claim_id"] for item in case["initial_memory"] if item["status"] == "ACTIVE"
        ]
        evidence_ids = [item["evidence_id"] for item in case["evidence"]]
        if action == "PROMOTE":
            assert delta["new_active_claim_ids"] == [candidate_id]
        elif action == "REVISE":
            assert delta["superseded_claim_ids"] == active_ids
            assert delta["new_active_claim_ids"] == [candidate_id]
            assert active_ids
        elif action == "REJECT":
            assert delta["rejected_claim_ids"] == [candidate_id]
        elif action == "QUARANTINE":
            assert delta["quarantined_claim_ids"] == [candidate_id]
        elif action == "ATTACH_EVIDENCE":
            assert delta["attached_evidence_ids"] == evidence_ids
            assert evidence_ids
        else:
            assert action in {"KEEP_EPISODIC", "ESCALATE_FOR_REVIEW"}
            assert not any(delta.values())


def test_cases_parse_and_cover_categories() -> None:
    categories: set[str] = set()
    labels: set[str] = set()
    for path in sorted(DEV.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["annotators"] == ["Sai Chetan Muppalla"]
        assert data["split"] == "dev"
        categories.add(data["category"])
        labels.add(data["gold"]["judge_label"])
        CandidateClaim.model_validate(data["candidate"])
        for item in data["evidence"]:
            EvidenceRecord.model_validate(item)
        for item in data["initial_memory"]:
            ExistingClaim.model_validate(item)
        JudgeLabel(data["gold"]["judge_label"])
    assert REQUIRED_CATEGORIES <= categories
    assert {
        "SUPPORTED",
        "REFUTED",
        "INSUFFICIENT_EVIDENCE",
        "MIXED_EVIDENCE",
        "INVALID_INPUT",
    } <= labels
