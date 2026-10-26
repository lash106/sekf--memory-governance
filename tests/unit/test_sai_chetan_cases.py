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
