"""Every labeled case uses a known category (SPEC.md section 13.3 plus documented extras).

A misspelled or shortened category splits one group into two in per-category results
(for example ``prompt_injection`` vs ``prompt_injection_memory_poisoning``).
"""

import json
from pathlib import Path

import pytest

DATASETS = Path(__file__).resolve().parents[2] / "datasets"

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

# Extra categories beyond the required 13, each backed by a rule in ANNOTATION_GUIDE.md.
EXTRA_CATEGORIES = {
    "supported_by_weak_source",  # guide B4: weak source, no conflict -> QUARANTINE (A3)
}

KNOWN_CATEGORIES = REQUIRED_CATEGORIES | EXTRA_CATEGORIES

CASE_FILES = sorted(
    path
    for split in ("dev", "test")
    for path in (DATASETS / split).rglob("*.json")
    if (DATASETS / split).is_dir()
)


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda p: str(p.relative_to(DATASETS)))
def test_case_category_is_known(path: Path) -> None:
    case = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(case, dict) or "category" not in case:
        pytest.skip("not a labeled case file")
    assert case["category"] in KNOWN_CATEGORIES, (
        f"{path.name}: category {case['category']!r} is not in SPEC.md 13.3 "
        "or EXTRA_CATEGORIES; fix the spelling or add it with its guide rule"
    )
