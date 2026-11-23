"""Expected-versus-actual memory-delta comparison."""

from __future__ import annotations

from uuid import uuid4

import pytest

from sekf.contracts import ClaimStatus, MemoryDelta, StatusChange
from sekf.evaluation.memory_delta import (
    BUCKETS,
    DeltaSet,
    compare_deltas,
    summarize_comparisons,
)


def _gold(**kwargs: list[str]) -> dict[str, list[str]]:
    return {name: kwargs.get(name, []) for name in BUCKETS}


def test_from_gold_rejects_unknown_buckets() -> None:
    with pytest.raises(ValueError, match="unknown delta buckets"):
        DeltaSet.from_gold({"made_up": []})


def test_from_memory_delta_maps_each_status_to_its_bucket() -> None:
    a, b, c, d, e = (uuid4() for _ in range(5))
    delta = MemoryDelta(
        decision_id=uuid4(),
        applied=True,
        status_changes=[
            StatusChange(claim_id=a, new_status=ClaimStatus.SUPERSEDED),
            StatusChange(claim_id=b, new_status=ClaimStatus.ACTIVE),
            StatusChange(claim_id=c, new_status=ClaimStatus.QUARANTINED),
            StatusChange(claim_id=d, new_status=ClaimStatus.REJECTED),
            StatusChange(claim_id=e, new_status=ClaimStatus.CANDIDATE),
        ],
        attached_evidence_ids=[uuid4()],
    )
    actual = DeltaSet.from_memory_delta(delta)
    assert actual.get("superseded_claim_ids") == {str(a)}
    assert actual.get("new_active_claim_ids") == {str(b)}
    assert actual.get("quarantined_claim_ids") == {str(c)}
    assert actual.get("rejected_claim_ids") == {str(d)}
    assert len(actual.get("attached_evidence_ids")) == 1
    assert actual.review_claim_ids == {str(e)}  # reported, not compared


def test_no_write_is_the_empty_delta() -> None:
    actual = DeltaSet.from_memory_delta(None)
    assert all(actual.get(name) == frozenset() for name in BUCKETS)


def test_exact_match_and_missing_unexpected() -> None:
    old, new, wrong = str(uuid4()), str(uuid4()), str(uuid4())
    expected = DeltaSet.from_gold(_gold(superseded_claim_ids=[old], new_active_claim_ids=[new]))
    same = DeltaSet.from_gold(_gold(superseded_claim_ids=[old], new_active_claim_ids=[new]))
    assert compare_deltas(expected, same).exact_match

    off = DeltaSet.from_gold(_gold(superseded_claim_ids=[wrong], new_active_claim_ids=[new]))
    cmp = compare_deltas(expected, off)
    assert not cmp.exact_match and cmp.mismatched_buckets == ["superseded_claim_ids"]
    diff = cmp.diffs[0]
    assert diff.missing == {old} and diff.unexpected == {wrong}


def test_review_claims_do_not_break_an_exact_match() -> None:
    cid = uuid4()
    delta = MemoryDelta(
        decision_id=uuid4(),
        applied=True,
        status_changes=[StatusChange(claim_id=cid, new_status=ClaimStatus.CANDIDATE)],
    )
    cmp = compare_deltas(DeltaSet.from_gold(_gold()), DeltaSet.from_memory_delta(delta))
    assert cmp.exact_match and cmp.review_claim_ids == {str(cid)}


def test_summary_rates_and_unused_buckets() -> None:
    x = str(uuid4())
    hit = compare_deltas(
        DeltaSet.from_gold(_gold(new_active_claim_ids=[x])),
        DeltaSet.from_gold(_gold(new_active_claim_ids=[x])),
    )
    miss = compare_deltas(
        DeltaSet.from_gold(_gold(new_active_claim_ids=[x])), DeltaSet.from_gold(_gold())
    )
    summary = summarize_comparisons([hit, miss])
    assert summary["cases"] == 2 and summary["exact_match"] == 1
    assert summary["exact_match_rate"] == 0.5
    per = summary["per_bucket"]
    assert per["new_active_claim_ids"] == {"precision": 1.0, "recall": 0.5}  # type: ignore[index]
    assert per["rejected_claim_ids"] == {"precision": None, "recall": None}  # type: ignore[index]
    assert summarize_comparisons([])["exact_match_rate"] is None
