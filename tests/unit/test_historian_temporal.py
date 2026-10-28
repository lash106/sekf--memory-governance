"""Unit tests for the Historian's pure temporal/provenance helpers (no LLM, no contracts)."""

from datetime import UTC, datetime, timedelta

import pytest

from sekf.historian.temporal import (
    IntervalRelation,
    ProvenanceBand,
    TemporalOrder,
    classify_validity,
    intervals_overlap,
    provenance_band,
    provenance_score,
    temporal_order,
)


def d(month: int, day: int = 1) -> datetime:
    return datetime(2026, month, day, tzinfo=UTC)


# --- intervals_overlap -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((d(1), d(6)), (d(3), d(9)), True),  # partial overlap
        ((d(1), d(12)), (d(3), d(4)), True),  # containment
        ((d(1), d(3)), (d(6), d(9)), False),  # disjoint
        ((d(1), d(10)), (d(10), None), False),  # touching, half-open
        ((d(1), None), (d(10), None), True),  # both open-ended (SPEC worked example)
        ((None, None), (d(5), d(6)), True),  # unbounded covers everything
        ((None, d(3)), (d(3), None), False),  # unbounded sides that only touch
    ],
)
def test_intervals_overlap(
    a: tuple[datetime | None, datetime | None],
    b: tuple[datetime | None, datetime | None],
    expected: bool,
) -> None:
    assert intervals_overlap(*a, *b) is expected
    assert intervals_overlap(*b, *a) is expected  # symmetric


def test_reversed_interval_rejected() -> None:
    with pytest.raises(ValueError, match="empty or reversed"):
        intervals_overlap(d(5), d(1), None, None)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        intervals_overlap(datetime(2026, 1, 1), None, None, None)


# --- classify_validity -------------------------------------------------------------------


def test_classify_validity_no_bounds_returns_none() -> None:
    assert classify_validity(None, None, None, None) is None


def test_classify_validity_lab_example_overlaps_until_closed() -> None:
    # Old "17:00" claim valid from Jan 1 with no end; new "18:00" valid from Oct 1.
    # Until the old claim is closed, the intervals overlap.
    assert classify_validity(d(10), None, d(1), None) is IntervalRelation.OVERLAPPING
    # Once the old claim is closed at Oct 1 they no longer overlap.
    assert classify_validity(d(10), None, d(1), d(10)) is IntervalRelation.NON_OVERLAPPING


# --- temporal_order ----------------------------------------------------------------------


def test_temporal_order() -> None:
    assert temporal_order(d(10), d(1)) is TemporalOrder.NEWER
    assert temporal_order(d(1), d(10)) is TemporalOrder.OLDER
    assert temporal_order(d(1), d(1)) is TemporalOrder.SAME


def test_temporal_order_tolerance() -> None:
    t = d(10)
    assert temporal_order(t + timedelta(seconds=30), t, timedelta(minutes=1)) is TemporalOrder.SAME
    assert temporal_order(t + timedelta(minutes=2), t, timedelta(minutes=1)) is TemporalOrder.NEWER


def test_temporal_order_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_order(d(1), d(1), timedelta(seconds=-1))


# --- provenance --------------------------------------------------------------------------


def test_provenance_score() -> None:
    assert provenance_score([0.3, 0.95, 0.5]) == 0.95
    assert provenance_score([]) is None


def test_provenance_score_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        provenance_score([1.2])


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.95, ProvenanceBand.HIGH),
        (0.80, ProvenanceBand.HIGH),
        (0.79, ProvenanceBand.MEDIUM),
        (0.60, ProvenanceBand.MEDIUM),
        (0.30, ProvenanceBand.LOW),
        (None, ProvenanceBand.UNKNOWN),
    ],
)
def test_provenance_band(score: float | None, band: ProvenanceBand) -> None:
    assert provenance_band(score) is band


def test_provenance_band_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        provenance_band(0.5, high_threshold=0.5, medium_threshold=0.9)
