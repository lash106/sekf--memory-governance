"""Pure temporal and provenance helpers for the Historian (SPEC.md section 18, Sprint 0).

Owner: Mahalakshmi.

Plain Python, no LLM, no contract imports, so these can be unit-tested in isolation and
reused by the deterministic part of Historian v1.

Conventions:
- Validity intervals are half-open ``[start, end)``. ``None`` means unbounded on that side
  (``start=None`` is "since forever", ``end=None`` is "still valid").
- All datetimes must be timezone-aware (contracts use UTC RFC 3339); naive values raise
  ``ValueError`` instead of being silently guessed.
"""

from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import StrEnum


class IntervalRelation(StrEnum):
    """How two validity intervals relate."""

    OVERLAPPING = "OVERLAPPING"
    NON_OVERLAPPING = "NON_OVERLAPPING"


class TemporalOrder(StrEnum):
    """Order of a candidate observation relative to an existing claim."""

    NEWER = "NEWER"
    OLDER = "OLDER"
    SAME = "SAME"


class ProvenanceBand(StrEnum):
    """Provenance quality bands. Names match the contract's ``ProvenanceQuality`` values."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


DEFAULT_HIGH_THRESHOLD = 0.80
DEFAULT_MEDIUM_THRESHOLD = 0.60  # matches source_reliability_threshold (SPEC.md 9.3)


def _require_aware(value: datetime | None, name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{name} must be timezone-aware, got naive datetime {value!r}")


def _check_interval(start: datetime | None, end: datetime | None, label: str) -> None:
    _require_aware(start, f"{label} start")
    _require_aware(end, f"{label} end")
    if start is not None and end is not None and end <= start:
        raise ValueError(f"{label} interval is empty or reversed: start={start}, end={end}")


def intervals_overlap(
    a_start: datetime | None,
    a_end: datetime | None,
    b_start: datetime | None,
    b_end: datetime | None,
) -> bool:
    """Return True if half-open intervals ``[a_start, a_end)`` and ``[b_start, b_end)`` overlap.

    Touching intervals do not overlap: ``[Jan 1, Oct 1)`` and ``[Oct 1, None)`` -> False.

    Example:
        >>> from datetime import UTC, datetime
        >>> jan, oct_ = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)
        >>> intervals_overlap(jan, None, oct_, None)
        True
        >>> intervals_overlap(jan, oct_, oct_, None)
        False
    """
    _check_interval(a_start, a_end, "a")
    _check_interval(b_start, b_end, "b")
    a_starts_before_b_ends = b_end is None or a_start is None or a_start < b_end
    b_starts_before_a_ends = a_end is None or b_start is None or b_start < a_end
    return a_starts_before_b_ends and b_starts_before_a_ends


def classify_validity(
    candidate_start: datetime | None,
    candidate_end: datetime | None,
    existing_start: datetime | None,
    existing_end: datetime | None,
) -> IntervalRelation | None:
    """Classify the validity relation between a candidate and an existing claim.

    Returns ``None`` when neither side states any validity bound: there is nothing to compare,
    and the caller should fall back to observation-time ordering instead of guessing overlap.
    """
    if all(v is None for v in (candidate_start, candidate_end, existing_start, existing_end)):
        return None
    if intervals_overlap(candidate_start, candidate_end, existing_start, existing_end):
        return IntervalRelation.OVERLAPPING
    return IntervalRelation.NON_OVERLAPPING


def temporal_order(
    candidate_time: datetime,
    existing_time: datetime,
    tolerance: timedelta = timedelta(0),
) -> TemporalOrder:
    """Order a candidate observation against an existing claim's timestamp.

    Differences within ``tolerance`` (inclusive) count as ``SAME``, which avoids treating two
    reports seconds apart as an update.
    """
    _require_aware(candidate_time, "candidate_time")
    _require_aware(existing_time, "existing_time")
    if tolerance < timedelta(0):
        raise ValueError("tolerance must be non-negative")
    delta = candidate_time - existing_time
    if abs(delta) <= tolerance:
        return TemporalOrder.SAME
    return TemporalOrder.NEWER if delta > timedelta(0) else TemporalOrder.OLDER


def provenance_score(reliabilities: Iterable[float]) -> float | None:
    """Aggregate per-source reliabilities (0-1) into one provenance score.

    v0 rule: the best available source sets the score (max). Returns ``None`` when there is
    no evidence, so "no provenance" is never confused with "low provenance".
    """
    values = list(reliabilities)
    for r in values:
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"reliability must be within [0, 1], got {r}")
    return max(values) if values else None


def provenance_band(
    score: float | None,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
) -> ProvenanceBand:
    """Map a provenance score to a quality band (``None`` -> ``UNKNOWN``).

    Example:
        >>> provenance_band(0.95), provenance_band(0.7), provenance_band(0.3), provenance_band(None)
        (<ProvenanceBand.HIGH: 'HIGH'>, <ProvenanceBand.MEDIUM: 'MEDIUM'>, \
<ProvenanceBand.LOW: 'LOW'>, <ProvenanceBand.UNKNOWN: 'UNKNOWN'>)
    """
    if not 0.0 <= medium_threshold <= high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= medium <= high <= 1")
    if score is None:
        return ProvenanceBand.UNKNOWN
    if score >= high_threshold:
        return ProvenanceBand.HIGH
    if score >= medium_threshold:
        return ProvenanceBand.MEDIUM
    return ProvenanceBand.LOW
