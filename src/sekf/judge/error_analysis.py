"""Judge and evidence-citation error analysis (SPEC.md Sprint 3). Owner: Sai Chetan.

Deterministic. A case can carry more than one finding. This module scores the
Judge label, its confidence, and the evidence ids it cited. It does not attribute
extraction, retrieval, policy, or storage errors: those need other owners' outputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sekf.contracts import JudgeLabel
from sekf.judge.calibration import label_confidence_calibration
from sekf.judge.metrics import JudgeMetricsReport, evaluate_judge_labels

# Same default as SPEC.md section 9.3 support_threshold and refute_threshold.
# Analysis only: this module does not apply policy and does not write memory.
DEFAULT_CONFIDENCE_THRESHOLD = 0.80

_DIRECTIONAL = frozenset({JudgeLabel.SUPPORTED, JudgeLabel.REFUTED})
_REQUIRED_FIELDS = (
    "case_id",
    "category",
    "gold_label",
    "predicted_label",
    "confidence",
    "supporting_evidence_ids",
    "refuting_evidence_ids",
    "provided_evidence_ids",
)


class EvidenceFinding(StrEnum):
    """One Judge or evidence-citation problem on a single case."""

    UNKNOWN_EVIDENCE_ID = "UNKNOWN_EVIDENCE_ID"
    DUPLICATE_CITATION = "DUPLICATE_CITATION"
    CITED_AS_BOTH = "CITED_AS_BOTH"
    MISSING_SUPPORTING_CITATION = "MISSING_SUPPORTING_CITATION"
    MISSING_REFUTING_CITATION = "MISSING_REFUTING_CITATION"
    WRONG_LABEL = "WRONG_LABEL"
    OVERCONFIDENT_WRONG_LABEL = "OVERCONFIDENT_WRONG_LABEL"
    UNDERCONFIDENT_DIRECTIONAL_LABEL = "UNDERCONFIDENT_DIRECTIONAL_LABEL"


@dataclass(frozen=True)
class JudgeObservation:
    """One offline comparison of a gold Judge label with a prediction."""

    case_id: str
    category: str
    gold_label: JudgeLabel
    predicted_label: JudgeLabel
    confidence: float
    supporting_evidence_ids: tuple[str, ...] = ()
    refuting_evidence_ids: tuple[str, ...] = ()
    provided_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseFindings:
    """Findings for one case. An empty `findings` tuple is a clean case."""

    case_id: str
    category: str
    gold_label: str
    predicted_label: str
    findings: tuple[EvidenceFinding, ...]


@dataclass(frozen=True)
class JudgeEvidenceReport:
    """Counts, rates, label metrics, and calibration for a batch of cases."""

    n: int
    n_clean: int
    finding_counts: dict[str, int]
    finding_rates: dict[str, float]
    by_category: dict[str, dict[str, int]]
    by_gold_label: dict[str, dict[str, int]]
    label_metrics: JudgeMetricsReport
    brier: float
    ece: float
    cases: tuple[CaseFindings, ...]


def _check_threshold(confidence_threshold: float) -> None:
    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or confidence_threshold < 0.0
        or confidence_threshold > 1.0
    ):
        raise ValueError("confidence_threshold must be a number between 0 and 1")


def _as_id_tuple(value: object, *, field: str, prefix: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{prefix}{field} must be a list of strings")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{prefix}{field} must be a list of non-empty strings")
        ids.append(item)
    return tuple(ids)


def observation_from_mapping(
    payload: Mapping[str, Any], *, line_no: int | None = None
) -> JudgeObservation:
    """Build one observation from a JSON object. Missing fields are errors."""

    prefix = f"line {line_no}: " if line_no is not None else ""
    missing = [key for key in _REQUIRED_FIELDS if key not in payload]
    if missing:
        raise ValueError(prefix + "missing fields: " + ", ".join(missing))
    case_id = payload["case_id"]
    category = payload["category"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(prefix + "case_id must be a non-empty string")
    if not isinstance(category, str) or not category:
        raise ValueError(prefix + "category must be a non-empty string")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(prefix + "confidence must be a number between 0 and 1")
    try:
        gold = JudgeLabel(payload["gold_label"])
        predicted = JudgeLabel(payload["predicted_label"])
    except ValueError as exc:
        raise ValueError(prefix + "unknown judge label") from exc
    return JudgeObservation(
        case_id=case_id,
        category=category,
        gold_label=gold,
        predicted_label=predicted,
        confidence=float(confidence),
        supporting_evidence_ids=_as_id_tuple(
            payload["supporting_evidence_ids"], field="supporting_evidence_ids", prefix=prefix
        ),
        refuting_evidence_ids=_as_id_tuple(
            payload["refuting_evidence_ids"], field="refuting_evidence_ids", prefix=prefix
        ),
        provided_evidence_ids=_as_id_tuple(
            payload["provided_evidence_ids"], field="provided_evidence_ids", prefix=prefix
        ),
    )


def _needs_support(label: JudgeLabel) -> bool:
    return label in {JudgeLabel.SUPPORTED, JudgeLabel.MIXED_EVIDENCE}


def _needs_refutation(label: JudgeLabel) -> bool:
    return label in {JudgeLabel.REFUTED, JudgeLabel.MIXED_EVIDENCE}


def classify_judge_observation(
    observation: JudgeObservation, *, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> CaseFindings:
    """List Judge and citation findings for one case, in a stable order."""

    _check_threshold(confidence_threshold)
    supporting = observation.supporting_evidence_ids
    refuting = observation.refuting_evidence_ids
    provided = set(observation.provided_evidence_ids)
    findings: list[EvidenceFinding] = []
    cited = (*supporting, *refuting)
    if any(item not in provided for item in cited):
        findings.append(EvidenceFinding.UNKNOWN_EVIDENCE_ID)
    if len(supporting) != len(set(supporting)) or len(refuting) != len(set(refuting)):
        findings.append(EvidenceFinding.DUPLICATE_CITATION)
    if set(supporting) & set(refuting):
        findings.append(EvidenceFinding.CITED_AS_BOTH)
    if _needs_support(observation.predicted_label) and not supporting:
        findings.append(EvidenceFinding.MISSING_SUPPORTING_CITATION)
    if _needs_refutation(observation.predicted_label) and not refuting:
        findings.append(EvidenceFinding.MISSING_REFUTING_CITATION)
    wrong = observation.gold_label is not observation.predicted_label
    if wrong:
        findings.append(EvidenceFinding.WRONG_LABEL)
        if observation.confidence >= confidence_threshold:
            findings.append(EvidenceFinding.OVERCONFIDENT_WRONG_LABEL)
    elif (
        observation.predicted_label in _DIRECTIONAL
        and observation.confidence < confidence_threshold
    ):
        findings.append(EvidenceFinding.UNDERCONFIDENT_DIRECTIONAL_LABEL)
    return CaseFindings(
        case_id=observation.case_id,
        category=observation.category,
        gold_label=observation.gold_label.value,
        predicted_label=observation.predicted_label.value,
        findings=tuple(findings),
    )


def _bucket_counts(cases: Sequence[CaseFindings], key_fn: Any) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[CaseFindings]] = {}
    for case in cases:
        grouped.setdefault(str(key_fn(case)), []).append(case)
    report: dict[str, dict[str, int]] = {}
    for key in sorted(grouped):
        group = grouped[key]
        counts: dict[str, int] = {
            "n": len(group),
            "n_clean": sum(1 for case in group if not case.findings),
        }
        for finding in EvidenceFinding:
            counts[finding.value] = sum(1 for case in group if finding in case.findings)
        report[key] = counts
    return report


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def analyze_judge_evidence(
    observations: Sequence[JudgeObservation],
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    n_bins: int = 10,
) -> JudgeEvidenceReport:
    """Summarize Judge label errors, citation errors, and confidence calibration."""

    _check_threshold(confidence_threshold)
    cases = tuple(
        classify_judge_observation(item, confidence_threshold=confidence_threshold)
        for item in observations
    )
    counts = Counter(finding.value for case in cases for finding in case.findings)
    finding_counts = {
        finding.value: int(counts.get(finding.value, 0)) for finding in EvidenceFinding
    }
    gold = [item.gold_label for item in observations]
    predicted = [item.predicted_label for item in observations]
    confidences = [item.confidence for item in observations]
    calibration = label_confidence_calibration(gold, predicted, confidences, n_bins=n_bins)
    n = len(observations)
    n_clean = sum(1 for case in cases if not case.findings)
    return JudgeEvidenceReport(
        n=n,
        n_clean=n_clean,
        finding_counts=finding_counts,
        finding_rates={name: _rate(count, n) for name, count in finding_counts.items()},
        by_category=_bucket_counts(cases, lambda case: case.category),
        by_gold_label=_bucket_counts(cases, lambda case: case.gold_label),
        label_metrics=evaluate_judge_labels(gold, predicted),
        brier=calibration.brier,
        ece=calibration.ece,
        cases=cases,
    )


def report_to_dict(report: JudgeEvidenceReport) -> dict[str, Any]:
    """JSON-ready report. Rates stay beside the raw counts from `finding_counts`."""

    metrics = report.label_metrics
    return {
        "n": report.n,
        "n_clean": report.n_clean,
        "finding_counts": report.finding_counts,
        "finding_rates": report.finding_rates,
        "by_category": report.by_category,
        "by_gold_label": report.by_gold_label,
        "label_metrics": {
            "n": metrics.n,
            "accuracy": metrics.accuracy,
            "contradiction_f1": metrics.contradiction_f1,
            "contradiction": {
                "precision": metrics.contradiction.precision,
                "recall": metrics.contradiction.recall,
                "f1": metrics.contradiction.f1,
                "support": metrics.contradiction.support,
            },
            "per_label_support": metrics.per_label_support,
            "confusion": metrics.confusion,
        },
        "brier": report.brier,
        "ece": report.ece,
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category,
                "gold_label": case.gold_label,
                "predicted_label": case.predicted_label,
                "findings": [finding.value for finding in case.findings],
            }
            for case in report.cases
        ],
    }
