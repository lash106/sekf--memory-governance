"""Sprint 3: Judge/evidence findings, calibration, and the offline JSONL scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_judge_evidence import main
from sekf.contracts import JudgeLabel
from sekf.judge.error_analysis import (
    EvidenceFinding,
    JudgeObservation,
    analyze_judge_evidence,
    observation_from_mapping,
)
from sekf.judge.freeze import assert_prompts_frozen


def _obs(**overrides: object) -> JudgeObservation:
    payload: dict[str, object] = {
        "case_id": "case-1",
        "category": "supported_new_claim",
        "gold_label": JudgeLabel.SUPPORTED,
        "predicted_label": JudgeLabel.SUPPORTED,
        "confidence": 0.91,
        "supporting_evidence_ids": ("e1",),
        "refuting_evidence_ids": (),
        "provided_evidence_ids": ("e1",),
    }
    payload.update(overrides)
    return JudgeObservation(**payload)  # type: ignore[arg-type]


def test_clean_supported_case_has_no_findings() -> None:
    report = analyze_judge_evidence([_obs()])
    assert report.n == 1
    assert report.n_clean == 1
    assert report.cases[0].findings == ()
    assert report.label_metrics.accuracy == 1.0
    assert report.brier == pytest.approx((0.91 - 1.0) ** 2)


def test_wrong_high_confidence_refutation_is_a_contradiction_miss() -> None:
    report = analyze_judge_evidence(
        [
            _obs(
                case_id="case-2",
                category="refuted_claim",
                gold_label=JudgeLabel.REFUTED,
                predicted_label=JudgeLabel.SUPPORTED,
                confidence=0.95,
                supporting_evidence_ids=("e1",),
                refuting_evidence_ids=(),
            )
        ]
    )
    assert report.cases[0].findings == (
        EvidenceFinding.WRONG_LABEL,
        EvidenceFinding.OVERCONFIDENT_WRONG_LABEL,
    )
    assert report.finding_counts["WRONG_LABEL"] == 1
    assert report.label_metrics.contradiction_f1 == 0.0
    assert report.by_category["refuted_claim"]["n"] == 1
    assert report.by_gold_label["REFUTED"]["OVERCONFIDENT_WRONG_LABEL"] == 1
    assert report.brier == pytest.approx((0.95 - 0.0) ** 2)


def test_citation_problems_are_separate_from_a_correct_label() -> None:
    report = analyze_judge_evidence(
        [
            _obs(
                predicted_label=JudgeLabel.MIXED_EVIDENCE,
                gold_label=JudgeLabel.MIXED_EVIDENCE,
                confidence=0.4,
                supporting_evidence_ids=("e1", "e1", "missing"),
                refuting_evidence_ids=("e1",),
            )
        ]
    )
    assert report.cases[0].findings == (
        EvidenceFinding.UNKNOWN_EVIDENCE_ID,
        EvidenceFinding.DUPLICATE_CITATION,
        EvidenceFinding.CITED_AS_BOTH,
    )
    low = analyze_judge_evidence([_obs(confidence=0.5)])
    assert low.cases[0].findings == (EvidenceFinding.UNDERCONFIDENT_DIRECTIONAL_LABEL,)
    empty = analyze_judge_evidence(
        [
            _obs(
                predicted_label=JudgeLabel.REFUTED,
                gold_label=JudgeLabel.REFUTED,
                supporting_evidence_ids=(),
                refuting_evidence_ids=(),
            )
        ]
    )
    assert EvidenceFinding.MISSING_REFUTING_CITATION in empty.cases[0].findings
    assert EvidenceFinding.WRONG_LABEL not in empty.cases[0].findings


def test_observation_mapping_and_script_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = {
        "case_id": "supported-new-01",
        "category": "supported_new_claim",
        "gold_label": "SUPPORTED",
        "predicted_label": "INSUFFICIENT_EVIDENCE",
        "confidence": 0.2,
        "supporting_evidence_ids": [],
        "refuting_evidence_ids": [],
        "provided_evidence_ids": ["e1"],
    }
    observation = observation_from_mapping(row, line_no=3)
    assert observation.gold_label is JudgeLabel.SUPPORTED
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main([str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n"] == 1
    assert printed["finding_counts"]["WRONG_LABEL"] == 1
    assert printed["label_metrics"]["accuracy"] == 0.0
    with pytest.raises(ValueError, match="missing fields"):
        observation_from_mapping({"case_id": "x"})
    assert main([str(tmp_path / "missing.jsonl")]) == 2


def test_empty_batch_and_bad_threshold() -> None:
    report = analyze_judge_evidence([])
    assert report.n == 0
    assert report.n_clean == 0
    assert report.brier == 0.0
    assert report.ece == 0.0
    with pytest.raises(ValueError, match="confidence_threshold"):
        analyze_judge_evidence([_obs()], confidence_threshold=1.5)


def test_part1_judge_prompts_remain_frozen() -> None:
    assert_prompts_frozen()
