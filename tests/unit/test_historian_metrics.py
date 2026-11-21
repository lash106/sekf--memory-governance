"""Unit tests for the Historian metrics (no LLM, no contracts)."""

from typing import Any

import pytest

from sekf.evaluation.historian_metrics import (
    Score,
    accuracy_by_category,
    conflict_type_accuracy,
    confusion_counts,
    joint_historian_accuracy,
    label_accuracy,
    provenance_checks,
    provenance_completeness,
    temporal_relation_accuracy,
)

OLD_CLAIM = "11111111-1111-4111-8111-111111111111"
HIST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
EVIDENCE = "33333333-3333-4333-8333-333333333333"

CANDIDATE = {"source_id": "official-schedule", "observed_at": "2026-10-01T20:00:00Z"}


def response(action: str, **overrides: object) -> dict:
    """A fully traceable DecideResponse as JSON; overrides break one part of it."""

    decision = {
        "final_action": action,
        "historian_assessment_id": HIST_ID,
        "previous_claim_ids": [OLD_CLAIM] if action == "REVISE" else [],
        "rule_ids_fired": ["R1"],
    }
    historian = {
        "assessment_id": HIST_ID,
        "provenance_quality": "HIGH",
        "supersedes_claim_id": OLD_CLAIM if action == "REVISE" else None,
    }
    judge = {"supporting_evidence_ids": [EVIDENCE]}
    data: dict[str, Any] = {"decision": decision, "historian": historian, "judge": judge}
    for key, value in overrides.items():
        part, _, name = key.partition("__")
        if name:
            data[part][name] = value
        else:
            data[part] = value
    return data


# --- Score ---------------------------------------------------------------------------------


def test_score_value_and_empty() -> None:
    assert Score(3, 4).value == 0.75
    assert Score(0, 0).value is None


# --- label accuracy ------------------------------------------------------------------------


def test_temporal_relation_accuracy_counts_missing_prediction_as_wrong() -> None:
    gold = ["NEWER_UPDATE", "SAME_PERIOD", "NON_OVERLAPPING_VALIDITY"]
    predicted = ["NEWER_UPDATE", None, "OVERLAPPING_VALIDITY"]
    score = temporal_relation_accuracy(gold, predicted)
    assert (score.correct, score.n) == (1, 3)


def test_label_accuracy_skips_cases_without_gold() -> None:
    score = label_accuracy(["NONE", None, ""], ["NONE", "DUPLICATE", "DUPLICATE"])
    assert (score.correct, score.n) == (1, 1)


def test_label_accuracy_ignores_case_and_spaces() -> None:
    assert conflict_type_accuracy(["TEMPORAL_UPDATE"], [" temporal_update "]).value == 1.0


def test_label_accuracy_accepts_enum_like_values() -> None:
    class FakeEnum:
        value = "NEWER_UPDATE"

    assert label_accuracy(["NEWER_UPDATE"], [FakeEnum()]).value == 1.0


def test_label_accuracy_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        label_accuracy(["NONE"], ["NONE", "NONE"])


def test_joint_accuracy_needs_both_fields_right() -> None:
    score = joint_historian_accuracy(
        ["NEWER_UPDATE", "NEWER_UPDATE", "SAME_PERIOD"],
        ["NEWER_UPDATE", "NEWER_UPDATE", "SAME_PERIOD"],
        ["TEMPORAL_UPDATE", "TEMPORAL_UPDATE", None],
        ["TEMPORAL_UPDATE", "VALUE_CONFLICT", "NONE"],
    )
    assert (score.correct, score.n) == (1, 2)


# --- error analysis helpers ----------------------------------------------------------------


def test_confusion_counts_only_lists_errors() -> None:
    counts = confusion_counts(
        ["NEWER_UPDATE", "NEWER_UPDATE", "SAME_PERIOD", "SAME_PERIOD"],
        ["NEWER_UPDATE", "SAME_PERIOD", "SAME_PERIOD", None],
    )
    assert counts == {("NEWER_UPDATE", "SAME_PERIOD"): 1, ("SAME_PERIOD", "MISSING"): 1}


def test_accuracy_by_category() -> None:
    result = accuracy_by_category(
        ["reliable_newer_correction", "duplicate_or_paraphrase", "reliable_newer_correction"],
        ["NEWER_UPDATE", "SAME_PERIOD", "NEWER_UPDATE"],
        ["NEWER_UPDATE", "SAME_PERIOD", "OLDER_INFORMATION"],
    )
    assert list(result) == ["duplicate_or_paraphrase", "reliable_newer_correction"]
    assert result["reliable_newer_correction"] == Score(1, 2)


# --- provenance ----------------------------------------------------------------------------


def test_provenance_checks_complete_revision() -> None:
    checks = provenance_checks(response("REVISE"), CANDIDATE)
    assert all(checks.values())


def test_history_link_does_not_apply_to_new_claims() -> None:
    assert provenance_checks(response("PROMOTE"), CANDIDATE)["history_linked"] is None


@pytest.mark.parametrize(
    ("overrides", "candidate", "failed"),
    [
        ({}, {"source_id": " ", "observed_at": "2026-10-01T20:00:00Z"}, "source_identified"),
        ({}, {"source_id": "x"}, "time_recorded"),
        ({"decision__historian_assessment_id": OLD_CLAIM}, CANDIDATE, "historian_linked"),
        ({"historian__provenance_quality": "UNKNOWN"}, CANDIDATE, "provenance_assessed"),
        ({"judge": None}, CANDIDATE, "evidence_cited"),
        ({"historian__supersedes_claim_id": None}, CANDIDATE, "history_linked"),
        ({"decision__previous_claim_ids": []}, CANDIDATE, "history_linked"),
        ({"decision__rule_ids_fired": []}, CANDIDATE, "rules_recorded"),
    ],
)
def test_provenance_checks_catch_each_gap(overrides: dict, candidate: dict, failed: str) -> None:
    checks = provenance_checks(response("REVISE", **overrides), candidate)
    assert checks[failed] is False
    assert all(value for name, value in checks.items() if name != failed)


def test_missing_historian_fails_every_historian_check() -> None:
    checks = provenance_checks(response("REVISE", historian=None), CANDIDATE)
    assert checks["historian_linked"] is False
    assert checks["provenance_assessed"] is False
    assert checks["history_linked"] is False
    assert checks["source_identified"] and checks["evidence_cited"]


def test_provenance_completeness_scores_only_knowledge_actions() -> None:
    items = [
        (response("PROMOTE"), CANDIDATE),
        (response("REVISE", historian__provenance_quality="UNKNOWN"), CANDIDATE),
        (response("QUARANTINE"), CANDIDATE),
        (response("REJECT"), CANDIDATE),
    ]
    summary = provenance_completeness(items)
    assert summary.fully_complete == Score(1, 2)
    assert summary.excluded == 2
    assert summary.per_check["provenance_assessed"] == Score(1, 2)
    assert summary.per_check["history_linked"] == Score(1, 1)


def test_provenance_completeness_empty() -> None:
    summary = provenance_completeness([])
    assert summary.fully_complete.value is None
    assert summary.excluded == 0
