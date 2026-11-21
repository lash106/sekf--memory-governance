"""Cohen's kappa, percent agreement, and the annotator comparison."""

import json
from pathlib import Path
from typing import Any

import pytest

import check_agreement
from sekf.evaluation.agreement import cohens_kappa, compare, load_cases, percent_agreement


def test_kappa_known_value() -> None:
    # observed 0.75, chance 0.5 -> (0.75 - 0.5) / (1 - 0.5) = 0.5
    a, b = ["y", "y", "n", "n"], ["y", "n", "n", "n"]
    assert percent_agreement(a, b) == 0.75
    assert cohens_kappa(a, b) == pytest.approx(0.5)


def test_kappa_extremes() -> None:
    labels = ["a", "b", "c", "a", "b", "c"]
    assert cohens_kappa(labels, labels) == 1.0
    assert cohens_kappa(["a", "b"] * 3, ["b", "a"] * 3) == pytest.approx(-1.0)  # always opposite
    assert cohens_kappa(["x"] * 4, ["x"] * 4) == 1.0  # chance agreement is 1: defined as 1.0
    assert cohens_kappa(["x"] * 4, ["y"] * 4) == pytest.approx(0.0)  # constant but different


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="no labels"):
        cohens_kappa([], [])


def case(case_id: str, **gold: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "candidate": {"claim_id": case_id, "claim_text": "t"},
        "initial_memory": [],
        "evidence": [],
        "gold": {
            "judge_label": "SUPPORTED",
            "temporal_relation": "NO_RELATED_CLAIM",
            "conflict_type": "NONE",
            "final_action": "PROMOTE",
        }
        | gold,
    }


def write(folder: Path, cases: list[dict[str, Any]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for c in cases:
        (folder / f"{c['case_id']}.json").write_text(json.dumps(c))
    return folder


def test_identical_labels_agree_perfectly(tmp_path: Path) -> None:
    cases = [case("c1"), case("c2", final_action="REJECT", judge_label="REFUTED")]
    report = compare(
        load_cases(write(tmp_path / "a", cases)), load_cases(write(tmp_path / "b", cases))
    )
    assert report.compared == 2 and report.worst_kappa() == 1.0
    assert report.problems(0.6) == []


def test_disagreements_are_listed_per_field(tmp_path: Path) -> None:
    names = ["c1", "c2", "c3", "c4"]
    a = [case(n) for n in names]
    b = [case(n) for n in names]
    b[1]["gold"]["final_action"] = "KEEP_EPISODIC"
    report = compare(load_cases(write(tmp_path / "a", a)), load_cases(write(tmp_path / "b", b)))
    action = next(f for f in report.fields if f.name == "final_action")
    assert action.percent == 0.75 and action.disagreements == [("c2", "PROMOTE", "KEEP_EPISODIC")]
    assert next(f for f in report.fields if f.name == "judge_label").kappa == 1.0


def test_missing_cases_and_changed_inputs_block_freezing(tmp_path: Path) -> None:
    a = [case("c1"), case("c2"), case("only-a")]
    b = [case("c1"), case("c2"), case("only-b")]
    b[1]["evidence"] = [{"evidence_id": "e", "content": "B saw different evidence"}]
    report = compare(load_cases(write(tmp_path / "a", a)), load_cases(write(tmp_path / "b", b)))
    assert report.compared == 1  # c2 is not scored: the annotators saw different inputs
    problems = "\n".join(report.problems(0.6))
    assert "only labeled by A: only-a" in problems and "only labeled by B: only-b" in problems
    assert "inputs differ between A and B: c2" in problems


def test_duplicate_case_ids_in_one_folder_are_an_error(tmp_path: Path) -> None:
    folder = write(tmp_path / "a", [case("c1")])
    (folder / "copy.json").write_text(json.dumps(case("c1")))
    with pytest.raises(ValueError, match="duplicate case_id c1"):
        load_cases(folder)


def test_cli_exit_codes_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    names = [f"c{i}" for i in range(6)]
    a = write(tmp_path / "a", [case(n) for n in names])
    same = write(tmp_path / "b1", [case(n) for n in names])
    assert check_agreement.main([str(a), str(same)]) == 0
    assert "final_action" in capsys.readouterr().out

    flipped = [case(n, final_action="REJECT" if i % 2 else "PROMOTE") for i, n in enumerate(names)]
    differ = write(tmp_path / "b2", flipped)
    assert check_agreement.main([str(a), str(differ)]) == 1
    out = capsys.readouterr().out
    assert "disagree" in out and "PROBLEM: final_action: kappa" in out
    assert check_agreement.main([str(a), str(differ), "--min-kappa", "-1"]) == 0  # threshold waived


def test_cli_fails_when_nothing_is_comparable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_agreement.main([str(empty), str(empty)]) == 1
