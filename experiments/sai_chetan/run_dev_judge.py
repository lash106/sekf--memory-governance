"""Score the dev split with the CI fake Judge and the lexical fixture Judge.

No network. The Judge request is the candidate and the evidence only.
Gold labels are read after the prediction, for scoring.

    python experiments/sai_chetan/run_dev_judge.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import (
    CandidateClaim,
    DecideMode,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeLabel,
    JudgeRequest,
    RunConfig,
)
from sekf.judge import FakeJudge, FixtureJudge
from sekf.judge.calibration import fit_temperature
from sekf.judge.error_analysis import (
    JudgeObservation,
    analyze_judge_evidence,
    report_to_dict,
)
from sekf.judge.fixture import PROMPT_VERSION
from sekf.memory.in_memory import InMemoryRepository

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "datasets" / "dev"
OUT = ROOT / "experiments" / "sai_chetan" / "dev_2026-11-20"
RUN = RunConfig(model="fake/none", temperature=0.0, seed=0)


def load_dev_cases() -> list[dict[str, Any]]:
    """Load labeled case files. Kept local so this script does not touch evaluation/."""

    cases: list[dict[str, Any]] = []
    for path in sorted(DEV.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "gold" in data and "candidate" in data:
            data["_path"] = str(path)
            cases.append(data)
    return cases


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _binary_f1(gold: list[str], predicted: list[str], positive: str) -> float:
    tp = fp = fn = 0
    for left, right in zip(gold, predicted, strict=True):
        if right == positive and left == positive:
            tp += 1
        elif right == positive:
            fp += 1
        elif left == positive:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    if not labels:
        return 0.0
    return sum(_binary_f1(gold, predicted, label) for label in labels) / len(labels)


def promotion_scores(gold: list[str], predicted: list[str]) -> tuple[float, float]:
    positive = FinalAction.PROMOTE.value
    tp = fp = fn = 0
    for left, right in zip(gold, predicted, strict=True):
        if right == positive and left == positive:
            tp += 1
        elif right == positive:
            fp += 1
        elif left == positive:
            fn += 1
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def revision_accuracy(gold: list[str], predicted: list[str]) -> float | None:
    positive = FinalAction.REVISE.value
    pairs = [(left, right) for left, right in zip(gold, predicted, strict=True) if left == positive]
    if not pairs:
        return None
    return sum(left == right for left, right in pairs) / len(pairs)


def _judge_request(case: dict[str, Any]) -> JudgeRequest:
    """Build the Judge input. Gold, memory, and the final action stay out."""

    return JudgeRequest(
        candidate=CandidateClaim.model_validate(case["candidate"]),
        evidence=[EvidenceRecord.model_validate(item) for item in case.get("evidence", [])],
        run_config=RUN,
    )


def _observation(
    case: dict[str, Any],
    label: JudgeLabel,
    confidence: float,
    supporting: list[Any],
    refuting: list[Any],
) -> JudgeObservation:
    evidence = case.get("evidence") or []
    return JudgeObservation(
        case_id=str(case["case_id"]),
        category=str(case["category"]),
        gold_label=JudgeLabel(case["gold"]["judge_label"]),
        predicted_label=label,
        confidence=confidence,
        supporting_evidence_ids=tuple(str(item) for item in supporting),
        refuting_evidence_ids=tuple(str(item) for item in refuting),
        provided_evidence_ids=tuple(str(item["evidence_id"]) for item in evidence),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _action_summary(gold: list[str], predicted: list[str]) -> dict[str, Any]:
    precision, recall = promotion_scores(gold, predicted)
    return {
        "n": len(gold),
        "macro_f1": macro_f1(gold, predicted),
        "promotion_precision": precision,
        "promotion_recall": recall,
        "revision_accuracy": revision_accuracy(gold, predicted),
    }


async def _score_judge(judge: Any, cases: list[dict[str, Any]]) -> list[JudgeObservation]:
    observations: list[JudgeObservation] = []
    for case in cases:
        assessment = await judge.evaluate(_judge_request(case))
        observations.append(
            _observation(
                case,
                assessment.label,
                assessment.confidence,
                assessment.supporting_evidence_ids,
                assessment.refuting_evidence_ids,
            )
        )
    return observations


async def _ensemble_actions(
    judge: Any, cases: list[dict[str, Any]]
) -> tuple[list[str], list[str], int]:
    gold: list[str] = []
    predicted: list[str] = []
    identical_panels = 0
    for case in cases:
        memory = InMemoryRepository(
            [ExistingClaim.model_validate(item) for item in case.get("initial_memory", [])]
        )
        request = DecideRequest(
            candidate=CandidateClaim.model_validate(case["candidate"]),
            evidence=[EvidenceRecord.model_validate(item) for item in case.get("evidence", [])],
            mode=DecideMode.DRY_RUN,
            run_config=RUN,
        )
        baseline = GenericEnsembleBaseline(judge, memory=memory)
        response = await baseline.decide(request)
        labels = []
        for offset in (0, 1, 2):
            member = await judge.evaluate(
                JudgeRequest(
                    candidate=request.candidate,
                    evidence=request.evidence,
                    run_config=RUN.model_copy(update={"seed": offset}),
                )
            )
            labels.append(member.label)
        if len(set(labels)) == 1:
            identical_panels += 1
        gold.append(str(case["gold"]["final_action"]))
        predicted.append(response.decision.final_action.value)
    return gold, predicted, identical_panels


def _rows(observations: list[JudgeObservation]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item.case_id,
            "category": item.category,
            "gold_label": item.gold_label.value,
            "predicted_label": item.predicted_label.value,
            "confidence": item.confidence,
            "supporting_evidence_ids": list(item.supporting_evidence_ids),
            "refuting_evidence_ids": list(item.refuting_evidence_ids),
            "provided_evidence_ids": list(item.provided_evidence_ids),
        }
        for item in observations
    ]


async def main() -> None:
    cases = load_dev_cases()
    fake = await _score_judge(FakeJudge(), cases)
    fixture = await _score_judge(FixtureJudge(), cases)
    fake_gold, fake_actions, fake_identical = await _ensemble_actions(FakeJudge(), cases)
    fixture_gold, fixture_actions, fixture_identical = await _ensemble_actions(
        FixtureJudge(), cases
    )
    correct = [item.gold_label is item.predicted_label for item in fixture]
    temperature = fit_temperature([item.confidence for item in fixture], correct)
    summary = {
        "owner": "Sai Chetan Muppalla",
        "split": "dev",
        "held_out_test": False,
        "n_cases": len(cases),
        "run_mode": {
            "SEKF_MEMORY": "memory",
            "SEKF_PROVIDER": "fake",
            "SEKF_HISTORIAN": "not invoked",
            "SEKF_SYNTHESIZER": "not invoked",
            "temperature": 0.0,
            "seed": 0,
            "policy": "not applied",
            "single_judge": "not run; no live key in this process",
            "fixture_prompt_version": PROMPT_VERSION,
            "fixture_prompt_hash_meaning": "sha256 of the version string, not of a prompt file",
            "cost_usd": 0.0,
            "cost_note": "Neither row calls a priced model. gpt-4o-mini rates are not reused.",
        },
        "fake_judge": report_to_dict(analyze_judge_evidence(fake)),
        "fixture_judge": report_to_dict(analyze_judge_evidence(fixture)),
        "generic_ensemble_fake": {
            **_action_summary(fake_gold, fake_actions),
            "identical_member_panels": fake_identical,
            "note": "Three FakeJudge votes. SUPPORTED maps to PROMOTE. REFUTED is never emitted.",
        },
        "generic_ensemble_fixture": {
            **_action_summary(fixture_gold, fixture_actions),
            "identical_member_panels": fixture_identical,
            "note": (
                "The fixture ignores seed, so three members match. "
                "That is the temperature-0 degeneration in SPEC.md section 12.3, recorded here."
            ),
        },
        "descriptive_temperature_fit_on_these_same_rows": temperature,
        "temperature_fit_note": (
            "Fit on these same dev rows. Not a selected threshold and not a test score."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUT / "fake_observations.jsonl", _rows(fake))
    _write_jsonl(OUT / "fixture_observations.jsonl", _rows(fixture))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"cases {len(cases)}")
    print(f"fake accuracy {summary['fake_judge']['label_metrics']['accuracy']:.4f}")
    print(f"fake contradiction_f1 {summary['fake_judge']['label_metrics']['contradiction_f1']:.4f}")
    print(f"fixture accuracy {summary['fixture_judge']['label_metrics']['accuracy']:.4f}")
    print(
        "fixture contradiction_f1 "
        f"{summary['fixture_judge']['label_metrics']['contradiction_f1']:.4f}"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
