"""Leave-one-author-out selection for policy thresholds and Judge temperature.

Thresholds and the temperature are fit on every author except the one being scored.
The held-out scores are the ones to report. This module does not call an LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sekf.contracts import CommitBundle, PolicyContext, Thresholds
from sekf.evaluation.cached import CachedCase
from sekf.evaluation.metrics import (
    ScoredCase,
    efficiency_report,
    macro_f1,
    paired_bootstrap_interval,
    unnecessary_quarantine_rate,
)
from sekf.judge.calibration import fit_temperature
from sekf.memory.markdown_projection import project_bundle
from sekf.policy import PolicyEngine, PolicyEngineV2

# Use this wording in slides and the paper until datasets/test/ is frozen.
DEV_CV_LABEL = "leave-one-author-out dev CV"


def default_grid() -> list[Thresholds]:
    """Small policy-v1 grid. policy-v2 replaces support and refute from the risk tier."""

    grid: list[Thresholds] = []
    for support in (0.6, 0.8, 0.92):
        for veto in (0.6, 0.8):
            grid.append(
                Thresholds(
                    support_threshold=support,
                    refute_threshold=support,
                    synthesizer_veto_threshold=veto,
                )
            )
    return grid


def _predict(engine: PolicyEngine, context: PolicyContext, thresholds: Thresholds) -> str:
    updated = context.model_copy(update={"thresholds": thresholds})
    return engine.decide(updated).final_action.value


def _choose_thresholds(train: list[CachedCase], grid: list[Thresholds]) -> Thresholds:
    engine = PolicyEngine()
    best = grid[0]
    best_key: tuple[float, float] | None = None
    for thresholds in grid:
        gold = [case.gold_action for case in train]
        predicted = [_predict(engine, case.context, thresholds) for case in train]
        key = (macro_f1(gold, predicted), -unnecessary_quarantine_rate(gold, predicted))
        if best_key is None or key > best_key:
            best_key = key
            best = thresholds
    return best


def _fit_temperature(train: list[CachedCase]) -> float:
    confidences: list[float] = []
    correct: list[bool] = []
    for case in train:
        judge = case.context.judge
        if judge is None or not case.gold_judge:
            continue
        confidences.append(judge.confidence)
        correct.append(judge.label.value == case.gold_judge)
    if not confidences:
        return 1.0
    return fit_temperature(confidences, correct)


def _macro(cases: list[CachedCase], predicted: list[str]) -> float:
    return macro_f1([case.gold_action for case in cases], predicted)


def leave_one_author_out(cases: list[CachedCase], grid: list[Thresholds]) -> dict[str, Any]:
    """Select on the other authors, then score the held-out author.

    Raises ValueError when fewer than two author groups are present. A single-split
    best score is not returned in its place.
    """

    if not grid:
        raise ValueError("threshold grid is empty")
    authors = sorted({case.author for case in cases})
    if len(authors) < 2:
        raise ValueError(
            "leave-one-author-out needs at least two author groups; "
            "a best score on the same split is not reported"
        )
    folds: list[dict[str, Any]] = []
    for held in authors:
        train = [case for case in cases if case.author != held]
        test = [case for case in cases if case.author == held]
        chosen = _choose_thresholds(train, grid)
        temperature = _fit_temperature(train)
        v1_predicted = [_predict(PolicyEngine(), case.context, chosen) for case in test]
        v2 = PolicyEngineV2(judge_temperature=temperature)
        v2_predicted = [v2.decide(case.context).final_action.value for case in test]
        folds.append(
            {
                "held_out_author": held,
                "n_train": len(train),
                "n_held_out": len(test),
                "chosen_thresholds": chosen.model_dump(mode="json"),
                "judge_temperature": temperature,
                "held_out_policy_v1_macro_f1": _macro(test, v1_predicted),
                "held_out_policy_v2_macro_f1": _macro(test, v2_predicted),
                "held_out_policy_v1_unnecessary_quarantine_rate": unnecessary_quarantine_rate(
                    [case.gold_action for case in test], v1_predicted
                ),
            }
        )
    v1_scores = [float(fold["held_out_policy_v1_macro_f1"]) for fold in folds]
    v2_scores = [float(fold["held_out_policy_v2_macro_f1"]) for fold in folds]
    return {
        "method": "leave_one_author_out",
        "metric_label": DEV_CV_LABEL,
        "note": (
            "Thresholds and Judge temperature are fit on the other authors only. "
            "This is leave-one-author-out dev CV. datasets/test is not frozen here."
        ),
        "folds": folds,
        "mean_held_out_policy_v1_macro_f1": sum(v1_scores) / len(v1_scores),
        "mean_held_out_policy_v2_macro_f1": sum(v2_scores) / len(v2_scores),
    }


def summarize_cached(cases: list[CachedCase], grid: list[Thresholds]) -> dict[str, Any]:
    """Held-out selection plus a descriptive full-split score that did not pick thresholds."""

    held_out = leave_one_author_out(cases, grid)
    v1 = PolicyEngine()
    v2 = PolicyEngineV2()
    gold = [case.gold_action for case in cases]
    v1_predicted = [v1.decide(case.context).final_action.value for case in cases]
    v2_decisions = [v2.decide(case.context) for case in cases]
    v2_predicted = [decision.final_action.value for decision in v2_decisions]
    interval = paired_bootstrap_interval(gold, v2_predicted, macro_f1, n_resamples=400, seed=0)
    scored = [
        ScoredCase(
            gold_action=case.gold_action,
            predicted_action=action,
            gold_judge=case.gold_judge or None,
            predicted_judge=None if case.context.judge is None else case.context.judge.label.value,
            retrieval_recall=case.retrieval_recall,
            latency_ms=case.latency_ms,
        )
        for case, action in zip(cases, v2_predicted, strict=True)
    ]
    return {
        "n_cases": len(cases),
        "metric_label": DEV_CV_LABEL,
        "headline": "leave_one_author_out",
        "leave_one_author_out": held_out,
        "descriptive_full_split": {
            "metric_label": "descriptive full dev split; not leave-one-author-out dev CV",
            "note": (
                "Default thresholds on the cached file. This is not a threshold "
                "selected on these same rows and is not a test score."
            ),
            "policy_v1_macro_f1": macro_f1(gold, v1_predicted),
            "policy_v2_macro_f1": interval.point,
            "policy_v2_macro_f1_ci95": [interval.low, interval.high],
        },
        "efficiency": efficiency_report(scored),
        "policy_v2_per_case": [
            {
                "case_id": case.case_id,
                "author": case.author,
                "gold_action": case.gold_action,
                "predicted_action": decision.final_action.value,
                "risk_tier": decision.risk_tier,
                "needs_verification": decision.needs_verification,
                "compliance_outcome": decision.compliance_outcome,
            }
            for case, decision in zip(cases, v2_decisions, strict=True)
        ],
    }


def project_cached(cases: list[CachedCase], root: Path, *, judge_temperature: float = 1.0) -> None:
    """Write the customer markdown vault from cached roles. No extra model call."""

    engine = PolicyEngineV2(judge_temperature=judge_temperature)
    for case in cases:
        context = case.context
        decision = engine.decide(context)
        project_bundle(
            root,
            CommitBundle(
                candidate=context.candidate,
                evidence=list(context.evidence),
                decision=decision,
                historian=context.historian,
                judge=context.judge,
                synthesizer=context.synthesizer,
                memory_before=list(context.related_claims),
                idempotency_key=context.idempotency_key,
                role_errors=list(context.role_errors),
                run_config=context.run_config,
            ),
        )
