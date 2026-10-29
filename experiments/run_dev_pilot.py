"""Offline development pilot for the labeled cases under datasets/dev.

Runs the council (deterministic Historian and Synthesizer, fake Judge, policy-v1)
and generic_ensemble on each case in dry_run. Scores direct_storage and
latest_write_wins from the specified rules in this script. Those two packages are
not in the repository. static_rag and the live single_judge are not run: there is
no API key in this environment.

This file does not tune prompts or thresholds. It records what the current code
returns on the dev split.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sekf.api.orchestrator import Governance
from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import (
    CandidateClaim,
    ClaimStatus,
    DecideMode,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
    JudgeLabel,
    RunConfig,
)
from sekf.historian.deterministic import DeterministicHistorian
from sekf.judge.fake import FakeJudge
from sekf.memory.in_memory import InMemoryRepository
from sekf.policy.engine import PolicyEngine
from sekf.synthesizer.deterministic import DeterministicSynthesizer

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "datasets" / "dev"
OUT = ROOT / "experiments" / "dev_pilot_2026-10-29"
ACTIONS = [action.value for action in FinalAction]
RUN = RunConfig(model="fake/none", temperature=0.0, seed=0, timeout_s=30.0)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(DEV.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "gold" not in data or "candidate" not in data:
            continue
        data["_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        cases.append(data)
    return cases


def request_for(case: dict[str, Any]) -> tuple[DecideRequest, list[ExistingClaim]]:
    memory = [ExistingClaim.model_validate(item) for item in case["initial_memory"]]
    request = DecideRequest(
        candidate=CandidateClaim.model_validate(case["candidate"]),
        evidence=[EvidenceRecord.model_validate(item) for item in case["evidence"]],
        mode=DecideMode.DRY_RUN,
        run_config=RUN,
    )
    return request, memory


def _f1(gold: list[str], pred: list[str], positive: str) -> float | None:
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        if p == positive and g == positive:
            tp += 1
        elif p == positive and g != positive:
            fp += 1
        elif g == positive and p != positive:
            fn += 1
    if tp + fp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _precision_recall(gold: list[str], pred: list[str], positive: str) -> tuple[float | None, float | None]:
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        if p == positive and g == positive:
            tp += 1
        elif p == positive and g != positive:
            fp += 1
        elif g == positive and p != positive:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return precision, recall


def macro_f1(gold: list[str], pred: list[str]) -> float:
    scores: list[float] = []
    for label in ACTIONS:
        score = _f1(gold, pred, label)
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def latest_write_wins(request: DecideRequest, memory: list[ExistingClaim]) -> str:
    """Specified rule: revise when a related ACTIVE claim is older than the candidate."""

    active = [
        claim
        for claim in memory
        if claim.status is ClaimStatus.ACTIVE
        and claim.subject_key == request.candidate.subject_key
        and claim.predicate_key == request.candidate.predicate_key
        and claim.claim_id != request.candidate.claim_id
    ]
    newer = [claim for claim in active if request.candidate.observed_at > claim.updated_at]
    if newer:
        return FinalAction.REVISE.value
    if active:
        return FinalAction.KEEP_EPISODIC.value
    return FinalAction.PROMOTE.value


def direct_storage() -> str:
    return FinalAction.PROMOTE.value


async def run_council(request: DecideRequest, memory: list[ExistingClaim]) -> dict[str, Any]:
    gov = Governance(
        memory=InMemoryRepository(memory),
        historian=DeterministicHistorian(),
        judge=FakeJudge(),
        synthesizer=DeterministicSynthesizer(),
        policy=PolicyEngine(),
    )
    started = time.perf_counter()
    response = await gov.decide(request)
    wall = int((time.perf_counter() - started) * 1000)
    return _row(response.decision.final_action.value, response, wall, response.judge)


async def run_ensemble(request: DecideRequest, memory: list[ExistingClaim]) -> dict[str, Any]:
    baseline = GenericEnsembleBaseline(FakeJudge(), memory=InMemoryRepository(memory))
    started = time.perf_counter()
    response = await baseline.decide(request)
    wall = int((time.perf_counter() - started) * 1000)
    return _row(response.decision.final_action.value, response, wall, response.judge)


def _row(action: str, response: Any, wall_ms: int, judge: Any) -> dict[str, Any]:
    cost = response.decision.cost
    return {
        "final_action": action,
        "judge_label": None if judge is None else judge.label.value,
        "wall_ms": wall_ms,
        "llm_calls": cost.llm_calls,
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "estimated_cost_usd": cost.estimated_cost_usd,
        "rule_ids": list(response.decision.rule_ids_fired),
    }


def rule_row(action: str) -> dict[str, Any]:
    return {
        "final_action": action,
        "judge_label": None,
        "wall_ms": 0,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "rule_ids": [],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_action"] for row in rows]
    pred = [row["final_action"] for row in rows]
    precision, recall = _precision_recall(gold, pred, FinalAction.PROMOTE.value)
    revision_gold = [row for row in rows if row["gold_action"] == FinalAction.REVISE.value]
    revision = (
        None
        if not revision_gold
        else sum(row["final_action"] == FinalAction.REVISE.value for row in revision_gold)
        / len(revision_gold)
    )
    judge_gold = [row["gold_judge"] for row in rows if row["judge_label"] is not None]
    judge_pred = [row["judge_label"] for row in rows if row["judge_label"] is not None]
    contradiction = None
    if judge_gold:
        contradiction = _f1(judge_gold, judge_pred, JudgeLabel.REFUTED.value)
        if contradiction is None:
            contradiction = 0.0
    walls = [row["wall_ms"] for row in rows]
    return {
        "n": len(rows),
        "final_action_macro_f1": round(macro_f1(gold, pred), 4),
        "promotion_precision": None if precision is None else round(precision, 4),
        "promotion_recall": None if recall is None else round(recall, 4),
        "revision_accuracy": None if revision is None else round(revision, 4),
        "contradiction_f1_refuted": None if contradiction is None else round(contradiction, 4),
        "median_latency_ms": int(statistics.median(walls)) if walls else 0,
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "tokens": sum(row["input_tokens"] + row["output_tokens"] for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in rows), 6),
        "predicted_actions": dict(Counter(pred)),
    }


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_readme(summary: dict[str, Any], n_cases: int, sha: str) -> None:
    systems = [
        ("direct_storage", "Specified rule in this script. Package is not in the repository. Every parsed candidate is PROMOTE."),
        ("latest_write_wins", "Specified rule in this script. Package is not in the repository. REVISE when a related ACTIVE claim is older than the candidate; KEEP_EPISODIC when a related ACTIVE claim is not older; otherwise PROMOTE."),
        ("static_rag", "Not run. The package is not in the repository and this pilot makes no model call."),
        ("single_judge", "Not run. No SEKF_LLM_API_KEY is configured, so the live one-model baseline was not called."),
        ("generic_ensemble", "Ran. Three FakeJudge votes. No network."),
        ("proposed council", "Ran. DeterministicHistorian, FakeJudge, DeterministicSynthesizer, policy-v1, dry_run. No network."),
    ]
    lines = [
        "# Development pilot, 29 October 2026",
        "",
        f"Git commit recorded at run time: `{sha}`.",
        f"Split: `dev`. Cases loaded: {n_cases}. Mode: `dry_run`. Temperature: 0. Seed: 0.",
        "Judge for the council and the ensemble: `FakeJudge` (SUPPORTED when evidence text contains the object text, otherwise INSUFFICIENT_EVIDENCE). It does not emit REFUTED, so contradiction F1 on the REFUTED class is 0 whenever that gold label is present.",
        "Macro-F1 averages F1 over the seven final actions that appear in the gold labels or the predictions. Promotion precision and recall use PROMOTE. Revision accuracy is the fraction of gold REVISE cases that returned REVISE. Contradiction F1 uses Judge label REFUTED as the positive class and is reported only when the system returns a Judge label.",
        "These numbers are a development pilot. They are not a held-out test result, and prompts were not changed to improve them.",
        "",
        "| System | n | Final-action macro-F1 | Promotion precision | Promotion recall | Revision accuracy | Contradiction F1 | Median latency (ms) | LLM calls | Tokens | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = [
        "direct_storage",
        "latest_write_wins",
        "static_rag",
        "single_judge",
        "generic_ensemble",
        "proposed_council",
    ]
    for key in order:
        block = summary["systems"][key]
        if block is None:
            lines.append(f"| {key} |  |  |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {name} | {n} | {f1} | {pp} | {pr} | {rev} | {c} | {lat} | {calls} | {tok} | {cost} |".format(
                name=key,
                n=block["n"],
                f1=fmt(block["final_action_macro_f1"]),
                pp=fmt(block["promotion_precision"]),
                pr=fmt(block["promotion_recall"]),
                rev=fmt(block["revision_accuracy"]),
                c=fmt(block["contradiction_f1_refuted"]),
                lat=block["median_latency_ms"],
                calls=block["llm_calls"],
                tok=block["tokens"],
                cost=fmt(block["estimated_cost_usd"]),
            )
        )
    lines.extend(["", "## What each row means", ""])
    for name, note in systems:
        lines.append(f"- {name}: {note}")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    cases = load_cases()
    per_case: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {
        "direct_storage": [],
        "latest_write_wins": [],
        "generic_ensemble": [],
        "proposed_council": [],
    }
    for case in cases:
        request, memory = request_for(case)
        gold_action = case["gold"]["final_action"]
        gold_judge = case["gold"]["judge_label"]
        runs = {
            "direct_storage": rule_row(direct_storage()),
            "latest_write_wins": rule_row(latest_write_wins(request, memory)),
            "generic_ensemble": await run_ensemble(request, memory),
            "proposed_council": await run_council(request, memory),
        }
        for name, result in runs.items():
            row = {
                "case_id": case["case_id"],
                "path": case["_path"],
                "category": case["category"],
                "gold_action": gold_action,
                "gold_judge": gold_judge,
                **result,
            }
            grouped[name].append(row)
            per_case.append({"system": name, **row})

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "split": "dev",
        "n_cases": len(cases),
        "judge": "FakeJudge",
        "historian": "DeterministicHistorian",
        "synthesizer": "DeterministicSynthesizer",
        "policy": "policy-v1",
        "mode": "dry_run",
        "systems": {
            "direct_storage": summarize(grouped["direct_storage"]),
            "latest_write_wins": summarize(grouped["latest_write_wins"]),
            "static_rag": None,
            "single_judge": None,
            "generic_ensemble": summarize(grouped["generic_ensemble"]),
            "proposed_council": summarize(grouped["proposed_council"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case:
            handle.write(json.dumps(row) + "\n")
    write_readme(summary, len(cases), summary["git_sha"])
    print(json.dumps(summary["systems"], indent=2))
    print(f"cases {len(cases)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
