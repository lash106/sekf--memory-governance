"""Development pilot for the labeled cases under datasets/dev.

Loads `.env` if present, then `.env.example` for any unset keys. A live Judge and
single_judge run only when `SEKF_LLM_API_KEY` is non-empty. `.env.example` ships
with `SEKF_PROVIDER=fake` and a blank key, so that file alone is offline.

This file does not tune prompts or thresholds. It records what the current code
returns on the dev split.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sekf.api.orchestrator import Governance
from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.baselines.single_judge.baseline import SingleJudgeBaseline
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
from sekf.errors import ProviderUnavailable
from sekf.historian.deterministic import DeterministicHistorian
from sekf.judge.fake import FakeJudge
from sekf.judge.interface import Judge
from sekf.judge.llm.judge import LLMJudge
from sekf.memory.in_memory import InMemoryRepository
from sekf.policy.engine import PolicyEngine
from sekf.providers.openai_compatible import OpenAICompatibleProvider
from sekf.synthesizer.deterministic import DeterministicSynthesizer

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "datasets" / "dev"
OUT = ROOT / "experiments" / "dev_pilot_2026-10-29"
ACTIONS = [action.value for action in FinalAction]


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_env() -> list[str]:
    """`.env.example`, then `.env`, then `huggingface.env`. Process env wins."""

    loaded: list[str] = []
    merged: dict[str, str] = {}
    for name in (".env.example", ".env", "huggingface.env"):
        path = ROOT / name
        if not path.is_file():
            continue
        merged.update(parse_env_file(path))
        loaded.append(name)
    for key, value in merged.items():
        if key not in os.environ:
            os.environ[key] = value
    return loaded


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def live_configured() -> bool:
    return bool((os.getenv("SEKF_LLM_API_KEY") or "").strip())


def live_run_config() -> RunConfig:
    model = (os.getenv("SEKF_LLM_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    return RunConfig(model=model, temperature=0.0, seed=0, timeout_s=60.0)


def fake_run_config() -> RunConfig:
    return RunConfig(model="fake/none", temperature=0.0, seed=0, timeout_s=30.0)


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(DEV.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "gold" not in data or "candidate" not in data:
            continue
        data["_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        cases.append(data)
    return cases


def request_for(
    case: dict[str, Any], run: RunConfig
) -> tuple[DecideRequest, list[ExistingClaim]]:
    memory = [ExistingClaim.model_validate(item) for item in case["initial_memory"]]
    request = DecideRequest(
        candidate=CandidateClaim.model_validate(case["candidate"]),
        evidence=[EvidenceRecord.model_validate(item) for item in case["evidence"]],
        mode=DecideMode.DRY_RUN,
        run_config=run,
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


def _precision_recall(
    gold: list[str], pred: list[str], positive: str
) -> tuple[float | None, float | None]:
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


def static_rag() -> str:
    """Specified static RAG: retrieve and answer, never write governed memory."""

    return FinalAction.KEEP_EPISODIC.value


async def run_council(
    request: DecideRequest, memory: list[ExistingClaim], judge: Judge
) -> dict[str, Any]:
    gov = Governance(
        memory=InMemoryRepository(memory),
        historian=DeterministicHistorian(),
        judge=judge,
        synthesizer=DeterministicSynthesizer(),
        policy=PolicyEngine(),
    )
    started = time.perf_counter()
    response = await gov.decide(request)
    wall = int((time.perf_counter() - started) * 1000)
    return _row(response.decision.final_action.value, response, wall, response.judge)


async def run_ensemble(
    request: DecideRequest, memory: list[ExistingClaim], judge: Judge
) -> dict[str, Any]:
    baseline = GenericEnsembleBaseline(judge, memory=InMemoryRepository(memory))
    started = time.perf_counter()
    response = await baseline.decide(request)
    wall = int((time.perf_counter() - started) * 1000)
    return _row(response.decision.final_action.value, response, wall, response.judge)


async def with_retries(op, *, attempts: int = 6, label: str = "") -> dict[str, Any]:
    """Retry live calls on HTTP 429. Do not retry other failures."""

    delay = 5.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = await op()
        except ProviderUnavailable as exc:
            last_error = exc
            if "rate limited" not in str(exc).lower() or attempt == attempts:
                raise
            print(f"{label} 429, retry {attempt}/{attempts} in {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if result.get("error") == "ProviderUnavailable":
            if attempt == attempts:
                return result
            print(f"{label} provider unavailable, retry {attempt}/{attempts} in {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        return result
    if last_error is not None:
        raise last_error
    raise ProviderUnavailable(f"{label} failed after retries")


async def run_single_judge(
    request: DecideRequest, memory: list[ExistingClaim], provider: OpenAICompatibleProvider
) -> dict[str, Any]:
    baseline = SingleJudgeBaseline(provider, memory=InMemoryRepository(memory))
    started = time.perf_counter()
    response = await baseline.decide(request)
    wall = int((time.perf_counter() - started) * 1000)
    return _row(response.decision.final_action.value, response, wall, None)


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
        "error": response.role_errors[0].error_type if response.role_errors else None,
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
        "error": None,
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
        "errors": sum(1 for row in rows if row.get("error")),
    }


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_readme(summary: dict[str, Any], n_cases: int, sha: str, notes: list[str]) -> None:
    lines = [
        "# Development pilot, 29 October 2026",
        "",
        f"Git commit recorded at run time: `{sha}`.",
        f"Split: `dev`. Cases loaded: {n_cases}. Mode: `dry_run`. Temperature: 0. Seed: 0.",
        f"Env files loaded: {', '.join(summary['env_files']) or 'none'}.",
        f"Live API used: {'yes' if summary['live'] else 'no'}.",
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
            lines.append(f"| {key} | not run |  |  |  |  |  |  |  |  |  |")
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
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    env_files = load_env()
    live = live_configured()
    fake_cfg = fake_run_config()
    live_cfg = live_run_config() if live else fake_cfg
    cases = load_cases()
    provider: OpenAICompatibleProvider | None = None
    live_judge: Judge | None = None
    if live:
        provider = OpenAICompatibleProvider.from_env()
        live_judge = LLMJudge(provider)

    grouped: dict[str, list[dict[str, Any]]] = {
        "direct_storage": [],
        "latest_write_wins": [],
        "static_rag": [],
        "generic_ensemble": [],
        "proposed_council": [],
        "single_judge": [],
    }
    per_case: list[dict[str, Any]] = []
    for case in cases:
        fake_request, memory = request_for(case, fake_cfg)
        gold_action = case["gold"]["final_action"]
        gold_judge = case["gold"]["judge_label"]
        runs: dict[str, dict[str, Any]] = {
            "direct_storage": rule_row(direct_storage()),
            "latest_write_wins": rule_row(latest_write_wins(fake_request, memory)),
            "static_rag": rule_row(static_rag()),
            "generic_ensemble": await run_ensemble(fake_request, memory, FakeJudge()),
        }
        if live and provider is not None and live_judge is not None:
            live_request, live_memory = request_for(case, live_cfg)
            print(f"live {case['case_id']}", flush=True)
            runs["proposed_council"] = await with_retries(
                lambda: run_council(live_request, live_memory, live_judge),
                label=f"{case['case_id']} council",
            )
            runs["single_judge"] = await with_retries(
                lambda: run_single_judge(live_request, live_memory, provider),
                label=f"{case['case_id']} single_judge",
            )
            await asyncio.sleep(0.8)
        else:
            runs["proposed_council"] = await run_council(fake_request, memory, FakeJudge())
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

    if provider is not None:
        await provider.aclose()

    notes = [
        "direct_storage: specified rule in this script. Package is not in the repository. Every parsed candidate is PROMOTE.",
        "latest_write_wins: specified rule in this script. Package is not in the repository. REVISE when a related ACTIVE claim is older than the candidate; KEEP_EPISODIC when a related ACTIVE claim is not older; otherwise PROMOTE.",
        "static_rag: specified rule in this script. Package is not in the repository. Retrieval may answer a question, but it never writes governed memory, so the scored memory action is KEEP_EPISODIC.",
        (
            "single_judge: live Mem0-style baseline through the OpenAI-compatible provider."
            if live
            else "single_judge: not run. `.env.example` leaves SEKF_LLM_API_KEY empty and SEKF_PROVIDER=fake. Copy it to `.env` and set a key to run this row."
        ),
        "generic_ensemble: three FakeJudge votes. No network. A live three-call panel was not run, to keep the live budget on the council and single_judge.",
        (
            "proposed_council: DeterministicHistorian, LLMJudge, DeterministicSynthesizer, policy-v1, dry_run."
            if live
            else "proposed_council: DeterministicHistorian, FakeJudge, DeterministicSynthesizer, policy-v1, dry_run. No network."
        ),
    ]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "split": "dev",
        "n_cases": len(cases),
        "env_files": env_files,
        "live": live,
        "provider": (os.getenv("SEKF_PROVIDER") or "fake"),
        "model": live_cfg.model if live else "fake/none",
        "historian": "DeterministicHistorian",
        "synthesizer": "DeterministicSynthesizer",
        "policy": "policy-v1",
        "mode": "dry_run",
        "systems": {
            "direct_storage": summarize(grouped["direct_storage"]),
            "latest_write_wins": summarize(grouped["latest_write_wins"]),
            "static_rag": summarize(grouped["static_rag"]),
            "single_judge": summarize(grouped["single_judge"]) if grouped["single_judge"] else None,
            "generic_ensemble": summarize(grouped["generic_ensemble"]),
            "proposed_council": summarize(grouped["proposed_council"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_case:
            handle.write(json.dumps(row) + "\n")
    write_readme(summary, len(cases), summary["git_sha"], notes)
    print(json.dumps({"live": live, "env_files": env_files, "systems": summary["systems"]}, indent=2))
    print(f"cases {len(cases)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
