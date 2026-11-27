"""Offline experiment runner for datasets/dev. No network.

Scores the real packages in `src/sekf/baselines/` (direct_storage, latest_write_wins,
static_rag, generic_ensemble with FakeJudge) plus the governed council on policy-v1.
single_judge is reported as not_run: it needs a live provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from sekf.api.orchestrator import Governance
from sekf.baselines import DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline
from sekf.baselines.generic_ensemble.baseline import GenericEnsembleBaseline
from sekf.contracts import (
    CandidateClaim,
    DecideMode,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
)
from sekf.evaluation.metrics import (
    false_promotion_rate,
    macro_f1,
    paired_bootstrap_interval,
    percentile,
    promotion_scores,
    revision_accuracy,
    unnecessary_quarantine_rate,
)
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge
from sekf.memory.in_memory import InMemoryRepository
from sekf.memory.retrieval import hybrid_search, retrieval_recall_at_k
from sekf.policy import PolicyEngine
from sekf.synthesizer import DeterministicSynthesizer

_DEV = Path(__file__).resolve().parents[3] / "datasets" / "dev"
_STORAGE = {
    "direct_storage": DirectStorageBaseline,
    "latest_write_wins": LatestWriteWinsBaseline,
    "static_rag": StaticRagBaseline,
}


def load_dev_cases(root: Path | None = None) -> list[dict[str, Any]]:
    """Load labeled case files. Files without a gold object are skipped."""

    base = root or _DEV
    cases: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "gold" in data and "candidate" in data:
            data["_path"] = str(path)
            cases.append(data)
    return cases


def _request(data: dict[str, Any]) -> DecideRequest:
    return DecideRequest(
        candidate=CandidateClaim.model_validate(data["candidate"]),
        evidence=[EvidenceRecord.model_validate(item) for item in data.get("evidence", [])],
        mode=DecideMode.DRY_RUN,
    )


def _memory(data: dict[str, Any]) -> InMemoryRepository:
    claims = [ExistingClaim.model_validate(item) for item in data.get("initial_memory", [])]
    return InMemoryRepository(claims)


async def _council_action(data: dict[str, Any]) -> tuple[str, int, int, float]:
    request = _request(data)
    governance = Governance(
        memory=_memory(data),
        historian=DeterministicHistorian(),
        judge=FakeJudge(),
        synthesizer=DeterministicSynthesizer(),
        policy=PolicyEngine(),
    )
    response = await governance.decide(request)
    cost = response.decision.cost
    return (
        response.decision.final_action.value,
        cost.input_tokens + cost.output_tokens,
        cost.llm_calls,
        cost.estimated_cost_usd,
    )


async def _baseline_action(name: str, data: dict[str, Any]) -> tuple[str, int, int, float]:
    """Run one packaged baseline in dry_run and return action plus cost fields."""

    request = _request(data)
    memory = _memory(data)
    if name == "generic_ensemble":
        response = await GenericEnsembleBaseline(FakeJudge(), memory=memory).decide(request)
    else:
        response = await _STORAGE[name](memory).decide(request)
    cost = response.decision.cost
    return (
        response.decision.final_action.value,
        cost.input_tokens + cost.output_tokens,
        cost.llm_calls,
        cost.estimated_cost_usd,
    )


def _recall(data: dict[str, Any]) -> float | None:
    candidate = CandidateClaim.model_validate(data["candidate"])
    claims = [ExistingClaim.model_validate(item) for item in data.get("initial_memory", [])]
    gold = data["gold"]
    delta = gold.get("expected_memory_delta") or {}
    relevant = [str(item) for item in delta.get("superseded_claim_ids", [])]
    if not relevant:
        relevant = [
            str(item["claim_id"])
            for item in data.get("initial_memory", [])
            if item.get("predicate_key") == data["candidate"]["predicate_key"]
        ]
    retrieved = [str(claim.claim_id) for claim in hybrid_search(candidate, claims, k=5)]
    return retrieval_recall_at_k(retrieved, relevant)


def _summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_action"] for row in rows]
    pred = [row["predicted_action"] for row in rows]
    precision, recall = promotion_scores(gold, pred)
    interval = paired_bootstrap_interval(gold, pred, macro_f1, n_resamples=400, seed=0)
    judge_rows = sum(1 for row in rows if row.get("gold_judge"))
    false_rate, false_hits, false_n = false_promotion_rate(
        gold,
        pred,
        gold_judge=[row.get("gold_judge") or "" for row in rows],
        gold_conflict=[row.get("gold_conflict") or "NONE" for row in rows],
    )
    recalls = [row["retrieval_recall"] for row in rows if row["retrieval_recall"] is not None]
    return {
        "system": name,
        "n": len(rows),
        "macro_f1": interval.point,
        "macro_f1_ci95": [interval.low, interval.high],
        "promotion_precision": precision,
        "promotion_recall": recall,
        "revision_accuracy": revision_accuracy(gold, pred),
        "unnecessary_quarantine_rate": unnecessary_quarantine_rate(gold, pred),
        "false_promotion_rate": false_rate,
        "false_promotion_hits": false_hits,
        "false_promotion_support": false_n,
        "retrieval_recall_at_5": (sum(recalls) / len(recalls)) if recalls else None,
        "p95_latency_ms": percentile([float(row["latency_ms"]) for row in rows], 95),
        "tokens": sum(row["tokens"] for row in rows),
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "estimated_cost_usd": sum(row["cost"] for row in rows),
        "judge_labels_present": judge_rows,
        "contradiction_f1_note": (
            "Judge contradiction F1 is computed when a system returns a Judge label. "
            "The deterministic baselines do not."
        ),
    }


async def run_dev_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one list of case objects. `cases` already includes the gold object."""

    buckets: dict[str, list[dict[str, Any]]] = {
        "direct_storage": [],
        "latest_write_wins": [],
        "static_rag": [],
        "generic_ensemble": [],
        "policy-v1": [],
    }
    for data in cases:
        gold = data["gold"]
        gold_action = str(gold["final_action"])
        recall = _recall(data)
        council = await _council_action(data)
        actions: dict[str, tuple[str, int, int, float]] = {
            "direct_storage": await _baseline_action("direct_storage", data),
            "latest_write_wins": await _baseline_action("latest_write_wins", data),
            "static_rag": await _baseline_action("static_rag", data),
            "generic_ensemble": await _baseline_action("generic_ensemble", data),
            "policy-v1": council,
        }
        for name, (action, tokens, calls, cost) in actions.items():
            row = {
                "case_id": data.get("case_id", data.get("_path")),
                "gold_action": gold_action,
                "predicted_action": action,
                "gold_judge": gold.get("judge_label"),
                "gold_conflict": gold.get("conflict_type"),
                "retrieval_recall": recall,
                "latency_ms": 0,
                "tokens": tokens,
                "llm_calls": calls,
                "cost": cost,
            }
            buckets[name].append(row)
    summaries = [_summarize(name, rows) for name, rows in buckets.items()]
    summaries.append(
        {
            "system": "single_judge",
            "status": "not_run",
            "reason": (
                "needs SEKF_PROVIDER=openai and SEKF_LLM_API_KEY; no substitute model is used"
            ),
        }
    )
    return {
        "n_cases": len(cases),
        "systems": summaries,
    }


async def start_offline_run(app: Any, *, limit: int = 0) -> dict[str, Any]:
    """Run the dev split, store the summary on the app, and return it."""

    cases = load_dev_cases()
    if limit > 0:
        cases = cases[:limit]
    if limit < 0:
        raise ValueError("limit must be >= 0")
    summary = await run_dev_cases(cases)
    run_id = str(uuid4())
    summary["run_id"] = run_id
    runs = getattr(app.state, "experiments", None)
    if runs is None:
        runs = {}
        app.state.experiments = runs
    runs[run_id] = summary
    return summary
