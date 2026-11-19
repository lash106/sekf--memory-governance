"""Offline experiment runner for datasets/dev. No network.

Scores direct_storage, latest_write_wins, static_rag, generic_ensemble (FakeJudge),
policy-v1, and policy-v2. single_judge is reported as not_run: it needs a live provider.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from sekf.api.orchestrator import Governance
from sekf.contracts import (
    CandidateClaim,
    DecideMode,
    DecideRequest,
    EvidenceRecord,
    ExistingClaim,
)
from sekf.evaluation.metrics import (
    ScoredCase,
    efficiency_report,
    executive_accuracy,
    executive_action,
    false_promotion_rate,
    macro_f1,
    paired_bootstrap_interval,
    percentile,
    promotion_scores,
    rate_where,
    revision_accuracy,
    unnecessary_quarantine_rate,
)
from sekf.evaluation.systems import (
    direct_storage_action,
    latest_write_wins_action,
    static_rag_action,
)
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge
from sekf.memory.in_memory import InMemoryRepository
from sekf.memory.retrieval import hybrid_search, retrieval_recall_at_k
from sekf.policy import PolicyEngine, PolicyEngineV2, compliance_outcome
from sekf.policy.v2 import EPISODIC_TTL_DAYS
from sekf.synthesizer import DeterministicSynthesizer

_DEV = Path(__file__).resolve().parents[3] / "datasets" / "dev"


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


async def _council_action(
    data: dict[str, Any], policy: PolicyEngine
) -> tuple[str, int, int, float, float, str | None]:
    request = _request(data)
    governance = Governance(
        memory=_memory(data),
        historian=DeterministicHistorian(),
        judge=FakeJudge(),
        synthesizer=DeterministicSynthesizer(),
        policy=policy,
    )
    started = time.perf_counter()
    response = await governance.decide(request)
    latency_ms = (time.perf_counter() - started) * 1000
    cost = response.decision.cost
    judge_label = response.judge.label.value if response.judge is not None else None
    return (
        response.decision.final_action.value,
        cost.input_tokens + cost.output_tokens,
        cost.llm_calls,
        cost.estimated_cost_usd,
        latency_ms,
        judge_label,
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
    # Council rows do not store a separate predicted judge label in this summary.
    false_rate, false_hits, false_n = false_promotion_rate(
        gold,
        pred,
        gold_judge=[row.get("gold_judge") or "" for row in rows],
        gold_conflict=[row.get("gold_conflict") or "NONE" for row in rows],
    )
    recalls = [row["retrieval_recall"] for row in rows if row["retrieval_recall"] is not None]
    executives = [
        (row["gold_executive"], row["predicted_executive"])
        for row in rows
        if row.get("gold_executive")
    ]
    return {
        "system": name,
        "n": len(rows),
        "macro_f1": interval.point,
        "macro_f1_ci95": [interval.low, interval.high],
        "promotion_precision": precision,
        "promotion_recall": recall,
        "revision_accuracy": revision_accuracy(gold, pred),
        "unnecessary_quarantine_rate": unnecessary_quarantine_rate(gold, pred),
        "escalation_rate": rate_where(pred, "ESCALATE_FOR_REVIEW"),
        "false_promotion_rate": false_rate,
        "false_promotion_hits": false_hits,
        "false_promotion_support": false_n,
        "retrieval_recall_at_5": (sum(recalls) / len(recalls)) if recalls else None,
        "executive_accuracy": executive_accuracy(
            [pair[0] for pair in executives], [pair[1] for pair in executives]
        )
        if executives
        else None,
        "p95_latency_ms": percentile([float(row["latency_ms"]) for row in rows], 95),
        "efficiency": efficiency_report(
            [
                ScoredCase(
                    gold_action=row["gold_action"],
                    predicted_action=row["predicted_action"],
                    gold_judge=row.get("gold_judge"),
                    predicted_judge=row.get("predicted_judge"),
                    retrieval_recall=row.get("retrieval_recall"),
                    latency_ms=float(row["latency_ms"]),
                )
                for row in rows
            ]
        ),
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

    per_case: list[dict[str, Any]] = []
    buckets: dict[str, list[dict[str, Any]]] = {
        "direct_storage": [],
        "latest_write_wins": [],
        "static_rag": [],
        "policy-v1": [],
        "policy-v2": [],
    }
    for data in cases:
        gold = data["gold"]
        gold_action = str(gold["final_action"])
        request = _request(data)
        memory = _memory(data)
        related = memory.search_related_claims(request.candidate)
        v1 = await _council_action(data, PolicyEngine())
        v2 = await _council_action(data, PolicyEngineV2())
        details = {
            "direct_storage": (direct_storage_action().value, 0, 0, 0.0, 0.0, None),
            "latest_write_wins": (
                latest_write_wins_action(request, related).value,
                0,
                0,
                0.0,
                0.0,
                None,
            ),
            "static_rag": (static_rag_action().value, 0, 0, 0.0, 0.0, None),
            "policy-v1": v1,
            "policy-v2": v2,
        }
        recall = _recall(data)
        compliance = compliance_outcome(request.candidate)
        for name, detail in details.items():
            action, tokens, calls, cost, latency_ms, predicted_judge = detail
            row = {
                "case_id": data.get("case_id", data.get("_path")),
                "gold_action": gold_action,
                "predicted_action": action,
                "gold_judge": gold.get("judge_label"),
                "predicted_judge": predicted_judge,
                "gold_conflict": gold.get("conflict_type"),
                "gold_executive": gold.get("executive_action"),
                "predicted_executive": executive_action(action, compliance),
                "retrieval_recall": recall,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "llm_calls": calls,
                "cost": cost,
            }
            buckets[name].append(row)
            if name == "policy-v2":
                per_case.append(row)
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
    summaries.append(
        {
            "system": "generic_ensemble",
            "status": "not_run_in_batch",
            "reason": (
                "Use POST /v1/baselines/generic_ensemble/decide. "
                "The batch runner keeps the council on DeterministicHistorian and FakeJudge."
            ),
        }
    )
    return {
        "n_cases": len(cases),
        "metric_label": (
            "descriptive full dev split; not leave-one-author-out dev CV and not a test score"
        ),
        "episodic_ttl_days": EPISODIC_TTL_DAYS,
        "systems": summaries,
        "policy_v2_per_case": per_case,
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
