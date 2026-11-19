"""Run the council once and store raw role JSON for later offline scoring.

`--live` uses the OpenAI-compatible provider and retries HTTP 429. The default path
uses the deterministic Historian, fake Judge, and deterministic Synthesizer, so tests
stay offline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sekf.api.orchestrator import Governance
from sekf.contracts import DecideResponse, PolicyContext, RoleFailure, Thresholds
from sekf.errors import ProviderUnavailable
from sekf.evaluation.cached import author_of
from sekf.evaluation.retry import with_retries
from sekf.evaluation.runner import _memory, _recall, _request
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge
from sekf.policy import PolicyEngine
from sekf.synthesizer import DeterministicSynthesizer


def _rate_limited(errors: list[RoleFailure]) -> bool:
    return any(
        err.error_type == "ProviderUnavailable" and "rate limited" in err.message.lower()
        for err in errors
    )


def prediction_record(
    data: dict[str, Any],
    response: DecideResponse,
    *,
    latency_ms: float,
) -> dict[str, Any]:
    """One JSONL object: gold, raw roles, and the PolicyContext to replay."""

    request = _request(data)
    memory = _memory(data)
    related = memory.search_related_claims(request.candidate)
    context = PolicyContext(
        candidate=request.candidate,
        evidence=list(request.evidence),
        related_claims=related,
        historian=response.historian,
        judge=response.judge,
        synthesizer=response.synthesizer,
        thresholds=request.thresholds or Thresholds(),
        role_errors=list(response.role_errors),
        run_config=request.run_config,
        idempotency_key=request.idempotency_key,
        now=response.decision.created_at,
    )
    judge = response.judge
    gold = data.get("gold") or {}
    source = str(data.get("_path") or data.get("case_id"))
    evidence_ids = [str(item.evidence_id) for item in request.evidence]
    record: dict[str, Any] = {
        "case_id": str(data.get("case_id") or source),
        "author": author_of(source),
        "source_path": source,
        "category": str(data.get("category") or "uncategorized"),
        "gold_action": str(gold.get("final_action") or ""),
        "gold_judge": str(gold.get("judge_label") or ""),
        "gold_label": str(gold.get("judge_label") or ""),
        "latency_ms": latency_ms,
        "retrieval_recall": _recall(data),
        "roles": {
            "historian": None
            if response.historian is None
            else response.historian.model_dump(mode="json"),
            "judge": None if judge is None else judge.model_dump(mode="json"),
            "synthesizer": None
            if response.synthesizer is None
            else response.synthesizer.model_dump(mode="json"),
        },
        "context": context.model_dump(mode="json"),
    }
    if judge is not None:
        record.update(
            {
                "predicted_label": judge.label.value,
                "confidence": judge.confidence,
                "supporting_evidence_ids": [str(item) for item in judge.supporting_evidence_ids],
                "refuting_evidence_ids": [str(item) for item in judge.refuting_evidence_ids],
                "provided_evidence_ids": evidence_ids,
            }
        )
    return record


async def capture_cases(
    cases: list[dict[str, Any]],
    dest: Path,
    *,
    historian: Any,
    judge: Any,
    synthesizer: Any,
) -> int:
    """Write predictions.jsonl. HTTP 429 inside a role is retried; other errors are not."""

    lines: list[str] = []
    for data in cases:
        request = _request(data)
        governance = Governance(
            memory=_memory(data),
            historian=historian,
            judge=judge,
            synthesizer=synthesizer,
            policy=PolicyEngine(),
        )

        async def _once(gov: Governance = governance, req: Any = request) -> DecideResponse:
            response = await gov.decide(req)
            if _rate_limited(list(response.role_errors)):
                raise ProviderUnavailable("LLM rate limited (HTTP 429)")
            return response

        started = time.perf_counter()
        response = await with_retries(_once, label=str(data.get("case_id", "")))
        latency_ms = (time.perf_counter() - started) * 1000
        lines.append(json.dumps(prediction_record(data, response, latency_ms=latency_ms)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def offline_roles() -> tuple[Any, Any, Any]:
    """Deterministic Historian, fake Judge, deterministic Synthesizer."""

    return DeterministicHistorian(), FakeJudge(), DeterministicSynthesizer()
