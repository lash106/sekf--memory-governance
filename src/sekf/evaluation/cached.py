"""Cached Historian, Judge, and Synthesizer outputs. No network.

A predictions file is JSONL. Each line is one dev case plus the raw role JSON and the
`PolicyContext` those roles produced. Threshold sweeps and the offline scorer read this
file instead of calling a model again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sekf.contracts import PolicyContext

_AUTHORS = ("krishna", "mahalakshmi", "shailen")


def author_of(path: str) -> str:
    """Author folder under datasets/dev. Top-level cases belong to Sai Chetan."""

    parts = {part.lower() for part in Path(path).parts}
    for name in _AUTHORS:
        if name in parts:
            return name
    return "sai_chetan"


@dataclass(frozen=True)
class CachedCase:
    case_id: str
    author: str
    gold_action: str
    gold_judge: str
    category: str
    latency_ms: float
    retrieval_recall: float | None
    context: PolicyContext


def load_predictions(path: Path) -> list[CachedCase]:
    """Read a predictions JSONL file. A bad line raises ValueError."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    cases: list[CachedCase] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_no}: prediction must be an object")
        cases.append(_case_from_payload(payload, line_no))
    return cases


def _case_from_payload(payload: dict[str, Any], line_no: int) -> CachedCase:
    prefix = f"line {line_no}: "
    if "context" not in payload or "gold_action" not in payload:
        raise ValueError(prefix + "need context and gold_action")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(prefix + "case_id must be a non-empty string")
    author = payload.get("author")
    if not isinstance(author, str) or not author:
        source = payload.get("source_path")
        if not isinstance(source, str) or not source:
            raise ValueError(prefix + "need author or source_path")
        author = author_of(source)
    gold_judge = payload.get("gold_judge", payload.get("gold_label", ""))
    if not isinstance(gold_judge, str):
        raise ValueError(prefix + "gold_judge must be a string")
    category = payload.get("category") or "uncategorized"
    if not isinstance(category, str):
        raise ValueError(prefix + "category must be a string")
    latency = payload.get("latency_ms", 0)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise ValueError(prefix + "latency_ms must be a number")
    recall = payload.get("retrieval_recall")
    if recall is not None and (isinstance(recall, bool) or not isinstance(recall, (int, float))):
        raise ValueError(prefix + "retrieval_recall must be a number or null")
    try:
        context = PolicyContext.model_validate(payload["context"])
    except ValueError as exc:
        raise ValueError(prefix + "context is not a PolicyContext") from exc
    return CachedCase(
        case_id=case_id,
        author=author,
        gold_action=str(payload["gold_action"]),
        gold_judge=gold_judge,
        category=category,
        latency_ms=float(latency),
        retrieval_recall=None if recall is None else float(recall),
        context=context,
    )
