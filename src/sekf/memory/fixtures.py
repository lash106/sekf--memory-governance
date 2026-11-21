"""Reusable initial-memory fixtures built from labeled dataset cases. Owner: Krishna.

A dataset case (`datasets/<split>/.../*.json`) holds the memory a system starts from
(`initial_memory`), the candidate and evidence it receives, and the gold final action and
memory delta. These helpers load a case and build a fresh, seeded store for it, so a test, a
baseline run and the experiment runner all start from exactly the same memory.

Only `SqlMemoryRepository` performs state transitions. `InMemoryRepository` records bundles
without changing claims, so it cannot be used to check a memory delta.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sekf.contracts import (
    CandidateClaim,
    EvidenceRecord,
    ExistingClaim,
    FinalAction,
)
from sekf.memory.sql import SqlMemoryRepository

DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets"

# Fixed so a store built twice for the same case is identical.
FIXTURE_NOW = datetime(2026, 10, 21, tzinfo=UTC)


@dataclass(frozen=True)
class MemoryCase:
    """One labeled case, parsed into contract objects."""

    case_id: str
    category: str
    candidate: CandidateClaim
    evidence: list[EvidenceRecord]
    initial_memory: list[ExistingClaim]
    gold_action: FinalAction
    # Raw `gold.expected_memory_delta`: bucket name -> claim or evidence ids as strings.
    expected_delta: dict[str, list[str]]

    @classmethod
    def from_dict(cls, case: dict[str, Any]) -> MemoryCase:
        gold = case["gold"]
        return cls(
            case_id=case["case_id"],
            category=case.get("category", ""),
            candidate=CandidateClaim.model_validate(case["candidate"]),
            evidence=[EvidenceRecord.model_validate(e) for e in case["evidence"]],
            initial_memory=[ExistingClaim.model_validate(c) for c in case["initial_memory"]],
            gold_action=FinalAction(gold["final_action"]),
            expected_delta={k: list(v) for k, v in gold["expected_memory_delta"].items()},
        )


def load_case(path: Path) -> MemoryCase:
    return MemoryCase.from_dict(json.loads(path.read_text(encoding="utf-8")))


def case_paths(split: str = "dev", owner: str | None = None) -> list[Path]:
    """Case files of a split, optionally only one owner's folder (for example `krishna`)."""

    folder = DATASETS_ROOT / split / owner if owner else DATASETS_ROOT / split
    return sorted(folder.rglob("*.json"))


def load_cases(split: str = "dev", owner: str | None = None) -> list[MemoryCase]:
    return [load_case(path) for path in case_paths(split, owner)]


def build_seeded_store(
    case: MemoryCase, *, clock: Callable[[], datetime] = lambda: FIXTURE_NOW
) -> SqlMemoryRepository:
    """A fresh in-memory SQLite store holding the case's `initial_memory` and nothing else."""

    repo = SqlMemoryRepository.from_url("sqlite://", clock=clock)
    repo.seed_claims(case.initial_memory)
    return repo
