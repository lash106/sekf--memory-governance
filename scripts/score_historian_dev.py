"""Score the Historian alone on the labeled dev cases. Offline, no network.

Owner: Mahalakshmi. Runs the deterministic Historian on every case under ``datasets/dev``
that has gold Historian labels, then reports temporal-reasoning accuracy, conflict-type
accuracy, the joint score, accuracy per category, and the most common errors.

This is the Historian side of the Sprint 2 "Historian-only ablation on dev" and the input
for the Sprint 3 error analysis (provenance / time / conflict). It scores role labels
only; final actions are scored by the experiment runner.

Usage:
    python scripts/score_historian_dev.py                 # print a summary
    python scripts/score_historian_dev.py --json out.json # also save per-case rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sekf.contracts import CandidateClaim, EvidenceRecord, ExistingClaim, HistorianRequest
from sekf.evaluation.historian_metrics import (
    Score,
    accuracy_by_category,
    conflict_type_accuracy,
    confusion_counts,
    joint_historian_accuracy,
    temporal_relation_accuracy,
)
from sekf.historian import DeterministicHistorian
from sekf.memory.in_memory import InMemoryRepository

DEV = Path(__file__).resolve().parents[1] / "datasets" / "dev"


def load_cases(root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        gold = data.get("gold") if isinstance(data, dict) else None
        if gold and "candidate" in data:
            data["_path"] = str(path.relative_to(root))
            cases.append(data)
    return cases


async def predict(case: dict[str, Any]) -> dict[str, Any]:
    """Run the Historian on one case. A failure is recorded, not raised."""

    candidate = CandidateClaim.model_validate(case["candidate"])
    memory = InMemoryRepository(
        [ExistingClaim.model_validate(item) for item in case.get("initial_memory", [])]
    )
    request = HistorianRequest(
        candidate=candidate,
        related_claims=memory.search_related_claims(candidate),
        evidence=[EvidenceRecord.model_validate(item) for item in case.get("evidence", [])],
    )
    try:
        assessment = await DeterministicHistorian().analyze(request)
    except Exception as error:  # noqa: BLE001 - recorded per case for error analysis
        return {"temporal_relation": None, "conflict_type": None, "error": repr(error)}
    return {
        "temporal_relation": assessment.temporal_relation.value,
        "conflict_type": assessment.conflict_type.value,
        "provenance_quality": assessment.provenance_quality.value,
        "error": None,
    }


def fmt(score: Score) -> str:
    if score.value is None:
        return "n/a (0 cases)"
    return f"{score.value:.1%} ({score.correct}/{score.n})"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEV, help="folder of case files")
    parser.add_argument("--json", type=Path, help="write per-case rows to this file")
    args = parser.parse_args()

    cases = load_cases(args.root)
    rows = []
    for case in cases:
        gold = case["gold"]
        predicted = await predict(case)
        rows.append(
            {
                "case_id": case.get("case_id", case["_path"]),
                "path": case["_path"],
                "category": case.get("category", "unknown"),
                "gold_temporal_relation": gold.get("temporal_relation"),
                "gold_conflict_type": gold.get("conflict_type"),
                "predicted_temporal_relation": predicted["temporal_relation"],
                "predicted_conflict_type": predicted["conflict_type"],
                "predicted_provenance_quality": predicted.get("provenance_quality"),
                "error": predicted["error"],
            }
        )

    g_rel = [row["gold_temporal_relation"] for row in rows]
    p_rel = [row["predicted_temporal_relation"] for row in rows]
    g_con = [row["gold_conflict_type"] for row in rows]
    p_con = [row["predicted_conflict_type"] for row in rows]

    print(f"Historian (deterministic) on {len(rows)} cases from {args.root}")
    print(f"  temporal-reasoning accuracy: {fmt(temporal_relation_accuracy(g_rel, p_rel))}")
    print(f"  conflict-type accuracy:      {fmt(conflict_type_accuracy(g_con, p_con))}")
    joint = joint_historian_accuracy(g_rel, p_rel, g_con, p_con)
    print(f"  both right:                  {fmt(joint)}")
    print(f"  Historian errors raised:     {sum(1 for row in rows if row['error'])}")

    categories = [row["category"] for row in rows]
    print("\nTemporal-reasoning accuracy by category:")
    for category, score in accuracy_by_category(categories, g_rel, p_rel).items():
        print(f"  {category:<40} {fmt(score)}")

    for title, gold, pred in (
        ("temporal_relation", g_rel, p_rel),
        ("conflict_type", g_con, p_con),
    ):
        errors = confusion_counts(gold, pred).most_common()
        print(f"\nMost common {title} errors (gold -> predicted):")
        for (expected, got), count in errors or [(("none", "none"), 0)]:
            print(f"  {expected} -> {got}: {count}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nPer-case rows written to {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
