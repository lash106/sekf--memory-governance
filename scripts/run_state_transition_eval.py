"""Score the storage baselines (and an offline council stand-in) on labeled dev cases.

For every case the script builds a fresh store holding the case's `initial_memory`, runs each
system in `commit` mode, and compares what actually changed in memory with the gold
`expected_memory_delta`. No network and no LLM: the three baselines are deterministic, and the
council row uses the offline Historian, Synthesizer and FakeJudge, so it is a wiring reference,
NOT the real system's accuracy. Real-model results come from the pilot runner.

Run:  python scripts/run_state_transition_eval.py [--owner krishna] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sekf.api import Governance
from sekf.baselines import DirectStorageBaseline, LatestWriteWinsBaseline, StaticRagBaseline
from sekf.evaluation.memory_delta import (
    CaseOutcome,
    Decider,
    action_accuracy,
    evaluate_case,
    summarize_comparisons,
)
from sekf.historian import DeterministicHistorian
from sekf.judge import FakeJudge
from sekf.memory import MemoryRepository
from sekf.memory.fixtures import MemoryCase, load_cases
from sekf.policy import PolicyEngine
from sekf.synthesizer import DeterministicSynthesizer

Factory = Callable[[MemoryRepository, Callable[[], datetime]], Decider]


def _council(repo: MemoryRepository, clock: Callable[[], datetime]) -> Decider:
    return Governance(
        memory=repo,
        historian=DeterministicHistorian(),
        judge=FakeJudge(),
        synthesizer=DeterministicSynthesizer(),
        policy=PolicyEngine(),
        clock=clock,
    )


SYSTEMS: dict[str, Factory] = {
    "direct_storage": lambda repo, clock: DirectStorageBaseline(repo, clock=clock),
    "latest_write_wins": lambda repo, clock: LatestWriteWinsBaseline(repo, clock=clock),
    "static_rag": lambda repo, clock: StaticRagBaseline(repo, clock=clock),
    "council_offline_reference": _council,
}


async def run(cases: list[MemoryCase]) -> dict[str, list[CaseOutcome]]:
    return {
        name: [await evaluate_case(factory, case) for case in cases]
        for name, factory in SYSTEMS.items()
    }


def report(cases: list[MemoryCase], results: dict[str, list[CaseOutcome]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for name, outcomes in results.items():
        summary = summarize_comparisons([o.comparison for o in outcomes])
        by_category: dict[str, list[CaseOutcome]] = {}
        for case, outcome in zip(cases, outcomes, strict=True):
            by_category.setdefault(case.category or "uncategorized", []).append(outcome)
        systems[name] = {
            "action_accuracy": action_accuracy(outcomes),
            "delta": summary,
            "delta_exact_match_by_category": {
                cat: sum(o.comparison.exact_match for o in group) / len(group)
                for cat, group in sorted(by_category.items())
            },
            "actions": dict(Counter(o.response.decision.final_action.value for o in outcomes)),
            "wrong_delta_cases": [o.case_id for o in outcomes if not o.comparison.exact_match],
        }
    return {"cases": len(cases), "systems": systems}


def to_markdown(data: dict[str, Any]) -> str:
    lines = [
        f"Cases: {data['cases']}",
        "",
        "| System | Action accuracy | Delta exact match |",
        "|---|---|---|",
    ]
    for name, s in data["systems"].items():
        d = s["delta"]
        lines.append(
            f"| {name} | {s['action_accuracy']:.2f} | "
            f"{d['exact_match']}/{d['cases']} ({d['exact_match_rate']:.2f}) |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--owner", default=None, help="only one owner's folder, e.g. krishna")
    parser.add_argument("--json", default=None, help="write the full report to this path")
    args = parser.parse_args()
    cases = load_cases(args.split, args.owner)
    data = report(cases, asyncio.run(run(cases)))
    print(to_markdown(data))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
