"""Sweep policy-v1 thresholds on cached role outputs. No extra LLM calls.

Each input line is a JSON PolicyContext plus `gold_action`. The script prints the
rows with the best macro-F1, then the lowest unnecessary-quarantine rate among those.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sekf.contracts import PolicyContext, Thresholds
from sekf.evaluation.metrics import macro_f1, unnecessary_quarantine_rate
from sekf.policy import PolicyEngine


def sweep(
    contexts: list[tuple[PolicyContext, str]], grid: list[Thresholds]
) -> list[dict[str, Any]]:
    engine = PolicyEngine()
    rows: list[dict[str, Any]] = []
    gold = [label for _, label in contexts]
    for thresholds in grid:
        predicted: list[str] = []
        for context, _label in contexts:
            decision = engine.decide(context.model_copy(update={"thresholds": thresholds}))
            predicted.append(decision.final_action.value)
        rows.append(
            {
                "thresholds": thresholds.model_dump(mode="json"),
                "macro_f1": macro_f1(gold, predicted),
                "unnecessary_quarantine_rate": unnecessary_quarantine_rate(gold, predicted),
            }
        )
    return rows


def default_grid() -> list[Thresholds]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline policy-v1 threshold sweep.")
    parser.add_argument("contexts", type=Path)
    args = parser.parse_args(argv)
    loaded: list[tuple[PolicyContext, str]] = []
    try:
        text = args.contexts.read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if "gold_action" not in payload or "context" not in payload:
            print(f"line {line_no}: need context and gold_action", file=sys.stderr)
            return 2
        loaded.append(
            (PolicyContext.model_validate(payload["context"]), str(payload["gold_action"]))
        )
    rows = sweep(loaded, default_grid())
    best = max(rows, key=lambda row: (row["macro_f1"], -row["unnecessary_quarantine_rate"]))
    json.dump({"best": best, "rows": rows}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
