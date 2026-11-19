"""Write an offline dev-split score file. No network.

python scripts/run_offline_experiment.py --out experiments/offline_dev.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sekf.evaluation.runner import load_dev_cases, run_dev_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score dev cases with fake roles.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        print("limit must be >= 0", file=sys.stderr)
        return 2
    cases = load_dev_cases()
    if args.limit:
        cases = cases[: args.limit]
    summary = asyncio.run(run_dev_cases(cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"cases {summary['n_cases']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
