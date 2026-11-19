"""Score the dev split without a new model call when a predictions file is given.

python scripts/run_offline_experiment.py --out experiments/offline_dev.json
python scripts/run_offline_experiment.py --predictions experiments/runs/dev/predictions.jsonl \
    --out experiments/offline_cached.json --markdown experiments/vault
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sekf.evaluation.cached import load_predictions
from sekf.evaluation.cv import default_grid, project_cached, summarize_cached
from sekf.evaluation.runner import load_dev_cases, run_dev_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score dev cases offline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.limit < 0:
        print("limit must be >= 0", file=sys.stderr)
        return 2
    try:
        if args.predictions is not None:
            cases = load_predictions(args.predictions)
            if args.limit:
                cases = cases[: args.limit]
            summary = summarize_cached(cases, default_grid())
            if args.markdown is not None:
                project_cached(cases, args.markdown)
                summary["markdown_dir"] = str(args.markdown)
        else:
            loaded = load_dev_cases()
            if args.limit:
                loaded = loaded[: args.limit]
            summary = asyncio.run(run_dev_cases(loaded))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"cases {summary['n_cases']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
