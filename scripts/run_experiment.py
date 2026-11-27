"""Run or resume one experiment with a manifest and a resumable results file.

Offline, rule-based Judge (no key needed):

    python scripts/run_experiment.py --dataset datasets/dev --run-dir runs/dev-fixture

Live model (needs SEKF_PROVIDER=openai, SEKF_LLM_API_KEY, SEKF_LLM_MODEL):

    python scripts/run_experiment.py --dataset datasets/dev --run-dir runs/dev-llm --judge env

Running the same command again resumes. A held-out split needs --confirm-held-out and a clean
git tree. See `sekf.evaluation.experiment` for the details.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sekf.evaluation.experiment import HeldOutRefused, RunMismatch, run_experiment
from sekf.evaluation.runner import ALL_SYSTEMS, JUDGE_CHOICES


def _table(summary: dict[str, object]) -> str:
    lines = [f"{'system':<26} {'macro_f1':>9} {'95% CI':>15} {'false_promo':>12} {'calls':>6}"]
    for item in summary["systems"]:  # type: ignore[attr-defined]
        if item.get("status") == "not_run":
            lines.append(f"{item['system']:<26} not run: {item['reason']}")
            continue
        low, high = item["macro_f1_ci95"]
        lines.append(
            f"{item['system']:<26} {item['macro_f1']:>9.3f} {f'[{low:.2f}, {high:.2f}]':>15} "
            f"{f'{item["false_promotion_hits"]}/{item["false_promotion_support"]}':>12} "
            f"{item['llm_calls']:>6}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", choices=JUDGE_CHOICES, default="fixture")
    parser.add_argument("--systems", nargs="+", choices=ALL_SYSTEMS, default=list(ALL_SYSTEMS))
    parser.add_argument("--limit", type=int, default=0, help="first N cases only (0 = all)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--confirm-held-out", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(
            run_experiment(
                dataset_dir=args.dataset,
                run_dir=args.run_dir,
                judge=args.judge,
                systems=args.systems,
                limit=args.limit,
                concurrency=args.concurrency,
                confirm_held_out=args.confirm_held_out,
                notes=args.notes,
            )
        )
    except (HeldOutRefused, RunMismatch, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    summary = result.summary
    print(_table(summary))
    print(
        f"\nrun {summary['run_id']}  judge={summary['judge']}  cases={summary['n_cases']}  "
        f"this call: {result.counts}  incomplete={summary['incomplete_pairs']}"
    )
    print(f"wrote {result.run_dir}/manifest.json, results.jsonl, summary.json")
    return 1 if summary["incomplete_pairs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
