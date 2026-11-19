"""Capture raw Historian, Judge, and Synthesizer JSON once.

Offline by default (fake Judge, no network):

    python scripts/capture_live_predictions.py --out experiments/runs/dev/predictions.jsonl

One live pass, with the same HTTP 429 retry as the dev pilot:

    python scripts/capture_live_predictions.py --live --out experiments/runs/live/predictions.jsonl

The file is the input for run_offline_experiment.py, sweep_policy_thresholds.py,
and analyze_judge_evidence.py. This script does not score or sweep.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from sekf.evaluation.capture import capture_cases, offline_roles
from sekf.evaluation.runner import load_dev_cases
from sekf.historian import LLMHistorian
from sekf.judge import LLMJudge
from sekf.judge.compose import load_project_env
from sekf.providers import create_provider_from_env
from sekf.synthesizer import LLMSynthesizer


def _live_ready() -> bool:
    load_project_env()
    provider = (os.getenv("SEKF_PROVIDER") or "").strip().lower()
    key = (os.getenv("SEKF_LLM_API_KEY") or "").strip()
    return provider in {"openai", "openai_compatible"} and bool(key)


async def _run(cases: list[dict], dest: Path, *, live: bool) -> int:
    provider = None
    if live:
        provider = create_provider_from_env()
        historian = LLMHistorian(provider)
        judge = LLMJudge(provider)
        synthesizer = LLMSynthesizer(provider)
    else:
        historian, judge, synthesizer = offline_roles()
    try:
        return await capture_cases(
            cases, dest, historian=historian, judge=judge, synthesizer=synthesizer
        )
    finally:
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write experiments/runs/<id>/predictions.jsonl.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        print("limit must be >= 0", file=sys.stderr)
        return 2
    if args.live and not _live_ready():
        print(
            "live capture needs SEKF_PROVIDER=openai and SEKF_LLM_API_KEY; no substitute model",
            file=sys.stderr,
        )
        return 2
    cases = load_dev_cases()
    if args.limit:
        cases = cases[: args.limit]
    count = asyncio.run(_run(cases, args.out, live=args.live))
    print(f"wrote {count} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
