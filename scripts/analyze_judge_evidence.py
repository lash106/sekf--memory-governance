"""Offline Judge/evidence error report from a JSONL file of observations.

Owner: Sai Chetan. No network and no LLM. Each line is one case:

    {"case_id": "...", "category": "supported_new_claim",
     "gold_label": "SUPPORTED", "predicted_label": "REFUTED", "confidence": 0.91,
     "supporting_evidence_ids": [], "refuting_evidence_ids": ["..."],
     "provided_evidence_ids": ["..."]}

The same reader accepts `predictions.jsonl` from `scripts/capture_live_predictions.py`.
The report is JSON on stdout. Policy, storage, retrieval, and extraction errors
are out of scope; pass only Judge outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sekf.judge.error_analysis import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    JudgeObservation,
    analyze_judge_evidence,
    observation_from_mapping,
    report_to_dict,
)


def load_observations(path: Path) -> list[JudgeObservation]:
    """Read non-empty JSONL lines. Invalid lines raise ValueError."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    observations: list[JudgeObservation] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_no}: observation must be an object")
        observations.append(_observation(payload, line_no))
    return observations


def _observation(payload: dict[str, Any], line_no: int) -> JudgeObservation:
    """Accept a Judge observation or one line of predictions.jsonl."""

    if "gold_label" in payload and "predicted_label" in payload:
        return observation_from_mapping(payload, line_no=line_no)
    context = payload.get("context")
    judge = context.get("judge") if isinstance(context, dict) else None
    if not isinstance(judge, dict):
        raise ValueError(f"line {line_no}: prediction has no judge output")
    evidence = context.get("evidence") if isinstance(context, dict) else None
    provided = []
    if isinstance(evidence, list):
        provided = [str(item.get("evidence_id")) for item in evidence if isinstance(item, dict)]
    mapped = {
        "case_id": payload.get("case_id"),
        "category": payload.get("category") or "uncategorized",
        "gold_label": payload.get("gold_judge", payload.get("gold_label")),
        "predicted_label": judge.get("label"),
        "confidence": judge.get("confidence"),
        "supporting_evidence_ids": judge.get("supporting_evidence_ids") or [],
        "refuting_evidence_ids": judge.get("refuting_evidence_ids") or [],
        "provided_evidence_ids": provided,
    }
    return observation_from_mapping(mapped, line_no=line_no)


def main(argv: list[str] | None = None) -> int:
    """Write one JSON report to stdout. Return 2 when the input cannot be scored."""

    parser = argparse.ArgumentParser(description="Score Judge labels and evidence citations.")
    parser.add_argument("observations", type=Path, help="JSONL file of Judge observations")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        observations = load_observations(args.observations)
        report = analyze_judge_evidence(
            observations, confidence_threshold=args.threshold, n_bins=args.bins
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    json.dump(report_to_dict(report), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
