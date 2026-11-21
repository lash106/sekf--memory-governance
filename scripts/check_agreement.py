"""Check how well two annotators agree before a test split is frozen.

    .venv/bin/python scripts/check_agreement.py A_DIR B_DIR [--min-kappa 0.6]

Each folder holds that annotator's copy of the cases (same case_id, independently filled-in
`gold` labels, identical inputs). Prints percent agreement and Cohen's kappa per gold field and
lists every disagreement. Exits 1 if a case is missing, inputs differ, or any kappa is below
`--min-kappa` (0.6 is conventionally "substantial" agreement). Resolve each disagreement by
discussion, and fix `datasets/ANNOTATION_GUIDE.md` if the rule was unclear, then re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sekf.evaluation.agreement import compare, load_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a_dir", type=Path, help="annotator A's cases")
    parser.add_argument("b_dir", type=Path, help="annotator B's cases")
    parser.add_argument("--min-kappa", type=float, default=0.6)
    args = parser.parse_args(argv)

    report = compare(load_cases(args.a_dir), load_cases(args.b_dir))
    print(f"compared {report.compared} case(s)")
    print(f"{'field':20} {'n':>4} {'agree%':>8} {'kappa':>7}")
    for f in report.fields:
        print(f"{f.name:20} {f.n:>4} {100 * f.percent:>7.1f}% {f.kappa:>7.2f}")
    for f in report.fields:
        for case_id, x, y in f.disagreements:
            print(f"  disagree  {f.name:18} {case_id}: A={x}  B={y}")
    problems = report.problems(args.min_kappa)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if not report.fields:
        print("PROBLEM: no comparable cases")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
