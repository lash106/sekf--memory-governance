"""Leave-one-author-out threshold sweep on cached role outputs. No extra LLM calls.

Each input line is a predictions object (`context`, `gold_action`, and `author` or
`source_path`). Thresholds are chosen on the other authors and scored on the held-out
author. The script does not print a best score fit on the same rows it reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sekf.evaluation.cached import load_predictions
from sekf.evaluation.cv import default_grid, leave_one_author_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leave-one-author-out policy threshold sweep.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args(argv)
    try:
        report = leave_one_author_out(load_predictions(args.predictions), default_grid())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
