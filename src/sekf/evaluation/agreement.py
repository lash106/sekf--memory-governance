"""Inter-annotator agreement for the double-labeled cases (SPEC.md section 13.4). Owner: Shailen.

Each frozen `test` case is labeled independently by two people. Agreement is measured with
Cohen's kappa (chance-corrected) alongside plain percent agreement, per gold field, so a field
the annotators disagree about points at an unclear rule in `datasets/ANNOTATION_GUIDE.md`.

The two annotators must have labeled IDENTICAL inputs (candidate, evidence, initial memory).
A case whose inputs differ between the two copies is reported, never scored.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLD_FIELDS = ("judge_label", "temporal_relation", "conflict_type", "final_action")
INPUT_FIELDS = ("candidate", "initial_memory", "evidence")


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b):
        raise ValueError("label lists must have the same length")
    if not a:
        raise ValueError("no labels to compare")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items.

    1.0 is perfect agreement, 0.0 is no better than chance, below 0 is worse than chance. When
    both raters always give one identical label, chance agreement is 1 and kappa is undefined;
    this returns 1.0 if they agree on every item and 0.0 otherwise.
    """
    observed = percent_agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in set(ca) | set(cb))
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass
class FieldAgreement:
    name: str
    n: int
    percent: float
    kappa: float
    disagreements: list[tuple[str, str, str]] = field(default_factory=list)  # (case, a, b)


@dataclass
class AgreementReport:
    fields: list[FieldAgreement]
    compared: int
    only_in_a: list[str]
    only_in_b: list[str]
    input_mismatches: list[str]

    def worst_kappa(self) -> float:
        return min((f.kappa for f in self.fields), default=1.0)

    def problems(self, min_kappa: float) -> list[str]:
        """Everything that should stop the split from being frozen."""
        out = [f"case only labeled by A: {c}" for c in self.only_in_a]
        out += [f"case only labeled by B: {c}" for c in self.only_in_b]
        out += [f"inputs differ between A and B: {c}" for c in self.input_mismatches]
        out += [
            f"{f.name}: kappa {f.kappa:.2f} < {min_kappa:.2f} ({len(f.disagreements)} disagree)"
            for f in self.fields
            if f.kappa < min_kappa
        ]
        return out


def load_cases(folder: Path) -> dict[str, dict[str, Any]]:
    """`{case_id: case}` for every case JSON under `folder` (searched recursively)."""
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.rglob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(case, dict) and "gold" in case and "case_id" in case:
            if case["case_id"] in cases:
                raise ValueError(f"duplicate case_id {case['case_id']} under {folder}")
            cases[case["case_id"]] = case
    return cases


def compare(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    fields: Sequence[str] = GOLD_FIELDS,
) -> AgreementReport:
    shared = sorted(set(a) & set(b))
    mismatched = [c for c in shared if any(a[c].get(k) != b[c].get(k) for k in INPUT_FIELDS)]
    usable = [c for c in shared if c not in set(mismatched)]
    results: list[FieldAgreement] = []
    if usable:
        for name in fields:
            la = [str(a[c]["gold"][name]) for c in usable]
            lb = [str(b[c]["gold"][name]) for c in usable]
            results.append(
                FieldAgreement(
                    name=name,
                    n=len(usable),
                    percent=percent_agreement(la, lb),
                    kappa=cohens_kappa(la, lb),
                    disagreements=[
                        (c, x, y) for c, x, y in zip(usable, la, lb, strict=True) if x != y
                    ],
                )
            )
    return AgreementReport(
        fields=results,
        compared=len(usable),
        only_in_a=sorted(set(a) - set(b)),
        only_in_b=sorted(set(b) - set(a)),
        input_mismatches=mismatched,
    )
