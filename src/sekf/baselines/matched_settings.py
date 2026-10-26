"""Matched-settings checks for fair baseline vs council comparison (SPEC 12.3).

Owner: Sai Chetan. Deterministic; no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from sekf.contracts import RunConfig, Thresholds


@dataclass(frozen=True)
class MatchedSettingsViolation:
    """One mismatched field between council and baseline run configuration."""

    field: str
    council_value: object
    baseline_value: object


def compare_run_configs(
    council: RunConfig, baseline: RunConfig, *, require_same_seed: bool = True
) -> list[MatchedSettingsViolation]:
    """Return mismatches that would unfairly advantage one system."""

    violations: list[MatchedSettingsViolation] = []
    if council.model != baseline.model:
        violations.append(MatchedSettingsViolation("model", council.model, baseline.model))
    if council.temperature != baseline.temperature:
        violations.append(
            MatchedSettingsViolation("temperature", council.temperature, baseline.temperature)
        )
    if require_same_seed and council.seed != baseline.seed:
        violations.append(MatchedSettingsViolation("seed", council.seed, baseline.seed))
    return violations


def compare_thresholds(council: Thresholds, baseline: Thresholds) -> list[MatchedSettingsViolation]:
    """Thresholds must match when both systems use the same policy thresholds object."""

    violations: list[MatchedSettingsViolation] = []
    for name in (
        "support_threshold",
        "refute_threshold",
        "historian_confidence_threshold",
        "source_reliability_threshold",
        "synthesizer_veto_threshold",
    ):
        left = getattr(council, name)
        right = getattr(baseline, name)
        if left != right:
            violations.append(MatchedSettingsViolation(name, left, right))
    return violations


def assert_matched_settings(
    council: RunConfig,
    baseline: RunConfig,
    *,
    council_thresholds: Thresholds | None = None,
    baseline_thresholds: Thresholds | None = None,
    require_same_seed: bool = True,
) -> None:
    """Raise ValueError listing mismatches; used by experiment runners before a pilot cell."""

    violations = compare_run_configs(council, baseline, require_same_seed=require_same_seed)
    if council_thresholds is not None and baseline_thresholds is not None:
        violations.extend(compare_thresholds(council_thresholds, baseline_thresholds))
    if violations:
        details = ", ".join(
            f"{v.field}: council={v.council_value!r} baseline={v.baseline_value!r}"
            for v in violations
        )
        raise ValueError(f"matched-settings violation: {details}")
