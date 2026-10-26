"""Single-judge baseline package (SPEC.md name: `single_judge`)."""

from sekf.baselines.single_judge.baseline import SingleJudgeBaseline
from sekf.baselines.single_judge.constants import DEFAULT_POLICY_VERSION
from sekf.baselines.single_judge.schemas import SingleJudgeLLMOutput

# Public alias kept for callers and tests that expect POLICY_VERSION.
POLICY_VERSION = DEFAULT_POLICY_VERSION

__all__ = [
    "DEFAULT_POLICY_VERSION",
    "POLICY_VERSION",
    "SingleJudgeBaseline",
    "SingleJudgeLLMOutput",
]
