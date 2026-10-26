"""Configurable defaults for the single-judge baseline.

Override via `SingleJudgeBaseline(..., prompt_version=..., policy_version=...)` when
adding `prompts/baselines/single_judge/v2.txt` or a later baseline policy id.
"""

DEFAULT_BASELINE_NAME = "single_judge"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_POLICY_VERSION = "single-judge-v1"
SYSTEM_MESSAGE = "You are the SEKF single-judge baseline. Reply with valid JSON only."
R1_EXPLANATION = "A role failed or returned invalid output; escalating for human review."
