"""Configurable defaults for the LLM Judge.

Override via `LLMJudge(..., prompt_role=..., prompt_version=...)` when experimenting
with newer prompt files under `prompts/judge/`.
"""

DEFAULT_PROMPT_ROLE = "judge"
DEFAULT_PROMPT_VERSION = "v1"
SYSTEM_MESSAGE = "You are the SEKF Judge. Reply with valid JSON only."
