"""LLM-backed Judge package. Swap prompt version or schema without touching `FakeJudge`."""

from sekf.judge.llm.judge import LLMJudge
from sekf.judge.llm.schemas import JudgeLLMOutput

__all__ = ["JudgeLLMOutput", "LLMJudge"]
