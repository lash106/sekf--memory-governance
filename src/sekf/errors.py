"""Typed errors. Failures must be explicit; none of these may be swallowed silently."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallUsage:
    """Provider usage that must survive a failed call (repairs included)."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0


class SekfError(Exception):
    """Base class for all SEKF errors."""

    def __init__(self, message: str, *, usage: CallUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


class ConfigurationError(SekfError):
    """An environment setting names something that does not exist. There is no silent fallback."""


class ProviderUnavailable(SekfError):
    """The LLM provider could not be reached or is not configured."""


class CommitRejected(SekfError):
    """A commit bundle was inconsistent and nothing was written."""


class RoleTimeout(SekfError):
    """A role (Historian, Judge, Synthesizer) exceeded its time limit."""


class RoleOutputInvalid(SekfError):
    """A role produced output that failed schema validation after repair attempts."""
