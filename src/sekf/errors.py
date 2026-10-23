"""Typed errors. Failures must be explicit; none of these may be swallowed silently."""


class SekfError(Exception):
    """Base class for all SEKF errors."""


class ProviderUnavailable(SekfError):
    """The LLM provider could not be reached or is not configured."""


class CommitRejected(SekfError):
    """A commit bundle was inconsistent and nothing was written."""


class RoleTimeout(SekfError):
    """A role (Historian, Judge, Synthesizer) exceeded its time limit."""


class RoleOutputInvalid(SekfError):
    """A role produced output that failed schema validation after repair attempts."""
