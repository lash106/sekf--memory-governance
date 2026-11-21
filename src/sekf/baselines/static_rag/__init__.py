"""Static-RAG baseline package (SPEC.md name: `static_rag`)."""

from sekf.baselines.static_rag.baseline import (
    DEFAULT_TOP_K,
    POLICY_VERSION,
    Retrieved,
    StaticRagBaseline,
    retrieve,
)

__all__ = ["DEFAULT_TOP_K", "POLICY_VERSION", "Retrieved", "StaticRagBaseline", "retrieve"]
