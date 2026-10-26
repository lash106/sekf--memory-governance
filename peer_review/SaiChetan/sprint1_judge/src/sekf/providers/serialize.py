"""Shared JSON helpers for role and baseline LLM payloads.

Owner: Sai Chetan. Kept provider-adjacent so Judge, baselines, and a future OpenAI
adapter can reuse one serialization policy without coupling to contracts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from pydantic import BaseModel


def dump_contract_json(model: BaseModel) -> str:
    """Canonical JSON for one Pydantic model (sorted keys, compact separators)."""

    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def dump_contract_list_json(models: Iterable[BaseModel]) -> str:
    """Canonical JSON array for a sequence of Pydantic models."""

    return "[" + ",".join(dump_contract_json(model) for model in models) + "]"
