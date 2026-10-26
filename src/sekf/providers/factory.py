"""Provider factory from environment (SPEC decision D1). Owner: Sai Chetan.

Never silently falls back across providers: an unknown `SEKF_PROVIDER` is an error.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from sekf.errors import ProviderUnavailable
from sekf.providers.base import StructuredLLMProvider
from sekf.providers.fake import FakeProvider
from sekf.providers.openai_compatible import OpenAICompatibleProvider


def create_provider_from_env(
    *,
    fake_responder: Callable[[list[dict[str, str]], type[BaseModel]], dict[str, Any] | BaseModel]
    | None = None,
) -> StructuredLLMProvider:
    """Return the configured provider.

    - `SEKF_PROVIDER=fake` (default): requires `fake_responder` for useful output.
    - `SEKF_PROVIDER=openai`: OpenAI-compatible live client.
    """

    name = (os.getenv("SEKF_PROVIDER") or "fake").strip().lower()
    if name == "fake":
        if fake_responder is None:

            def _empty(_messages: list[dict[str, str]], _model: type[BaseModel]) -> dict[str, Any]:
                return {}

            fake_responder = _empty
        return FakeProvider(fake_responder)
    if name in {"openai", "openai_compatible"}:
        return OpenAICompatibleProvider.from_env()
    raise ProviderUnavailable(
        f"unknown SEKF_PROVIDER={name!r}; expected 'fake' or 'openai' (no silent fallback)"
    )
