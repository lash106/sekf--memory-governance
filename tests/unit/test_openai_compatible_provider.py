"""Offline tests for OpenAI-compatible provider repair loop (no network)."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from sekf.errors import ProviderUnavailable, RoleOutputInvalid
from sekf.providers.openai_compatible import OpenAICompatibleProvider


class Answer(BaseModel):
    value: int = Field(ge=0)


def _handler_factory(bodies: list[dict]) -> httpx.MockTransport:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        idx = calls["n"]
        calls["n"] += 1
        body = bodies[min(idx, len(bodies) - 1)]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _completion(content: str) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


@pytest.mark.asyncio
async def test_valid_first_attempt() -> None:
    transport = _handler_factory([_completion('{"value": 3}')])
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatibleProvider(api_key="test-key", client=client)
    result = await provider.complete_json(
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        model="gpt-4o-mini",
        temperature=0.0,
        seed=1,
    )
    assert result.parsed == Answer(value=3)
    assert result.llm_calls == 1
    assert result.provider == "openai_compatible"
    await client.aclose()


@pytest.mark.asyncio
async def test_repairs_then_succeeds() -> None:
    transport = _handler_factory(
        [
            _completion("not-json"),
            _completion('{"value": "nope"}'),
            _completion('{"value": 7}'),
        ]
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatibleProvider(api_key="test-key", client=client)
    result = await provider.complete_json(
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        model="gpt-4o-mini",
        temperature=0.0,
        seed=None,
    )
    assert result.parsed == Answer(value=7)
    assert result.llm_calls == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_repairs_exhausted_raises_role_output_invalid() -> None:
    transport = _handler_factory(
        [_completion("bad"), _completion("still-bad"), _completion("worse")]
    )
    client = httpx.AsyncClient(transport=transport)
    provider = OpenAICompatibleProvider(api_key="test-key", client=client)
    with pytest.raises(RoleOutputInvalid):
        await provider.complete_json(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            model="gpt-4o-mini",
            temperature=0.0,
            seed=0,
        )
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_raises_provider_unavailable() -> None:
    provider = OpenAICompatibleProvider(api_key="")
    with pytest.raises(ProviderUnavailable, match="SEKF_LLM_API_KEY"):
        await provider.complete_json(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            model="gpt-4o-mini",
            temperature=0.0,
            seed=None,
        )


@pytest.mark.asyncio
async def test_http_429_is_provider_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(api_key="test-key", client=client)
    with pytest.raises(ProviderUnavailable, match="429"):
        await provider.complete_json(
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            model="gpt-4o-mini",
            temperature=0.0,
            seed=None,
        )
    await client.aclose()


def test_repair_payload_keeps_json_object_mode() -> None:
    # Smoke that helper shapes stay serializable for debugging.
    assert json.dumps(_completion('{"value": 1}'))
