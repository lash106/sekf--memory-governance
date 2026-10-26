"""OpenAI-compatible structured JSON provider (SPEC.md section 11). Owner: Sai Chetan.

Uses httpx only (no vendor SDK). Offline tests must mock transport or use FakeProvider.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from sekf.errors import ProviderUnavailable, RoleOutputInvalid
from sekf.providers.base import LLMResult

MAX_REPAIR_ATTEMPTS = 2  # after the first call => at most 3 total LLM calls
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Rough USD / 1M tokens for gpt-4o-mini-class models; recorded for audit, not billing.
_DEFAULT_INPUT_COST_PER_MTOK = 0.15
_DEFAULT_OUTPUT_COST_PER_MTOK = 0.60


class OpenAICompatibleProvider:
    """Chat Completions provider with JSON validation and ≤2 repair attempts."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_s: float = 60.0,
        max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
        client: httpx.AsyncClient | None = None,
        input_cost_per_mtok: float = _DEFAULT_INPUT_COST_PER_MTOK,
        output_cost_per_mtok: float = _DEFAULT_OUTPUT_COST_PER_MTOK,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("SEKF_LLM_API_KEY", "")
        self._base_url = (base_url or os.getenv("SEKF_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self._default_model = default_model or os.getenv("SEKF_LLM_MODEL") or DEFAULT_MODEL
        self._timeout_s = timeout_s
        self._max_repair_attempts = max_repair_attempts
        self._client = client
        self._owns_client = client is None
        self._input_cost_per_mtok = input_cost_per_mtok
        self._output_cost_per_mtok = output_cost_per_mtok

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        """Build from `SEKF_LLM_*` environment variables."""

        return cls()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResult:
        if not self._api_key:
            raise ProviderUnavailable(
                "SEKF_LLM_API_KEY is not set; refuse to call a live provider silently"
            )

        model_id = model if model and model != "fake/none" else self._default_model
        working = list(messages)
        total_in = 0
        total_out = 0
        total_latency = 0
        attempts = 0
        last_error: Exception | None = None
        max_attempts = 1 + self._max_repair_attempts

        while attempts < max_attempts:
            attempts += 1
            started = time.perf_counter()
            try:
                raw_text, usage = await self._chat_completion(
                    messages=working,
                    model=model_id,
                    temperature=temperature,
                    seed=seed,
                )
            except ProviderUnavailable:
                raise
            except Exception as exc:  # network / protocol
                raise ProviderUnavailable(f"LLM request failed: {exc}") from exc
            total_latency += int((time.perf_counter() - started) * 1000)
            total_in += int(usage.get("prompt_tokens", 0))
            total_out += int(usage.get("completion_tokens", 0))

            try:
                data = json.loads(raw_text)
                parsed = response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
                working = [
                    *working,
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for the required schema. "
                            f"Validation error: {exc}. Reply again with ONLY corrected JSON."
                        ),
                    },
                ]
                continue

            cost = (total_in / 1_000_000.0) * self._input_cost_per_mtok + (
                total_out / 1_000_000.0
            ) * self._output_cost_per_mtok
            return LLMResult(
                parsed=parsed,
                model=model_id,
                provider="openai_compatible",
                llm_calls=attempts,
                input_tokens=total_in,
                output_tokens=total_out,
                latency_ms=total_latency,
                estimated_cost_usd=cost,
            )

        raise RoleOutputInvalid(
            f"structured output invalid after {attempts} attempt(s): {last_error}"
        )

    async def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> tuple[str, dict[str, Any]]:
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"HTTP error talking to LLM: {exc}") from exc

        if response.status_code == 429:
            raise ProviderUnavailable("LLM rate limited (HTTP 429)")
        if response.status_code in {401, 403}:
            raise ProviderUnavailable(f"LLM auth failed (HTTP {response.status_code})")
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"LLM error HTTP {response.status_code}: {response.text[:300]}"
            )

        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(f"unexpected LLM response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderUnavailable("LLM returned empty content")
        usage = body.get("usage") or {}
        return content, usage if isinstance(usage, dict) else {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client
