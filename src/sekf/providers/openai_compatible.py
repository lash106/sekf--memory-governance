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

from sekf.errors import CallUsage, ProviderUnavailable, RoleOutputInvalid
from sekf.providers.base import LLMResult

MAX_REPAIR_ATTEMPTS = 2  # after the first call => at most 3 total LLM calls
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Versioned public list prices (USD per 1M tokens). Unknown models are not priced as mini.
COST_RATE_VERSION = "openai-public-2026-10"
MODEL_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}


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
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
        cost_rate_version: str = COST_RATE_VERSION,
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
        if (input_cost_per_mtok is None) != (output_cost_per_mtok is None):
            raise ValueError("set both input_cost_per_mtok and output_cost_per_mtok, or neither")
        self._input_cost_per_mtok = input_cost_per_mtok
        self._output_cost_per_mtok = output_cost_per_mtok
        self.cost_rate_version = cost_rate_version

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        """Build from `SEKF_LLM_*` environment variables.

        `SEKF_LLM_INPUT_COST_PER_MTOK` and `SEKF_LLM_OUTPUT_COST_PER_MTOK` override the
        versioned table for every model in this process. Omit both to use `MODEL_USD_PER_MTOK`.
        """

        raw_in = os.getenv("SEKF_LLM_INPUT_COST_PER_MTOK")
        raw_out = os.getenv("SEKF_LLM_OUTPUT_COST_PER_MTOK")
        if (raw_in is None) != (raw_out is None):
            raise ProviderUnavailable(
                "set both SEKF_LLM_INPUT_COST_PER_MTOK and "
                "SEKF_LLM_OUTPUT_COST_PER_MTOK, or neither"
            )
        if raw_in is None or raw_out is None:
            return cls()
        return cls(input_cost_per_mtok=float(raw_in), output_cost_per_mtok=float(raw_out))

    def rates_for(self, model_id: str) -> tuple[float, float]:
        """USD per 1M tokens for `model_id`. Unknown models cost 0, not the mini rate."""

        if self._input_cost_per_mtok is not None and self._output_cost_per_mtok is not None:
            return self._input_cost_per_mtok, self._output_cost_per_mtok
        if model_id in MODEL_USD_PER_MTOK:
            return MODEL_USD_PER_MTOK[model_id]
        for known, rates in MODEL_USD_PER_MTOK.items():
            if model_id.startswith(known + "-"):
                return rates
        return 0.0, 0.0

    def _usage(
        self, *, attempts: int, total_in: int, total_out: int, latency_ms: int, model_id: str
    ) -> CallUsage:
        input_rate, output_rate = self.rates_for(model_id)
        cost = (total_in / 1_000_000.0) * input_rate + (total_out / 1_000_000.0) * output_rate
        return CallUsage(
            llm_calls=attempts,
            input_tokens=total_in,
            output_tokens=total_out,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )

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
                raw_text, token_usage = await self._chat_completion(
                    messages=working,
                    model=model_id,
                    temperature=temperature,
                    seed=seed,
                )
            except ProviderUnavailable as exc:
                exc.usage = self._usage(
                    attempts=attempts,
                    total_in=total_in,
                    total_out=total_out,
                    latency_ms=total_latency,
                    model_id=model_id,
                )
                raise
            except Exception as exc:  # network / protocol
                raise ProviderUnavailable(
                    f"LLM request failed: {exc}",
                    usage=self._usage(
                        attempts=attempts,
                        total_in=total_in,
                        total_out=total_out,
                        latency_ms=total_latency,
                        model_id=model_id,
                    ),
                ) from exc
            total_latency += int((time.perf_counter() - started) * 1000)
            total_in += int(token_usage.get("prompt_tokens", 0))
            total_out += int(token_usage.get("completion_tokens", 0))

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

            call_usage = self._usage(
                attempts=attempts,
                total_in=total_in,
                total_out=total_out,
                latency_ms=total_latency,
                model_id=model_id,
            )
            return LLMResult(
                parsed=parsed,
                model=model_id,
                provider="openai_compatible",
                llm_calls=call_usage.llm_calls,
                input_tokens=call_usage.input_tokens,
                output_tokens=call_usage.output_tokens,
                latency_ms=call_usage.latency_ms,
                estimated_cost_usd=call_usage.estimated_cost_usd,
            )

        call_usage = self._usage(
            attempts=attempts,
            total_in=total_in,
            total_out=total_out,
            latency_ms=total_latency,
            model_id=model_id,
        )
        raise RoleOutputInvalid(
            f"structured output invalid after {attempts} attempt(s): {last_error}",
            usage=call_usage,
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
