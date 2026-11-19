"""Retry a live call when the provider reports HTTP 429. No other errors are retried.

The provider itself raises `ProviderUnavailable` and does not retry. This helper is the
only retry, shared by the dev pilot and the prediction capture script.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sekf.errors import ProviderUnavailable


async def with_retries[T](
    op: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    label: str = "",
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Call `op` again only when the failure is an HTTP 429 rate limit."""

    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    sleeper = sleep or asyncio.sleep
    delay = 5.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = await op()
        except ProviderUnavailable as exc:
            last_error = exc
            if "rate limited" not in str(exc).lower() or attempt == attempts:
                raise
            print(f"{label} 429, retry {attempt}/{attempts} in {delay:.0f}s", flush=True)
            await sleeper(delay)
            delay = min(delay * 2, 60.0)
            continue
        if isinstance(result, dict) and result.get("error") == "ProviderUnavailable":
            if attempt == attempts:
                return result
            print(
                f"{label} provider unavailable, retry {attempt}/{attempts} in {delay:.0f}s",
                flush=True,
            )
            await sleeper(delay)
            delay = min(delay * 2, 60.0)
            continue
        return result
    if last_error is not None:
        raise last_error
    raise ProviderUnavailable(f"{label} failed after retries")
