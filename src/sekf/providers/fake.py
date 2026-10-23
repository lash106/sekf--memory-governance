"""Deterministic fake provider: no network, no credentials. Used by tests and CI."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from sekf.errors import RoleOutputInvalid
from sekf.providers.base import LLMResult

Responder = Callable[[list[dict[str, str]], type[BaseModel]], dict[str, Any] | BaseModel]


class FakeProvider:
    """Returns whatever `responder` produces, validated against `response_model`.

    Raises `RoleOutputInvalid` if the canned output does not validate, which mirrors how a
    real provider behaves once repair attempts are exhausted.
    """

    def __init__(self, responder: Responder, *, model_name: str = "fake/deterministic") -> None:
        self._responder = responder
        self._model_name = model_name
        self.calls: list[list[dict[str, str]]] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        model: str,
        temperature: float,
        seed: int | None,
    ) -> LLMResult:
        self.calls.append(messages)
        raw = self._responder(messages, response_model)
        try:
            if isinstance(raw, BaseModel):
                parsed = response_model.model_validate(raw.model_dump())
            else:
                parsed = response_model.model_validate(raw)
        except ValidationError as exc:
            raise RoleOutputInvalid(f"fake output failed validation: {exc}") from exc
        return LLMResult(parsed=parsed, model=self._model_name, provider="fake", llm_calls=0)
