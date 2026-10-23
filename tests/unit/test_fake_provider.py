import pytest
from pydantic import BaseModel

from sekf.errors import RoleOutputInvalid
from sekf.providers import FakeProvider


class Answer(BaseModel):
    value: int


async def call(provider: FakeProvider) -> BaseModel:
    result = await provider.complete_json(
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        model="fake",
        temperature=0.0,
        seed=1,
    )
    return result.parsed


async def test_returns_validated_model() -> None:
    provider = FakeProvider(lambda _m, _r: {"value": 3})
    assert await call(provider) == Answer(value=3)
    assert len(provider.calls) == 1


async def test_invalid_output_raises_typed_error() -> None:
    provider = FakeProvider(lambda _m, _r: {"value": "not-an-int"})
    with pytest.raises(RoleOutputInvalid):
        await call(provider)
