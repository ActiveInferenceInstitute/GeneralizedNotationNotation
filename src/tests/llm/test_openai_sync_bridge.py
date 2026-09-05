"""Synchronous analysis bridges active loops without leaking or retrying calls."""

import asyncio
import gc
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from llm.providers.openai_provider import OpenAIProvider


@pytest.mark.parametrize("inside_loop", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_analysis_bridge_calls_provider_once(
    monkeypatch: pytest.MonkeyPatch, inside_loop: bool, fail: bool
) -> None:
    provider = OpenAIProvider(api_key="test-key")
    calls = []

    async def generate(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if fail:
            raise RuntimeError("provider rejected request")
        return SimpleNamespace(content="result")

    monkeypatch.setattr(provider, "generate_response", generate)

    async def from_loop() -> str:
        return provider.analyze("model", "test")

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", RuntimeWarning)
        if fail:
            with pytest.raises(RuntimeError, match="provider rejected request"):
                asyncio.run(from_loop()) if inside_loop else provider.analyze(
                    "model", "test"
                )
        else:
            result = (
                asyncio.run(from_loop())
                if inside_loop
                else provider.analyze("model", "test")
            )
            assert result == "result"
        gc.collect()
    assert calls == [1]
    assert not [w for w in observed if "never awaited" in str(w.message)]
