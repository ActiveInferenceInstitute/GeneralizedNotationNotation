"""Offline sync analysis must neither leak coroutines nor retry provider errors."""

from __future__ import annotations

import asyncio
import gc
import warnings
from typing import NoReturn

import pytest

from gnn.llm.providers.openai_provider import OpenAIProvider


def test_running_event_loop_does_not_create_unawaited_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider(api_key="offline-test")
    calls: list[object] = []

    async def response(messages: object) -> str:
        calls.append(messages)
        return "analysis"

    monkeypatch.setattr(provider, "generate_response", response)

    async def caller() -> str:
        return provider.analyze("model", "structure")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        assert asyncio.run(caller()) == "analysis"
        gc.collect()
    assert not [
        warning for warning in captured if "never awaited" in str(warning.message)
    ]
    assert len(calls) == 1


@pytest.mark.parametrize("active_loop", [False, True])
def test_provider_runtime_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, active_loop: bool
) -> None:
    provider = OpenAIProvider(api_key="offline-test")
    calls: list[object] = []

    async def response(messages: object) -> NoReturn:
        calls.append(messages)
        raise RuntimeError("provider failed")

    monkeypatch.setattr(provider, "generate_response", response)

    async def caller() -> str:
        return provider.analyze("model", "structure")

    with pytest.raises(RuntimeError, match="provider failed"):
        if active_loop:
            asyncio.run(caller())
        else:
            provider.analyze("model", "structure")
    assert len(calls) == 1
