#!/usr/bin/env python3
"""Unit tests for LLM pipeline model wiring (Step 14): Ollama tag + SUMMARY provider order."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
def test_summary_task_prefers_ollama_when_registered() -> None:
    from llm.llm_processor import LLMProcessor
    from llm.providers.base_provider import ProviderType

    proc = LLMProcessor(
        preferred_providers=[ProviderType.OLLAMA, ProviderType.OPENAI]
    )
    mock_ollama = MagicMock()
    mock_ollama.provider_type = ProviderType.OLLAMA
    mock_openai = MagicMock()
    mock_openai.provider_type = ProviderType.OPENAI
    proc.providers = {
        ProviderType.OLLAMA: mock_ollama,
        ProviderType.OPENAI: mock_openai,
    }

    from llm.llm_processor import AnalysisType

    best = proc.get_best_provider_for_task(AnalysisType.SUMMARY)
    assert best is mock_ollama


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_summarize_with_ollama_model_uses_ollama_provider_and_config() -> None:
    from llm.llm_operations import LLMOperations
    from llm.providers.base_provider import LLMConfig, LLMResponse, ProviderType

    mock_response = LLMResponse(
        content="summary text",
        model_used="smollm2:135m-instruct-q4_K_S",
        provider="ollama",
    )
    mock_analyze = AsyncMock(return_value=mock_response)

    ops = LLMOperations.__new__(LLMOperations)
    ops.processor = MagicMock()
    ops.processor.analyze_gnn = mock_analyze
    ops._initialized = True  # type: ignore[attr-defined]

    out = await LLMOperations._async_summarize_gnn(
        ops, "dummy gnn", max_length=100, ollama_model="smollm2:135m-instruct-q4_K_S"
    )
    assert out == "summary text"
    mock_analyze.assert_awaited_once()
    call_kw = mock_analyze.call_args.kwargs
    assert call_kw["provider_type"] == ProviderType.OLLAMA
    assert isinstance(call_kw["config"], LLMConfig)
    assert call_kw["config"].model == "smollm2:135m-instruct-q4_K_S"


@pytest.mark.unit
def test_structured_prompt_get_response_passes_model_name(tmp_path: Path) -> None:
    """process_llm structured prompt loop must pass model_name matching cache key (ollama_model)."""
    from llm.providers.base_provider import ProviderType

    captured: dict = {}

    async def fake_get_response(*args, **kwargs):
        captured.update(kwargs)
        from llm.providers.base_provider import LLMResponse

        return LLMResponse(content="ok", model_used="m", provider="ollama")

    target = tmp_path / "gnn_in"
    target.mkdir()
    out = tmp_path / "llm_out"
    out.mkdir()
    (target / "one.md").write_text(
        "## GNNSection\nX\n## ModelName\nM\n## StateSpaceBlock\ns[1]\n",
        encoding="utf-8",
    )

    with patch("llm.processor.LLMProcessor") as MockProc:
        instance = MockProc.return_value
        instance.initialize = AsyncMock(return_value=True)
        instance.close = AsyncMock(return_value=None)
        instance.get_available_providers = MagicMock(return_value=[ProviderType.OLLAMA])
        instance.get_response = fake_get_response

        with patch("llm.processor.analyze_gnn_file_with_llm", new=AsyncMock(return_value={"status": "SUCCESS"})):
            with patch("llm.processor.generate_model_insights", return_value={}):
                with patch("llm.processor.generate_code_suggestions", return_value={}):
                    with patch("llm.processor.generate_documentation", return_value={}):
                        with patch(
                            "llm.processor._start_ollama_if_needed",
                            return_value=(True, ["smollm2:135m-instruct-q4_K_S"]),
                        ):
                            with patch(
                                "llm.processor._select_best_ollama_model",
                                return_value="smollm2:135m-instruct-q4_K_S",
                            ):
                                with patch("llm.processor.LLMCache") as MockLLMCache:
                                    _cache_inst = MagicMock()
                                    _cache_inst.get = MagicMock(return_value=None)
                                    _cache_inst.put = MagicMock()
                                    _cache_inst.summary = MagicMock(
                                        return_value={
                                            "hits": 0,
                                            "misses": 0,
                                            "writes": 0,
                                            "hit_ratio_pct": 0.0,
                                            "cache_dir": str(tmp_path),
                                            "entries_on_disk": 0,
                                        }
                                    )
                                    MockLLMCache.return_value = _cache_inst
                                    with patch(
                                        "llm.processor.generate_llm_summary",
                                        return_value="# summary\n",
                                    ):
                                        with patch("llm.processor.get_prompt") as gp:
                                            gp.return_value = {
                                                "system_message": "s",
                                                "user_prompt": "u",
                                                "max_tokens": 512,
                                            }
                                            from llm.processor import process_llm

                                            process_llm(target, out, verbose=False)

    assert captured.get("model_name") == "smollm2:135m-instruct-q4_K_S"
