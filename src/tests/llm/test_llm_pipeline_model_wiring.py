#!/usr/bin/env python3
"""Unit tests for LLM pipeline model wiring (Step 14): Ollama tag + SUMMARY provider order."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.unit
def test_summary_task_prefers_ollama_when_registered() -> None:
    from llm.llm_processor import LLMProcessor
    from llm.providers.base_provider import ProviderType

    class ProviderProbe:
        def __init__(self, ptype):
            self.provider_type = ptype

    proc = LLMProcessor(
        preferred_providers=[ProviderType.OLLAMA, ProviderType.OPENAI]
    )
    ollama_provider = ProviderProbe(ProviderType.OLLAMA)
    openai_provider = ProviderProbe(ProviderType.OPENAI)
    proc.providers = {
        ProviderType.OLLAMA: ollama_provider,
        ProviderType.OPENAI: openai_provider,
    }

    from llm.llm_processor import AnalysisType

    best = proc.get_best_provider_for_task(AnalysisType.SUMMARY)
    assert best is ollama_provider


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_summarize_with_ollama_model_uses_ollama_provider_and_config() -> None:
    from llm.llm_operations import LLMOperations
    from llm.providers.base_provider import LLMConfig, LLMResponse, ProviderType

    response = LLMResponse(
        content="summary text",
        model_used="smollm2:135m-instruct-q4_K_S",
        provider="ollama",
    )

    class RecordingProcessor:
        def __init__(self):
            self.called = False
            self.call_args = {}
        async def analyze_gnn(self, **kwargs):
            self.called = True
            self.call_args = kwargs
            return response

    ops = LLMOperations.__new__(LLMOperations)
    ops.processor = RecordingProcessor()
    ops._initialized = True  # type: ignore[attr-defined]

    out = await LLMOperations._async_summarize_gnn(
        ops, "sample gnn", max_length=100, ollama_model="smollm2:135m-instruct-q4_K_S"
    )
    assert out == "summary text"
    assert ops.processor.called
    call_kw = ops.processor.call_args
    assert call_kw["provider_type"] == ProviderType.OLLAMA
    assert isinstance(call_kw["config"], LLMConfig)
    assert call_kw["config"].model == "smollm2:135m-instruct-q4_K_S"


@pytest.mark.unit
def test_structured_prompt_get_response_passes_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """process_llm structured prompt loop must pass model_name matching cache key (ollama_model)."""
    from llm.providers.base_provider import ProviderType

    captured: dict = {}

    async def controlled_get_response(*args, **kwargs):
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

    with monkeypatch.context() as m:
        async def controlled_init(*args, **kwargs): return True
        async def controlled_close(*args, **kwargs): return None
        
        class InMemoryProcessor:
            def __init__(self, *args, **kwargs):
                self.initialize = controlled_init
                self.close = controlled_close
                self.get_response = controlled_get_response
                self.get_available_providers = lambda: [ProviderType.OLLAMA]

        m.setattr("llm.processor.LLMProcessor", lambda *args, **kwargs: InMemoryProcessor())
        
        async def controlled_analyze(*args, **kwargs): return {"status": "SUCCESS"}
        m.setattr("llm.processor.analyze_gnn_file_with_llm", controlled_analyze)
        
        m.setattr("llm.processor.generate_model_insights", lambda *args, **kwargs: {})
        m.setattr("llm.processor.generate_code_suggestions", lambda *args, **kwargs: {})
        m.setattr("llm.processor.generate_documentation", lambda *args, **kwargs: {})
        m.setattr("llm.processor._start_ollama_if_needed", lambda *args, **kwargs: (True, ["smollm2:135m-instruct-q4_K_S"]))
        m.setattr("llm.processor._select_best_ollama_model", lambda *args, **kwargs: "smollm2:135m-instruct-q4_K_S")
        
        class InMemoryCache:
            def get(self, *args, **kwargs): return None
            def put(self, *args, **kwargs): pass
            def summary(self, *args, **kwargs): 
                return {
                    "hits": 0, "misses": 0, "writes": 0, "hit_ratio_pct": 0.0, 
                    "cache_dir": str(tmp_path), "entries_on_disk": 0
                }
                
        m.setattr("llm.processor.LLMCache", lambda *args, **kwargs: InMemoryCache())
        m.setattr("llm.processor.generate_llm_summary", lambda *args, **kwargs: "# summary\n")
        m.setattr("llm.processor.get_prompt", lambda *args, **kwargs: {
            "system_message": "s", "user_prompt": "u", "max_tokens": 512
        })
        
        from llm.processor import process_llm
        process_llm(target, out, verbose=False)

    assert captured.get("model_name") == "smollm2:135m-instruct-q4_K_S"
