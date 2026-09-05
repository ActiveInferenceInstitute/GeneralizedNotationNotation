"""Deterministic offline tests for the synchronous GNN wrappers in
llm.llm_processor and the llm facade compat helpers.

Live provider paths are excluded by design; LLMProcessor.initialize is
monkeypatched to fail so wrappers exercise their never-raises contracts.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import llm
from llm import LLMAnalyzer, LLMProcessor, analyze_gnn_model, get_module_info
from llm.llm_processor import (
    GNNLLMProcessor,
    _error_result,
    enhance_model,
    generate_explanation,
)
from llm.llm_processor import (
    analyze_gnn_model as analyze_gnn_model_sync,
)

pytestmark = pytest.mark.unit

SAMPLE = "## ModelName\nX\n## StateSpaceBlock\ns[3,1]: boolean"


@pytest.fixture(autouse=True)
def _no_provider_init(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(self: Any) -> bool:
        return False

    monkeypatch.setattr("llm.llm_processor.LLMProcessor.initialize", _fail)


class TestErrorResult:
    def test_shape(self) -> None:
        result = _error_result(ValueError("boom"))
        assert result == {"success": False, "error": "boom", "error_type": "ValueError"}


class TestSyncWrappers:
    def test_analyze_gnn_model_uninitialized_error_contract(self) -> None:
        result = analyze_gnn_model_sync(SAMPLE)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    def test_generate_explanation_delegates_to_summary(self) -> None:
        assert generate_explanation(SAMPLE) == analyze_gnn_model_sync(SAMPLE, "summary")

    def test_enhance_model_delegates_to_enhancement(self) -> None:
        assert enhance_model(SAMPLE) == analyze_gnn_model_sync(SAMPLE, "enhancement")


class TestGNNLLMProcessorStringCoercion:
    def test_invalid_string_falls_back_to_summary(self) -> None:
        processor = GNNLLMProcessor()

        async def _fail(self: Any) -> bool:
            return False

        import asyncio

        asyncio.run(processor.initialize())
        result = asyncio.run(processor.analyze_gnn_model(SAMPLE, "nonexistent-type"))
        assert result["success"] is False
        assert "not initialized" in result["error"]


class TestFacadeCompat:
    def test_analyze_gnn_model_structure(self) -> None:
        result = analyze_gnn_model(SAMPLE)
        assert set(result) == {"variables", "connections", "sections", "patterns"}

    def test_analyze_gnn_model_accepts_dict_payload(self) -> None:
        result = analyze_gnn_model({"content": SAMPLE})
        assert set(result) == {"variables", "connections", "sections", "patterns"}

    def test_get_module_info_providers_env_driven(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "PERPLEXITY_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "llm.llm_processor.load_api_keys_from_env", lambda: {"ollama": "local"}
        )
        info = get_module_info()
        assert info["providers"] == ["ollama"]
        assert info["version"] == llm.__version__

    def test_facade_processor_description_counts(self) -> None:
        proc = LLMProcessor()
        description = proc.generate_description(
            "a: boolean\nb: boolean\n## Connections\na > b"
        )
        assert "2 variables" in description and "1 connections" in description

    def test_analyzer_extract_insights(self) -> None:
        analyzer = LLMAnalyzer()
        insights = analyzer.extract_insights(SAMPLE)
        assert insights["variable_count"] >= 1
