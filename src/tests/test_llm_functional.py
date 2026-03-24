#!/usr/bin/env python3
"""
Functional tests for the LLM module.

Tests LLMProcessor initialization, prompt generation, provider discovery,
and graceful degradation without live LLM providers.
"""

import pytest
import sys
import os
import logging
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.llm_processor import (
    LLMProcessor,
    AnalysisType,
    GNNLLMProcessor,
    load_api_keys_from_env,
    get_default_provider_configs,
    get_preferred_providers_from_env,
    create_gnn_llm_processor,
)
from llm.providers.base_provider import ProviderType
from llm.prompts import (
    PromptType,
    get_prompt,
    get_all_prompt_types,
    get_prompt_title,
    get_default_prompt_sequence,
    GNN_ANALYSIS_PROMPTS,
)
from llm.processor import _select_best_ollama_model


class TestLLMProcessorInitialization:
    """Test LLMProcessor construction and configuration."""

    @pytest.mark.unit
    def test_default_initialization(self) -> None:
        """LLMProcessor should initialize with default provider order."""
        proc = LLMProcessor()
        assert proc.preferred_providers[0] == ProviderType.OLLAMA
        assert proc._initialized is False
        assert proc.providers == {}

    @pytest.mark.unit
    def test_custom_provider_order(self) -> None:
        """LLMProcessor should accept a custom provider ordering."""
        order = [ProviderType.OPENAI, ProviderType.PERPLEXITY]
        proc = LLMProcessor(preferred_providers=order)
        assert proc.preferred_providers == order

    @pytest.mark.unit
    def test_custom_api_keys(self) -> None:
        """LLMProcessor should store provided API keys."""
        keys = {"openai": "sk-test-123", "ollama": "local"}
        proc = LLMProcessor(api_keys=keys)
        assert proc.api_keys == keys

    @pytest.mark.unit
    def test_get_available_providers_empty_before_init(self) -> None:
        """Before initialization, no providers should be available."""
        proc = LLMProcessor()
        assert proc.get_available_providers() == []

    @pytest.mark.unit
    def test_get_provider_returns_none_for_missing(self) -> None:
        """get_provider should return None for an unregistered provider."""
        proc = LLMProcessor()
        assert proc.get_provider(ProviderType.OPENAI) is None

    @pytest.mark.unit
    def test_get_best_provider_returns_none_when_empty(self) -> None:
        """get_best_provider_for_task returns None when no providers registered."""
        proc = LLMProcessor()
        result = proc.get_best_provider_for_task(AnalysisType.SUMMARY)
        assert result is None

    @pytest.mark.unit
    def test_get_provider_info_empty(self) -> None:
        """get_provider_info should return empty dict before init."""
        proc = LLMProcessor()
        assert proc.get_provider_info() == {}


class TestLLMPrompts:
    """Test prompt generation and template system."""

    @pytest.mark.unit
    def test_get_all_prompt_types(self) -> None:
        """get_all_prompt_types should return all PromptType enum values."""
        types = get_all_prompt_types()
        assert isinstance(types, list)
        assert len(types) > 0
        for pt in types:
            assert isinstance(pt, PromptType)

    @pytest.mark.unit
    def test_get_prompt_returns_required_keys(self) -> None:
        """get_prompt should produce a dict with system_message and user_prompt."""
        sample_gnn = "## GNNSection\nActInfPOMDP\n## ModelName\nTestModel"
        result = get_prompt(PromptType.EXPLAIN_MODEL, sample_gnn)
        assert "system_message" in result
        assert "user_prompt" in result
        assert isinstance(result["system_message"], str)
        assert len(result["system_message"]) > 0

    @pytest.mark.unit
    def test_get_prompt_inserts_gnn_content(self) -> None:
        """The user_prompt should contain the GNN content passed in."""
        gnn_content = "## StateSpaceBlock\nA[3,3]"
        result = get_prompt(PromptType.SUMMARIZE_CONTENT, gnn_content)
        assert gnn_content in result["user_prompt"]

    @pytest.mark.unit
    def test_get_prompt_title(self) -> None:
        """get_prompt_title should return a non-empty title string."""
        title = get_prompt_title(PromptType.EXPLAIN_MODEL)
        assert isinstance(title, str)
        assert len(title) > 0

    @pytest.mark.unit
    def test_default_prompt_sequence(self) -> None:
        """get_default_prompt_sequence should return a list of PromptTypes."""
        seq = get_default_prompt_sequence()
        assert isinstance(seq, list)
        assert len(seq) > 0
        for item in seq:
            assert isinstance(item, PromptType)

    @pytest.mark.unit
    def test_all_prompt_types_generate_valid_prompt(self) -> None:
        """Every implemented PromptType should produce a valid prompt dict."""
        gnn = "## ModelName\nMyModel\n## StateSpaceBlock\ns[3,1]"
        for pt in GNN_ANALYSIS_PROMPTS:
            result = get_prompt(pt, gnn)
            assert "system_message" in result, f"Missing system_message for {pt}"
            assert "user_prompt" in result, f"Missing user_prompt for {pt}"


class TestModelSelection:
    """Test Ollama model selection logic."""

    @pytest.mark.unit
    def test_select_preferred_model(self, monkeypatch: Any) -> None:
        """Should pick the first matching preferred model."""
        monkeypatch.setattr("llm.processor._get_llm_config", lambda: {})
        logger = logging.getLogger("test")
        available = ["llama2:7b", "gemma3:4b", "phi3"]
        selected = _select_best_ollama_model(available, logger)
        assert selected == "gemma3:4b"

    @pytest.mark.unit
    def test_select_fallback_to_first(self, monkeypatch: Any) -> None:
        """Should fall back to first model if none match preferences."""
        monkeypatch.setattr("llm.processor._get_llm_config", lambda: {})
        logger = logging.getLogger("test")
        available = ["custom-model:latest"]
        selected = _select_best_ollama_model(available, logger)
        assert selected == "custom-model:latest"

    @pytest.mark.unit
    def test_select_default_when_empty(self, monkeypatch: Any) -> None:
        """Should return default model when no models available."""
        monkeypatch.setattr("llm.processor._get_llm_config", lambda: {})
        logger = logging.getLogger("test")
        from llm.defaults import DEFAULT_OLLAMA_MODEL

        selected = _select_best_ollama_model([], logger)
        assert selected == DEFAULT_OLLAMA_MODEL

    def test_select_from_env_variable(self, monkeypatch: Any) -> None:
        """Should honor OLLAMA_MODEL environment variable."""
        monkeypatch.setattr("llm.processor._get_llm_config", lambda: {})
        logger = logging.getLogger("test")
        original_val = os.environ.get("OLLAMA_MODEL")
        os.environ["OLLAMA_MODEL"] = "my-custom:latest"
        try:
            selected = _select_best_ollama_model(["gemma3:4b"], logger)
            assert selected == "my-custom:latest"
        finally:
            if original_val is not None:
                os.environ["OLLAMA_MODEL"] = original_val
            else:
                del os.environ["OLLAMA_MODEL"]


class TestEnvironmentLoading:
    """Test environment-based configuration helpers."""

    def test_load_api_keys_includes_ollama(self) -> None:
        """Ollama should be included by default when not disabled."""
        original_env = os.environ.copy()
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["OPENROUTER_API_KEY"] = ""
        os.environ["PERPLEXITY_API_KEY"] = ""
        os.environ["OLLAMA_DISABLED"] = "0"
        try:
            keys = load_api_keys_from_env()
            assert "ollama" in keys
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def test_load_api_keys_ollama_disabled(self) -> None:
        """When OLLAMA_DISABLED=1, ollama key should not be present."""
        original_env = os.environ.copy()
        os.environ["OLLAMA_DISABLED"] = "1"
        try:
            keys = load_api_keys_from_env()
            assert "ollama" not in keys
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    @pytest.mark.unit
    def test_get_default_provider_configs_structure(self) -> None:
        """Provider configs should contain expected provider keys."""
        configs = get_default_provider_configs()
        assert "ollama" in configs
        assert "openai" in configs
        assert "openrouter" in configs
        assert "perplexity" in configs

    @pytest.mark.unit
    def test_get_preferred_providers_from_env_default(self) -> None:
        """Default preferred providers should start with OLLAMA."""
        providers = get_preferred_providers_from_env()
        assert isinstance(providers, list)
        assert providers[0] == ProviderType.OLLAMA


class TestGNNLLMProcessor:
    """Test the specialized GNN LLM processor wrapper."""

    @pytest.mark.unit
    def test_create_gnn_llm_processor(self) -> None:
        """create_gnn_llm_processor should return a GNNLLMProcessor."""
        proc = create_gnn_llm_processor()
        assert isinstance(proc, GNNLLMProcessor)
        assert proc.initialized is False

    @pytest.mark.unit
    def test_gnn_processor_not_initialized_returns_error(self) -> None:
        """analyze_gnn_model should return error dict when not initialized."""
        import asyncio
        proc = GNNLLMProcessor()
        result = asyncio.run(proc.analyze_gnn_model("test content"))
        assert result["success"] is False
        assert "not initialized" in result["error"]
