#!/usr/bin/env python3
"""Tests for non-Ollama LLM provider classes — importability and structural contracts.

These tests validate that every provider class in ``src/llm/providers/`` meets
the ``BaseLLMProvider`` contract without making live API calls. They do NOT
require Ollama, API keys, or network access.

Design:
    - Uses ``sys.path.insert(0, …)`` (not ``src.`` prefix) per the project's
      import convention (see ``.agent_rules/testing.md``).
    - Tests the structural contract: every provider class inherits from
      ``BaseLLMProvider``, exposes ``provider_type``, ``default_model``,
      ``available_models``, and ``validate_config``.
    - Live HTTP tests belong in a separate file marked with ``requires_api_key``
      pytest marker and are excluded from the default CI run.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for direct imports (per project convention)
SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from llm.providers.base_provider import BaseLLMProvider, LLMConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Structural contract: every provider class must meet these
# ---------------------------------------------------------------------------
PROVIDER_CHECKLIST: list[tuple[str, str, str]] = [
    ("openai_provider", "OpenAIProvider", "llm.providers.openai_provider"),
    ("openrouter_provider", "OpenRouterProvider", "llm.providers.openrouter_provider"),
    ("perplexity_provider", "PerplexityProvider", "llm.providers.perplexity_provider"),
    ("ollama_provider", "OllamaProvider", "llm.providers.ollama_provider"),
]


class TestProviderContract:
    """Every provider class must import, instantiate, and expose the public contract."""

    @pytest.mark.parametrize(
        ("module_name", "class_name", "import_path"),
        PROVIDER_CHECKLIST,
        ids=[p[1] for p in PROVIDER_CHECKLIST],
    )
    def test_provider_importable(self, module_name: str, class_name: str,
                                 import_path: str) -> None:
        """Each provider class imports cleanly and inherits from BaseLLMProvider."""
        mod = __import__(import_path, fromlist=[class_name])
        cls: type = getattr(mod, class_name)
        assert issubclass(cls, BaseLLMProvider), (
            f"{class_name} does not inherit from BaseLLMProvider"
        )

        # Instantiate with no API key (structural test, not a live call)
        instance = cls(api_key="test-key")
        assert instance.provider_type is not None
        assert isinstance(instance.default_model, str)
        assert isinstance(instance.available_models, list)
        assert hasattr(instance, "validate_config")
        assert hasattr(instance, "generate_response")
        assert hasattr(instance, "generate_stream")
        assert hasattr(instance, "initialize")

    def test_validate_config_contract(self) -> None:
        """validate_config accepts an LLMConfig and returns bool."""
        for _module_name, class_name, import_path in PROVIDER_CHECKLIST:
            mod = __import__(import_path, fromlist=[class_name])
            cls: type = getattr(mod, class_name)
            instance = cls(api_key="test-key")
            config = LLMConfig(model="test")
            result = instance.validate_config(config)
            assert isinstance(result, bool)

    def test_ollama_default_model_matches(self) -> None:
        """The central default model constant is a non-empty string."""
        from llm.defaults import DEFAULT_OLLAMA_MODEL

        assert isinstance(DEFAULT_OLLAMA_MODEL, str)
        assert len(DEFAULT_OLLAMA_MODEL) > 0

    def test_provider_lazy_accessors_resolve(self) -> None:
        """Lazy provider factories from ``llm.providers.__init__`` resolve."""
        from llm.providers import (
            get_ollama_provider_class,
            get_openai_provider_class,
            get_openrouter_provider_class,
            get_perplexity_provider_class,
        )

        for factory, expected_name in [
            (get_ollama_provider_class, "OllamaProvider"),
            (get_openai_provider_class, "OpenAIProvider"),
            (get_openrouter_provider_class, "OpenRouterProvider"),
            (get_perplexity_provider_class, "PerplexityProvider"),
        ]:
            cls = factory()
            assert cls.__name__ == expected_name, (
                f"Expected {expected_name}, got {cls.__name__}"
            )

    def test_llm_defaults_accessible(self) -> None:
        """The llm/defaults module exports model constants for config."""
        from llm.defaults import DEFAULT_OLLAMA_MODEL

        assert DEFAULT_OLLAMA_MODEL == "smollm2:135m-instruct-q4_K_S"

    def test_provider_type_enum(self) -> None:
        """ProviderType enum covers all four expected providers."""
        from llm.providers.base_provider import ProviderType

        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.OPENROUTER.value == "openrouter"
        assert ProviderType.PERPLEXITY.value == "perplexity"
        assert ProviderType.OLLAMA.value == "ollama"


class TestProviderConcreteBehavior:
    """Edge-case behavior for providers that can be tested without network."""

    def test_ollama_provider_instantiation(self) -> None:
        """OllamaProvider can be instantiated with default model."""
        from llm.providers.ollama_provider import OllamaProvider

        instance = OllamaProvider(api_key="")
        assert instance.default_model is not None
        assert isinstance(instance.default_model, str)

    def test_openai_provider_instantiation(self) -> None:
        """OpenAIProvider can be instantiated with api_key."""
        from llm.providers.openai_provider import OpenAIProvider

        instance = OpenAIProvider(api_key="sk-test-key-12345")
        assert instance.default_model is not None

    def test_openrouter_provider_instantiation(self) -> None:
        """OpenRouterProvider can be instantiated with api_key."""
        from llm.providers.openrouter_provider import OpenRouterProvider

        instance = OpenRouterProvider(api_key="sk-or-test-key")
        assert instance.default_model is not None

    def test_perplexity_provider_instantiation(self) -> None:
        """PerplexityProvider can be instantiated with api_key."""
        from llm.providers.perplexity_provider import PerplexityProvider

        instance = PerplexityProvider(api_key="pplx-test-key")
        assert instance.default_model is not None