#!/usr/bin/env python3
"""
Test LLM Ollama Provider

Unit tests cover `OllamaProvider` configuration and validation without a running
daemon. Chat/stream tests use `asyncio.run()` so they collect under `--strict-markers`
without `pytest-anyio`; they are marked `safe_to_fail` and skip when Ollama is
not reachable. Model name defaults follow `OLLAMA_TEST_MODEL`, then `OLLAMA_MODEL`,
then ``llm.defaults.DEFAULT_OLLAMA_MODEL`` (smollm2 instruct).
"""

import asyncio
import os
import shutil
import subprocess  # nosec B404 -- subprocess calls with controlled/trusted input
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _ollama_available() -> bool:
    """Check if Ollama is available AND service is running."""
    try:
        import ollama  # noqa: F401
        # Python client available, try to list models to verify service is running
        try:
            ollama.list()
            return True
        except Exception:
            # Python client installed but service not running
            return False
    except ImportError:
        # Fall back to CLI check
        if shutil.which("ollama") is not None:
            # CLI exists, check if service is running by trying to list models
            try:
                result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            except Exception:
                return False
        return False


from llm.defaults import DEFAULT_OLLAMA_MODEL

OLLAMA_TEST_MODEL = os.getenv(
    "OLLAMA_TEST_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
)


@pytest.mark.unit
@pytest.mark.safe_to_fail
def test_import_ollama_provider():
    try:
        from llm.providers.ollama_provider import OllamaProvider  # noqa: F401
    except Exception as e:
        pytest.skip(f"Ollama provider import unavailable: {e}")


@pytest.mark.unit
@pytest.mark.safe_to_fail
def test_ollama_provider_initialize(monkeypatch):
    from llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider()
    ok = provider.initialize()

    # The provider has a CLI recovery that can succeed even when the Python
    # ollama package fails to list models. Test that initialization succeeds
    # if either the Python client OR CLI recovery works.
    if ok:
        # Initialization succeeded (either via Python client or CLI recovery)
        assert provider.is_initialized is True
        info = provider.get_provider_info()
        assert info["provider_type"] == "ollama"
    else:
        # Ollama is not available at all - neither Python client nor CLI
        assert provider.is_initialized is False


@pytest.mark.unit
@pytest.mark.safe_to_fail
@pytest.mark.timeout(30)  # Prevent hanging if Ollama is slow
def test_ollama_simple_chat(monkeypatch):
    if not _ollama_available():
        pytest.skip("Ollama not available locally")

    async def _run() -> None:
        from llm.providers.base_provider import LLMConfig, LLMMessage
        from llm.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        assert provider.initialize() is True

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Reply with the word 'ok'."),
        ]
        config = LLMConfig(model=OLLAMA_TEST_MODEL, max_tokens=32, temperature=0.0)

        result = await provider.generate_response(messages, config)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert result.provider == "ollama"

    asyncio.run(_run())


@pytest.mark.unit
@pytest.mark.safe_to_fail
@pytest.mark.timeout(120)  # CLI stream can approach OLLAMA_TIMEOUT on loaded hosts
def test_ollama_streaming(monkeypatch):
    if not _ollama_available():
        pytest.skip("Ollama not available locally")

    async def _run() -> None:
        from llm.providers.base_provider import LLMConfig, LLMMessage
        from llm.providers.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        assert provider.initialize() is True

        messages = [
            LLMMessage(role="user", content="Give a 1-sentence summary of Active Inference.")
        ]
        config = LLMConfig(model=OLLAMA_TEST_MODEL, max_tokens=64, temperature=0.2, stream=True)

        chunks: list[str] = []
        async for chunk in provider.generate_stream(messages, config):
            chunks.append(chunk)
            if len("".join(chunks)) > 20:
                break

        text = "".join(chunks)
        assert isinstance(text, str)
        assert len(text) > 0

    asyncio.run(_run())


@pytest.mark.integration
@pytest.mark.slow  # This test makes real LLM calls which can be slow
@pytest.mark.safe_to_fail
@pytest.mark.timeout(30)  # Prevent hanging during pipeline runs
def test_processor_uses_ollama_when_no_keys(monkeypatch):
    # Clear cloud keys to force local provider preference
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")

    async def _run() -> None:
        from llm.llm_processor import LLMConfig, LLMProcessor
        from llm.providers.base_provider import LLMMessage, ProviderType

        processor = LLMProcessor()
        initialized = False
        try:
            initialized = await processor.initialize()
        except Exception:
            initialized = False

        # The LLM processor can initialize with Ollama via CLI recovery even if
        # the Python ollama package check fails. Test both cases appropriately.
        if not initialized:
            pytest.skip("Ollama provider not available for initialization")

        assert initialized is True

        if not _ollama_available():
            pytest.skip("Ollama service not running for LLM queries")

        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Reply with the word 'ok'."),
        ]
        result = await processor.get_response(
            messages,
            provider_type=ProviderType.OLLAMA,
            max_tokens=32,
            temperature=0.0,
            config=LLMConfig(model=OLLAMA_TEST_MODEL),
        )
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    asyncio.run(_run())


@pytest.mark.unit
def test_llm_processor_includes_ollama_timeout_from_env(monkeypatch):
    """Bare ``LLMProcessor()`` must apply ``get_default_provider_configs`` (OLLAMA_TIMEOUT, etc.)."""
    monkeypatch.setenv("OLLAMA_TIMEOUT", "99")
    from llm.llm_processor import LLMProcessor

    proc = LLMProcessor()
    assert proc.provider_configs.get("ollama", {}).get("timeout") == 99.0


@pytest.mark.unit
class TestOllamaProviderConfig:
    """OllamaProvider behavior that does not require a running Ollama server."""

    def test_validate_config_rejects_non_positive_max_tokens(self):
        from llm.providers.base_provider import LLMConfig
        from llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig(max_tokens=0)) is False
        assert p.validate_config(LLMConfig(max_tokens=-1)) is False

    def test_validate_config_rejects_temperature_out_of_range(self):
        from llm.providers.base_provider import LLMConfig
        from llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig(temperature=-0.1)) is False
        assert p.validate_config(LLMConfig(temperature=2.1)) is False

    def test_validate_config_accepts_defaults_and_edges(self):
        from llm.providers.base_provider import LLMConfig
        from llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.validate_config(LLMConfig()) is True
        assert p.validate_config(LLMConfig(max_tokens=1, temperature=0.0)) is True
        assert p.validate_config(LLMConfig(max_tokens=512, temperature=2.0)) is True

    def test_default_model_and_available_models(self):
        from llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.default_model == OllamaProvider.DEFAULT_MODEL
        assert p.available_models == list(OllamaProvider.AVAILABLE_MODELS)

        q = OllamaProvider(default_model="custom:tag")
        assert q.default_model == "custom:tag"

    def test_provider_type_and_info_uninitialized(self):
        from llm.providers.base_provider import ProviderType
        from llm.providers.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.provider_type == ProviderType.OLLAMA
        info = p.get_provider_info()
        assert info["provider_type"] == "ollama"
        assert info["is_initialized"] is False
        assert info["default_model"] == OllamaProvider.DEFAULT_MODEL

