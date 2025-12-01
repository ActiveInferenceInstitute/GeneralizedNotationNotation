#!/usr/bin/env python3
"""
Test LLM Ollama Provider

These tests exercise the Ollama provider end-to-end. They are marked safe_to_fail
and will skip if a local Ollama runtime is not detected. The model used is
configurable via OLLAMA_TEST_MODEL, defaulting to 'gemma2:2b'.
"""

import os
import sys
import shutil
import subprocess
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
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return result.returncode == 0
            except Exception:
                return False
        return False


OLLAMA_TEST_MODEL = os.getenv("OLLAMA_TEST_MODEL", os.getenv("OLLAMA_MODEL", "smollm2:135m-instruct-q4_K_S"))


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
    
    # The provider has a CLI fallback that can succeed even when the Python
    # ollama package fails to list models. Test that initialization succeeds
    # if either the Python client OR CLI fallback works.
    if ok:
        # Initialization succeeded (either via Python client or CLI fallback)
        assert provider.is_initialized() is True
        info = provider.get_provider_info()
        assert info["provider_type"] == "ollama"
    else:
        # Ollama is not available at all - neither Python client nor CLI
        assert provider.is_initialized() is False


@pytest.mark.unit
@pytest.mark.safe_to_fail
def test_ollama_simple_chat(monkeypatch):
    if not _ollama_available():
        pytest.skip("Ollama not available locally")

    from llm.providers.ollama_provider import OllamaProvider
    from llm.providers.base_provider import LLMMessage, LLMConfig
    import asyncio

    provider = OllamaProvider()
    assert provider.initialize() is True

    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Reply with the word 'ok'."),
    ]
    config = LLMConfig(model=OLLAMA_TEST_MODEL, max_tokens=32, temperature=0.0)

    async def _run():
        res = await provider.generate_response(messages, config)
        return res

    result = asyncio.run(_run())
    assert isinstance(result.content, str)
    assert len(result.content) > 0
    assert result.provider == "ollama"


@pytest.mark.unit
@pytest.mark.safe_to_fail
def test_ollama_streaming(monkeypatch):
    if not _ollama_available():
        pytest.skip("Ollama not available locally")

    from llm.providers.ollama_provider import OllamaProvider
    from llm.providers.base_provider import LLMMessage, LLMConfig
    import asyncio

    provider = OllamaProvider()
    assert provider.initialize() is True

    messages = [
        LLMMessage(role="user", content="Give a 1-sentence summary of Active Inference.")
    ]
    config = LLMConfig(model=OLLAMA_TEST_MODEL, max_tokens=64, temperature=0.2, stream=True)

    async def _run():
        chunks = []
        async for chunk in provider.generate_stream(messages, config):
            chunks.append(chunk)
            if len("".join(chunks)) > 20:
                break
        return "".join(chunks)

    text = asyncio.run(_run())
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.integration
@pytest.mark.safe_to_fail
def test_processor_uses_ollama_when_no_keys(monkeypatch):
    # Clear cloud keys to force local provider preference
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_PROVIDER", "ollama")

    from llm.llm_processor import LLMProcessor, AnalysisType, LLMConfig

    processor = LLMProcessor()
    initialized = False
    try:
        import asyncio

        async def _init():
            return await processor.initialize()

        initialized = asyncio.run(_init())
    except Exception:
        initialized = False

    # The LLM processor can initialize with Ollama via CLI fallback even if
    # the Python ollama package check fails. Test both cases appropriately.
    if not initialized:
        # Ollama not available at all - skip the rest of the test
        pytest.skip("Ollama provider not available for initialization")
        return

    # If initialized, verify we can use the processor
    assert initialized is True
    
    # Skip the actual LLM call if Ollama service is not running
    if not _ollama_available():
        # Initialized via CLI but service not running for queries
        pytest.skip("Ollama service not running for LLM queries")
        return
    
    # Try a short analysis
    content = "## ModelName\nTestModel\n\n## StateSpaceBlock\ns[2]\n"

    async def _run():
        return await processor.analyze_gnn(content, AnalysisType.SUMMARY, config=LLMConfig(model=OLLAMA_TEST_MODEL))

    result = asyncio.run(_run())
    assert isinstance(result.content, str)
    assert len(result.content) > 0


