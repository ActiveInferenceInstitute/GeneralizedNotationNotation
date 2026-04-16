# LLM Provider Sub-module

## Overview

Implements the provider abstraction layer for the LLM module. Each provider implements the `BaseLLMProvider` abstract class, ensuring consistent behavior across different LLM backends.

## Architecture

```
providers/
├── __init__.py                # Provider registry and factory (48 lines)
├── base_provider.py           # Abstract base class defining the provider interface (255 lines)
├── ollama_provider.py         # Ollama local model provider (330 lines)
├── openai_provider.py         # OpenAI API provider (354 lines)
├── openrouter_provider.py     # OpenRouter multi-model gateway (436 lines)
└── perplexity_provider.py     # Perplexity AI provider (419 lines)
```

## Provider Interface

All providers implement `BaseLLMProvider` with these core methods:

- **`generate(prompt, **kwargs) → str`** — Synchronous text generation.
- **`generate_stream(prompt, **kwargs) → AsyncGenerator`** — Streaming text generation.
- **`get_available_models() → List[str]`** — List models available from this provider.
- **`health_check() → bool`** — Verify provider connectivity.

## Provider Selection

The provider factory in `__init__.py` selects providers based on:
1. Explicit configuration in `input/config.yaml`
2. Environment variables (`OLLAMA_MODEL`, `OPENAI_API_KEY`, etc.)
3. Fallback chain: Ollama → OpenRouter → OpenAI → Perplexity

## Parent Module

See [llm/AGENTS.md](../AGENTS.md) for the overall LLM architecture.

**Version**: 1.6.0
