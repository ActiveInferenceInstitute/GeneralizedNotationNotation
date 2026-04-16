# LLM Providers

Provider abstraction layer implementing multiple LLM backends with a unified interface.

## Available Providers

| Provider | File | Backend |
|----------|------|---------|
| Ollama | `ollama_provider.py` | Local Ollama server |
| OpenAI | `openai_provider.py` | OpenAI API |
| OpenRouter | `openrouter_provider.py` | Multi-model gateway |
| Perplexity | `perplexity_provider.py` | Perplexity AI |

## Usage

```python
from llm.providers import get_provider

provider = get_provider("ollama")  # or "openai", "openrouter", "perplexity"
response = provider.generate("Explain this GNN model...")
```

## Adding a New Provider

1. Create `new_provider.py` inheriting from `BaseLLMProvider`
2. Implement `generate()`, `generate_stream()`, `get_available_models()`, `health_check()`
3. Register in `__init__.py` provider factory

## See Also

- [Parent: llm/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
