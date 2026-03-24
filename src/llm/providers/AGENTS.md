# LLM Providers

## Module overview

**Purpose**: Provider adapters behind `LLMProcessor` (Step 13).

**Status**: Production ready

## Provider types (`base_provider.ProviderType`)

| Value | Module | Notes |
|-------|--------|--------|
| `ollama` | `ollama_provider.py` | Local; Python client preferred, CLI fallback |
| `openai` | `openai_provider.py` | Requires `OPENAI_API_KEY` |
| `openrouter` | `openrouter_provider.py` | Requires `OPENROUTER_API_KEY` |
| `perplexity` | `perplexity_provider.py` | Requires `PERPLEXITY_API_KEY` |

Shared types: `LLMMessage`, `LLMConfig`, `LLMResponse`, `BaseLLMProvider`.

Factory accessors (lazy import): `get_ollama_provider_class`, `get_openai_provider_class`, `get_openrouter_provider_class`, `get_perplexity_provider_class`.

## `OllamaProvider` (`ollama_provider.py`)

**Construction** (`**kwargs`): `base_url` (optional), `default_model`, `default_max_tokens` (default 256), `timeout` (if omitted: reads `OLLAMA_TIMEOUT`, default `60` seconds).

**Class attributes**: `AVAILABLE_MODELS` (suggested tags), `DEFAULT_MODEL` (alias of `llm.defaults.DEFAULT_OLLAMA_MODEL`, currently `smollm2:135m-instruct-q4_K_S`).

| Method | Description |
|--------|-------------|
| `initialize() -> bool` | Loads PyPI `ollama` and checks `chat` + `list()`; on failure, sets CLI mode if `ollama` is on `PATH` |
| `validate_config(config) -> bool` | Rejects `max_tokens <= 0`; temperature must be in `[0.0, 2.0]` |
| `generate_response(messages, config)` | Async; CLI path tries `ollama chat <model> --json` then `ollama run <model> <prompt>` |
| `generate_stream(messages, config)` | Async; CLI emits a single chunk from `ollama run` |
| `analyze(content, task) -> str` | Sync wrapper around `generate_response` |
| `close()` | Clears initialized flag |

**Properties**: `provider_type`, `default_model`, `available_models`, `is_initialized` (from base). `get_provider_info()` returns a dict snapshot.

## Testing

- `src/tests/test_llm_ollama.py` — unit tests for validation/defaults; optional live Ollama tests (`safe_to_fail`)
- `src/tests/test_llm_ollama_integration.py` — processor selection and detection helpers

```bash
uv run pytest src/tests/test_llm_ollama.py src/tests/test_llm_ollama_integration.py -v
```
