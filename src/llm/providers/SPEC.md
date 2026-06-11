# LLM Providers — Technical Specification

**Version**: 1.6.0

## Provider Interface Contract

All providers must implement `BaseLLMProvider`:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]: ...
    @abstractmethod
    def get_available_models(self) -> List[str]: ...
    @abstractmethod
    def health_check(self) -> bool: ...
```

## Provider Configuration

| Provider | Env Variable | Default Model |
|----------|-------------|---------------|
| Ollama | `OLLAMA_MODEL` | `smollm2:135m-instruct-q4_K_S` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3-8b` |
| Perplexity | `PERPLEXITY_API_KEY` | `llama-3.1-sonar-small` |

## Fallback Chain

Ollama → OpenRouter → OpenAI → Perplexity
