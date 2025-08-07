#!/usr/bin/env python3
"""
Ollama LLM Provider

Local provider using the `ollama` Python package to call a local Ollama server.

Requirements:
- pip install ollama>=0.3.0
- Run `ollama serve` and ensure a model is pulled, e.g. `ollama pull gemma2:2b`
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
import shutil
import subprocess
import json

from .base_provider import (
    BaseLLMProvider,
    ProviderType,
    LLMResponse,
    LLMMessage,
    LLMConfig,
)

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama implementation of the LLM provider interface."""

    # Include tiny models for low-latency defaults; larger ones stay optional
    AVAILABLE_MODELS = [
        "smollm2:135m-instruct-q4_K_S",
        "smollm2:360m",
        "tinyllama:1.1b",
        "smollm2:1.7b",
        "gemma2:2b",
        "mistral:7b",
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen2:7b",
    ]

    DEFAULT_MODEL = "smollm2:135m-instruct-q4_K_S"

    def __init__(self, **kwargs):
        super().__init__(api_key=None, **kwargs)
        self.base_url = kwargs.get("base_url")  # optional custom host
        self.default_model_override = kwargs.get("default_model")
        self.default_max_tokens = kwargs.get("default_max_tokens", 256)
        self.default_timeout = kwargs.get("timeout", 60.0)
        self._ollama = None
        self._use_cli = False

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    @property
    def default_model(self) -> str:
        return self.default_model_override or self.DEFAULT_MODEL

    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS.copy()

    def initialize(self) -> bool:
        # Prefer Python client
        try:
            import ollama  # type: ignore
            self._ollama = ollama
            try:
                _ = self._ollama.list()
            except Exception as e:
                logger.debug(f"Ollama list models failed (non-fatal): {e}")
            self._is_initialized = True
            logger.info("Ollama provider initialized (python client)")
            return True
        except ImportError:
            # Fallback to CLI if available
            if shutil.which("ollama"):
                self._use_cli = True
                self._is_initialized = True
                logger.info("Ollama provider initialized (CLI fallback)")
                return True
            logger.error("Ollama not available. Install python client with 'pip install ollama' or install Ollama CLI from https://ollama.ai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            return False

    def validate_config(self, config: LLMConfig) -> bool:
        # Ollama accepts many models; don't strictly enforce model list
        if config.max_tokens is not None and config.max_tokens <= 0:
            logger.error("max_tokens must be positive")
            return False
        if config.temperature is not None and not (0.0 <= config.temperature <= 2.0):
            logger.error("temperature must be between 0.0 and 2.0")
            return False
        return True

    async def generate_response(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        if not self.is_initialized():
            raise RuntimeError("Ollama provider not initialized")

        if config is None:
            config = LLMConfig()

        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")

        # Convert to Ollama message format
        ollama_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        try:
            import asyncio
            if self._use_cli:
                # Build prompt by joining message contents, preserving order
                prompt = "\n\n".join(m.content for m in messages)
                model = config.model or self.default_model

                def _call_cli() -> Dict[str, Any]:
                    # Prefer JSON mode via `ollama chat` if available; fallback to `ollama run`
                    try:
                        completed = subprocess.run(
                            ["ollama", "chat", model, "--json"],
                            input=json.dumps({
                                "messages": [{"role": m.role, "content": m.content} for m in messages],
                                "options": {
                                    "num_predict": config.max_tokens or self.default_max_tokens,
                                    "temperature": config.temperature if config.temperature is not None else 0.2,
                                }
                            }),
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=self.default_timeout,
                        )
                        if completed.returncode == 0 and completed.stdout.strip():
                            return json.loads(completed.stdout)
                    except Exception:
                        pass
                    # Fallback to `ollama run`
                    completed = subprocess.run(
                        ["ollama", "run", model, prompt],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=self.default_timeout,
                    )
                    return {"model": model, "message": {"content": completed.stdout.strip()}}

                response = await asyncio.to_thread(_call_cli)
            else:
                # Python client
                def _call_py() -> Dict[str, Any]:
                    return self._ollama.chat(
                        model=config.model or self.default_model,
                        messages=ollama_messages,
                        options={
                            "num_predict": config.max_tokens or self.default_max_tokens,
                            "temperature": config.temperature if config.temperature is not None else 0.2,
                        },
                    )
                response = await asyncio.to_thread(_call_py)

            content = response.get("message", {}).get("content", "")
            model_used = response.get("model", config.model or self.default_model)
            return LLMResponse(
                content=content,
                model_used=model_used,
                provider=self.provider_type.value,
                usage=None,
                finish_reason=None,
                metadata={"raw": {k: v for k, v in response.items() if k != "message"}},
            )
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_initialized():
            raise RuntimeError("Ollama provider not initialized")

        if config is None:
            config = LLMConfig(stream=True)
        else:
            config.stream = True

        if not self.validate_config(config):
            raise ValueError("Invalid configuration parameters")

        # Ollama's Python client supports streaming via generate with stream=True
        try:
            import asyncio
            if self._use_cli:
                # CLI streaming not standardized; emit single chunk
                def _call_cli_once() -> str:
                    prompt = "\n\n".join(m.content for m in messages)
                    model = config.model or self.default_model
                    completed = subprocess.run(
                        ["ollama", "run", model, prompt],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=self.default_timeout,
                    )
                    return completed.stdout
                text = await asyncio.to_thread(_call_cli_once)
                yield text
            else:
                def _iter():
                    return self._ollama.generate(
                        model=config.model or self.default_model,
                        prompt="\n\n".join(m.content for m in messages if m.role in ("system", "user")),
                        stream=True,
                        options={
                            "num_predict": config.max_tokens or self.default_max_tokens,
                            "temperature": config.temperature if config.temperature is not None else 0.2,
                        },
                    )
                iterator = await asyncio.to_thread(_iter)
                for chunk in iterator:
                    if isinstance(chunk, dict):
                        token = chunk.get("response") or ""
                    else:
                        token = str(chunk)
                    if token:
                        yield token
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    async def close(self):
        # No persistent connection to close for Ollama
        self._is_initialized = False
        logger.info("Ollama provider closed")


