"""
LLM Providers Module

This module contains implementations for different LLM providers:
- OpenAI: Original OpenAI GPT models and services
- OpenRouter: Unified access to multiple AI models
- Perplexity: AI-powered search and reasoning
"""

from .base_provider import (
    BaseLLMProvider,
    ProviderType,
    LLMResponse,
    LLMMessage,
    LLMConfig,
)

# Lazy accessors to avoid importing heavy/optional dependencies at module import time
def get_openai_provider_class():
    from .openai_provider import OpenAIProvider
    return OpenAIProvider

def get_openrouter_provider_class():
    from .openrouter_provider import OpenRouterProvider
    return OpenRouterProvider

def get_perplexity_provider_class():
    from .perplexity_provider import PerplexityProvider
    return PerplexityProvider

def get_ollama_provider_class():
    from .ollama_provider import OllamaProvider
    return OllamaProvider

__all__ = [
    'BaseLLMProvider',
    'ProviderType',
    'LLMResponse',
    'LLMMessage',
    'LLMConfig',
    'get_openai_provider_class',
    'get_openrouter_provider_class',
    'get_perplexity_provider_class',
    'get_ollama_provider_class',
]
