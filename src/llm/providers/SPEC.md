# Specification: Providers

## Design Requirements
This module (`providers`) maps structural logic to the overall execution graph.
It ensures that `Providers` tasks resolve without runtime dependency loops.

## Components
Expected available types: BaseLLMProvider, LLMConfig, LLMMessage, LLMResponse, OllamaProvider, OpenAIProvider, OpenRouterProvider, PerplexityProvider, ProviderType
