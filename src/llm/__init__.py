"""
LLM Module for GNN Pipeline

This module provides Large Language Model operations for analyzing
and processing GNN files using multiple provider APIs.
"""

# Core processor and types
from .llm_processor import (
    LLMProcessor,
    AnalysisType,
    initialize_global_processor,
    create_processor_from_env,
    get_processor,
    close_global_processor,
    load_api_keys_from_env,
    get_default_provider_configs,
    get_preferred_providers_from_env
)

# Provider system
from .providers import (
    BaseLLMProvider,
    ProviderType,
    LLMResponse,
    LLMMessage,
    LLMConfig,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider
)

# Legacy support
from .llm_operations import (
    LLMOperations,
    llm_ops,
    construct_prompt,
    get_llm_response,
    load_api_key
)

__all__ = [
    # Core processor
    'LLMProcessor',
    'AnalysisType',
    'initialize_global_processor',
    'create_processor_from_env',
    'get_processor',
    'close_global_processor',
    
    # Utility functions
    'load_api_keys_from_env',
    'get_default_provider_configs', 
    'get_preferred_providers_from_env',
    
    # Provider system
    'BaseLLMProvider',
    'ProviderType',
    'LLMResponse',
    'LLMMessage',
    'LLMConfig',
    'OpenAIProvider',
    'OpenRouterProvider',
    'PerplexityProvider',
    
    # Legacy support
    'LLMOperations',
    'llm_ops',
    'construct_prompt',
    'get_llm_response',
    'load_api_key'
] 