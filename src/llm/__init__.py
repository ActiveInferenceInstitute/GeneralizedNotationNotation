"""
LLM Module for GNN Pipeline

This module provides Large Language Model operations for analyzing
and processing GNN files using multiple provider APIs.
"""

# Core processor and types
from .llm_processor import (
    LLMProcessor,
    GNNLLMProcessor,
    AnalysisType,
    initialize_global_processor,
    create_processor_from_env,
    get_processor,
    close_global_processor,
    load_api_keys_from_env,
    get_default_provider_configs,
    get_preferred_providers_from_env,
    create_gnn_llm_processor,
    analyze_gnn_with_llm,
    explain_gnn_model,
    analyze_gnn_model,
    generate_explanation,
    enhance_model
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

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "LLM-powered GNN analysis and enhancement"

# Feature availability flags
FEATURES = {
    'multi_provider_support': True,
    'gnn_analysis': True,
    'model_explanation': True,
    'enhancement_suggestions': True,
    'model_validation': True,
    'model_comparison': True,
    'streaming_responses': True
}

# Main API functions
__all__ = [
    # Core processor
    'LLMProcessor',
    'GNNLLMProcessor',
    'AnalysisType',
    'initialize_global_processor',
    'create_processor_from_env',
    'get_processor',
    'close_global_processor',
    
    # GNN-specific functions
    'create_gnn_llm_processor',
    'analyze_gnn_with_llm',
    'explain_gnn_model',
    'analyze_gnn_model',
    'generate_explanation',
    'enhance_model',
    
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
    'load_api_key',
    
    # Metadata
    'FEATURES',
    '__version__'
]


def get_module_info():
    """Get comprehensive information about the LLM module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'analysis_types': [],
        'supported_providers': []
    }
    
    # Analysis types
    info['analysis_types'].extend([
        'Model summary generation',
        'Structure analysis',
        'Enhancement suggestions',
        'Model validation',
        'Model comparison',
        'Natural language explanation'
    ])
    
    # Supported providers
    info['supported_providers'].extend(['OpenAI', 'OpenRouter', 'Perplexity'])
    
    return info


def get_analysis_options() -> dict:
    """Get information about available analysis options."""
    return {
        'analysis_types': {
            'summary': 'Generate model summary',
            'structure': 'Analyze model structure',
            'enhancement': 'Suggest improvements',
            'validation': 'Validate model correctness',
            'comparison': 'Compare multiple models',
            'questions': 'Generate questions about model'
        },
        'provider_options': {
            'openai': 'OpenAI GPT models',
            'openrouter': 'OpenRouter API access',
            'perplexity': 'Perplexity AI models'
        },
        'output_formats': {
            'text': 'Plain text output',
            'json': 'Structured JSON output',
            'markdown': 'Markdown formatted output'
        },
        'processing_modes': {
            'synchronous': 'Synchronous processing',
            'asynchronous': 'Asynchronous processing',
            'streaming': 'Streaming responses'
        }
    } 