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
    'generate_model_description',
    'validate_model_structure',
    'enhance_model_parameters',
    'generate_llm_report',
    
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


# Test-compatible function alias
def generate_model_description(gnn_file_path, **kwargs):
    """Generate a description of a GNN model (test-compatible alias)."""
    return explain_gnn_model(gnn_file_path, **kwargs)

def validate_model_structure(gnn_file_path, **kwargs):
    """Validate model structure (test-compatible alias)."""
    return analyze_gnn_model(gnn_file_path, **kwargs)

def enhance_model_parameters(gnn_file_path, **kwargs):
    """Enhance model parameters (test-compatible alias)."""
    return enhance_model(gnn_file_path, **kwargs)

def generate_llm_report(llm_results, output_path=None):
    """Generate LLM report (test-compatible alias)."""
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "llm_results": llm_results,
        "summary": {
            "total_analyses": len(llm_results) if isinstance(llm_results, list) else 1,
            "successful_analyses": sum(1 for r in llm_results if r.get('success', False)) if isinstance(llm_results, list) else (1 if llm_results.get('success', False) else 0)
        }
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report 