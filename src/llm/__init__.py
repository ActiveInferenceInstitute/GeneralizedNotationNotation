"""
LLM module for GNN Processing Pipeline.

This module provides LLM-enhanced analysis and processing for GNN models.
It avoids importing heavy optional dependencies at import time; functions
are provided as thin wrappers that import implementations on first use.
"""

__version__ = "1.6.0"

from .defaults import DEFAULT_OLLAMA_MODEL

FEATURES = {
    "openai_integration": True,
    "anthropic_integration": True,
    "ollama_integration": True,
    "multi_provider_support": True,
    "model_analysis": True,
    "structured_prompting": True,
    "mcp_integration": True
}

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def process_llm(*args: Any, **kwargs: Any) -> bool:
    """Delegate to processor.process_llm — returns True on success."""
    from .processor import process_llm as _impl
    return _impl(*args, **kwargs)

# Phase 6: llm submodules are in-tree; fallback shims removed as dead code.
# If any import here fails, it's a real bug that must surface in CI — not be
# silently papered over.
from .llm_processor import (
    AnalysisType,
    create_processor_from_env,
    get_default_provider_configs,
    get_preferred_providers_from_env,
    initialize_global_processor,
    load_api_keys_from_env,
)
from .llm_processor import (
    LLMProcessor as UnifiedLLMProcessor,
)
from .llm_processor import (
    get_processor as get_global_processor,
)

from .providers import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    ProviderType,
)

from .analyzer import (
    analyze_gnn_file_with_llm,
    calculate_complexity_metrics,
    extract_connections,
    extract_sections,
    extract_variables,
    identify_patterns,
    perform_semantic_analysis,
)

from .generator import (
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary,
    generate_model_insights,
)


class LLMProcessor:
    """Minimal processor facade exposing methods expected by tests."""

    def analyze(self, content: str) -> Dict[str, Any]:
        return {
            "variables": extract_variables(content),
            "connections": extract_connections(content),
            "sections": extract_sections(content),
        }

    # Methods expected by tests
    def analyze_model(self, model_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        text = ""
        if isinstance(model_data, dict):
            text = model_data.get("content", "")
        elif isinstance(model_data, str):
            text = model_data
        return self.analyze(text)

    def generate_description(self, content: str) -> str:
        variables = [v.get("name", "var") for v in extract_variables(content)]
        return f"Model with {len(variables)} variables and {len(extract_connections(content))} connections"


class LLMAnalyzer:
    """Simple analyzer class exposing analysis helpers expected by tests."""

    def analyze_content(self, content: str) -> Dict[str, Any]:
        return {
            "variables": extract_variables(content),
            "connections": extract_connections(content),
            "sections": extract_sections(content),
            "patterns": identify_patterns(content, extract_variables(content), extract_connections(content)),
        }

    def extract_insights(self, content: str) -> Dict[str, Any]:
        return perform_semantic_analysis(content, extract_variables(content), extract_connections(content))


def get_module_info() -> Dict[str, Any]:
    """Return basic module info required by tests."""
    return {
        "version": __version__,
        "description": "LLM-enhanced analysis utilities for GNN",
        "features": FEATURES,
        "providers": get_available_providers(),
    }


def analyze_gnn_model(model_content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compatibility helper exposing synchronous analysis expected by tests."""
    # Prefer analyzer class if available
    try:
        analyzer = LLMAnalyzer()
        return analyzer.analyze_content(model_content if isinstance(model_content, str) else model_content.get('content',''))
    except Exception:
        return {
            'variables': extract_variables(model_content),
            'connections': extract_connections(model_content)
        }


def generate_model_description(content: str) -> str:
    """Generate a natural language description of a GNN model from its content."""
    proc = LLMProcessor()
    return proc.generate_description(content)


def get_available_providers() -> list:
    """Return a list of available provider identifiers (best-effort)."""
    providers = ["ollama"]
    try:
        # Importing lazily to avoid heavy deps
        from .providers import openai_provider as _openai  # noqa: F401
        providers.append("openai")
    except ImportError:
        logger.debug("openai provider not installed, skipping")
    try:
        from .providers import openrouter_provider as _openrouter  # noqa: F401
        providers.append("openrouter")
    except ImportError:
        logger.debug("openrouter provider not installed, skipping")
    return providers


__all__ = [
    'process_llm',
    'analyze_gnn_file_with_llm',
    'extract_variables',
    'extract_connections',
    'extract_sections',
    'perform_semantic_analysis',
    'calculate_complexity_metrics',
    'identify_patterns',
    'generate_model_insights',
    'generate_code_suggestions',
    'generate_documentation',
    'generate_llm_summary',
    'LLMProcessor',
    'LLMAnalyzer',
    'get_module_info',
    'get_available_providers',
    'AnalysisType',
    'UnifiedLLMProcessor',
    'load_api_keys_from_env',
    'ProviderType',
    'initialize_global_processor',
    'get_global_processor',
    'create_processor_from_env',
    'get_default_provider_configs',
    'get_preferred_providers_from_env',
    'LLMConfig', 'LLMMessage', 'LLMResponse', 'BaseLLMProvider',
    'DEFAULT_OLLAMA_MODEL',
    '__version__'
]
