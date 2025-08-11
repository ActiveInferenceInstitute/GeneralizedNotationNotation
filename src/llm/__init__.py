"""
LLM module for GNN Processing Pipeline.

This module provides LLM-enhanced analysis and processing for GNN models.
"""

from .processor import process_llm
from .analyzer import (
    analyze_gnn_file_with_llm,
    extract_variables,
    extract_connections,
    extract_sections,
    perform_semantic_analysis,
    calculate_complexity_metrics,
    identify_patterns
)
from .generator import (
    generate_model_insights,
    generate_code_suggestions,
    generate_documentation,
    generate_llm_summary
)

__version__ = "1.0.0"


class LLMProcessor:
    """Minimal processor facade exposing methods expected by tests."""

    def analyze(self, content: str) -> dict:
        return {
            "variables": extract_variables(content),
            "connections": extract_connections(content),
            "sections": extract_sections(content),
        }

    # Methods expected by tests
    def analyze_model(self, model_data: dict) -> dict:
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

    def analyze_content(self, content: str) -> dict:
        return {
            "variables": extract_variables(content),
            "connections": extract_connections(content),
            "sections": extract_sections(content),
            "patterns": identify_patterns(content, extract_variables(content), extract_connections(content)),
        }

    def extract_insights(self, content: str) -> dict:
        return perform_semantic_analysis(content, extract_variables(content), extract_connections(content))


def get_module_info() -> dict:
    """Return basic module info required by tests."""
    return {
        "version": __version__,
        "description": "LLM-enhanced analysis utilities for GNN",
        "providers": get_available_providers(),
    }


def get_available_providers() -> list:
    """Return a list of available provider identifiers (best-effort)."""
    providers = ["ollama"]
    try:
        # Importing lazily to avoid heavy deps
        from .providers import openai_provider as _openai  # noqa: F401
        providers.append("openai")
    except Exception:
        pass
    try:
        from .providers import openrouter_provider as _openrouter  # noqa: F401
        providers.append("openrouter")
    except Exception:
        pass
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
    '__version__'
]
