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
    """Simple facade expected by tests to expose processing capability."""

    def analyze(self, content: str) -> dict:
        # Use analyzer functions to produce a compact analysis result
        return {
            "variables": extract_variables(content),
            "connections": extract_connections(content),
            "sections": extract_sections(content),
        }


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
    '__version__'
]
