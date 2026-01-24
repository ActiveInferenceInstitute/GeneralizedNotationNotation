"""
Intelligent Analysis module for GNN Processing Pipeline.

This module provides intelligent AI-powered analysis of pipeline execution results,
including failure analysis, performance bottleneck identification, per-step analysis
with yellow/red flag detection, and executive report generation using LLM infrastructure.
"""

__version__ = "2.0.0"
FEATURES = {
    "pipeline_analysis": True,
    "failure_root_cause": True,
    "performance_optimization": True,
    "llm_powered_insights": True,
    "executive_reports": True,
    "mcp_integration": True,
    "per_step_analysis": True,
    "yellow_red_flags": True,
    "rule_based_fallback": True
}

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

# Import processor functions and classes
from .processor import (
    process_intelligent_analysis,
    analyze_pipeline_summary,
    analyze_individual_steps,
    generate_executive_report,
    identify_bottlenecks,
    extract_failure_context,
    generate_recommendations,
    StepAnalysis
)

# Import analyzer functions
from .analyzer import (
    IntelligentAnalyzer,
    AnalysisContext,
    calculate_pipeline_health_score,
    classify_failure_severity,
    detect_performance_patterns,
    generate_optimization_suggestions
)


def get_module_info() -> Dict[str, Any]:
    """Get module information."""
    return {
        "version": __version__,
        "description": "Intelligent AI-powered pipeline analysis",
        "features": list(FEATURES.keys()),
        "report_formats": ["markdown", "json", "html"],
        "llm_backends": ["openai", "anthropic", "local"]
    }


def get_supported_analysis_types() -> List[str]:
    """Get list of supported analysis types."""
    return [
        "failure_analysis",
        "performance_analysis",
        "optimization_recommendations",
        "executive_summary",
        "trend_analysis",
        "comparative_analysis"
    ]


def validate_pipeline_summary(summary: Dict[str, Any]) -> bool:
    """Validate that a pipeline summary has the required structure."""
    required_fields = [
        "start_time",
        "steps",
        "overall_status"
    ]
    return all(field in summary for field in required_fields)


def check_analysis_tools() -> Dict[str, Dict[str, Any]]:
    """Check availability of analysis tools and LLM backends."""
    tools = {}

    # Check LLM processor
    try:
        from llm.llm_processor import get_processor
        tools['llm_processor'] = {
            'available': True,
            'description': 'LLM processor for AI-powered analysis'
        }
    except ImportError:
        tools['llm_processor'] = {
            'available': False,
            'description': 'LLM processor not available'
        }

    # Check numpy for statistics
    try:
        import numpy
        tools['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        tools['numpy'] = {'available': False, 'version': None}

    # Check pandas for data analysis
    try:
        import pandas
        tools['pandas'] = {
            'available': True,
            'version': pandas.__version__
        }
    except ImportError:
        tools['pandas'] = {'available': False, 'version': None}

    return tools


__all__ = [
    # Module info
    '__version__',
    'FEATURES',
    'get_module_info',
    'get_supported_analysis_types',
    'validate_pipeline_summary',
    'check_analysis_tools',
    # Processor functions and classes
    'process_intelligent_analysis',
    'analyze_pipeline_summary',
    'analyze_individual_steps',
    'generate_executive_report',
    'identify_bottlenecks',
    'extract_failure_context',
    'generate_recommendations',
    'StepAnalysis',
    # Analyzer classes and functions
    'IntelligentAnalyzer',
    'AnalysisContext',
    'calculate_pipeline_health_score',
    'classify_failure_severity',
    'detect_performance_patterns',
    'generate_optimization_suggestions'
]
