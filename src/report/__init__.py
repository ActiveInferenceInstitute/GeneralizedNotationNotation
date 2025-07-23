#!/usr/bin/env python3
"""
Report Generation Module

This module provides comprehensive analysis and reporting capabilities for the GNN pipeline.
It generates unified reports from all pipeline outputs including HTML, Markdown, and JSON formats.
"""

from .generator import (
    generate_comprehensive_report,
    generate_html_report_file,
    generate_markdown_report_file,
    generate_json_report_file,
    generate_custom_report,
    validate_report_data
)
from .analyzer import (
    collect_pipeline_data, 
    analyze_step_directory,
    analyze_file_types_across_steps,
    analyze_step_dependencies,
    analyze_errors,
    get_pipeline_health_score
)
from .formatters import (
    generate_html_report, 
    generate_markdown_report,
    get_health_color
)
from .mcp import (
    generate_pipeline_report,
    analyze_pipeline_data,
    generate_custom_report as mcp_generate_custom_report,
    get_report_module_info,
    register_tools
)

__all__ = [
    # Generator functions
    'generate_comprehensive_report',
    'generate_html_report_file',
    'generate_markdown_report_file', 
    'generate_json_report_file',
    'generate_custom_report',
    'validate_report_data',
    
    # Analyzer functions
    'collect_pipeline_data',
    'analyze_step_directory',
    'analyze_file_types_across_steps',
    'analyze_step_dependencies',
    'analyze_errors',
    'get_pipeline_health_score',
    
    # Formatter functions
    'generate_html_report',
    'generate_markdown_report',
    'get_health_color',
    
    # MCP functions
    'generate_pipeline_report',
    'analyze_pipeline_data',
    'mcp_generate_custom_report',
    'get_report_module_info',
    'register_tools'
] 