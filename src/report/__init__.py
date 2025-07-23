#!/usr/bin/env python3
"""
Report Generation Module

This module provides comprehensive analysis and reporting capabilities for the GNN pipeline.
It generates unified reports from all pipeline outputs including HTML, Markdown, and JSON formats.
"""

from .generator import generate_comprehensive_report
from .analyzer import collect_pipeline_data, analyze_step_directory
from .formatters import generate_html_report, generate_markdown_report

__all__ = [
    'generate_comprehensive_report',
    'collect_pipeline_data', 
    'analyze_step_directory',
    'generate_html_report',
    'generate_markdown_report'
] 