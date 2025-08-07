"""
report module for GNN Processing Pipeline.

This module provides report capabilities with fallback implementations.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

# Import processor functions
from .processor import (
    process_report,
    generate_comprehensive_report,
    analyze_gnn_file,
    generate_html_report,
    generate_markdown_report
)

__all__ = [
    # Processor functions
    'process_report',
    'generate_comprehensive_report',
    'analyze_gnn_file',
    'generate_html_report',
    'generate_markdown_report'
]
