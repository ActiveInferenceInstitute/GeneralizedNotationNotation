"""
research module for GNN Processing Pipeline.

This module provides research capabilities with fallback implementations.
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
    process_research
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "research processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'basic_processing': True,
    'fallback_mode': True
}

__all__ = [
    'process_research',
    'FEATURES',
    '__version__'
]
