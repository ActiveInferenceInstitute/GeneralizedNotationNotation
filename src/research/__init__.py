"""
research module for GNN Processing Pipeline.

This module provides research capabilities with fallback implementations.
"""

# Import processor functions - single source of truth
from .processor import process_research

# Module metadata
__version__ = "1.1.3"
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
