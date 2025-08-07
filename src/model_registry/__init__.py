"""
Model Registry Module

This module provides model versioning, registry management, and metadata handling
for GNN model specifications.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

# Import core registry functionality
from .registry import ModelRegistry, process_model_registry

# Import legacy wrapper functions
from .legacy import (
    registry
)

# Re-export main classes and functions
__all__ = [
    'ModelRegistry',
    'registry',
    'process_model_registry'
]
