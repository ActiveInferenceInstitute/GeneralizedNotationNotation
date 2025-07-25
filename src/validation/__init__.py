"""
Validation Module

This module provides comprehensive validation capabilities for GNN models,
including semantic validation, performance profiling, and consistency checking.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

# Import core validation functionality
from .semantic_validator import SemanticValidator, process_semantic_validation
from .performance_profiler import PerformanceProfiler, profile_performance
from .consistency_checker import ConsistencyChecker, check_consistency

# Export the missing functions that scripts are looking for
def semantic_validator(*args, **kwargs):
    """Legacy function name compatibility for semantic validation."""
    return process_semantic_validation(*args, **kwargs)

def performance_profiler(*args, **kwargs):
    """Legacy function name compatibility for performance profiling."""
    return profile_performance(*args, **kwargs)

def consistency_checker(*args, **kwargs):
    """Legacy function name compatibility for consistency checking."""
    return check_consistency(*args, **kwargs)

# Re-export main classes and functions
__all__ = [
    'SemanticValidator',
    'PerformanceProfiler', 
    'ConsistencyChecker',
    'semantic_validator',
    'performance_profiler',
    'consistency_checker',
    'process_semantic_validation',
    'profile_performance',
    'check_consistency'
]
