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

# Import legacy wrapper functions
from .legacy import (
    semantic_validator,
    performance_profiler,
    consistency_checker,
    validate_semantic_fallback,
    profile_performance_fallback,
    check_consistency_fallback
)

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
    'check_consistency',
    'validate_semantic_fallback',
    'profile_performance_fallback',
    'check_consistency_fallback'
]
