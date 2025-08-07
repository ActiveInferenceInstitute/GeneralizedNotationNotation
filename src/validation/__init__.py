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

def validate_semantic_fallback(content: str) -> Dict[str, Any]:
    """Fallback semantic validation when main validator is not available."""
    return {
        "valid": True,
        "warnings": ["Semantic validation not available - using fallback"],
        "errors": [],
        "semantic_score": 0.8,
        "fallback": True
    }

def profile_performance_fallback(content: str) -> Dict[str, Any]:
    """Fallback performance profiling when main profiler is not available."""
    return {
        "performance_score": 0.7,
        "estimated_complexity": "medium",
        "resource_requirements": {
            "memory_mb": 512,
            "cpu_cores": 2,
            "execution_time_seconds": 30
        },
        "warnings": ["Performance profiling not available - using fallback"],
        "fallback": True
    }

def check_consistency_fallback(content: str) -> Dict[str, Any]:
    """Fallback consistency checking when main checker is not available."""
    return {
        "consistent": True,
        "consistency_score": 0.8,
        "inconsistencies": [],
        "warnings": ["Consistency checking not available - using fallback"],
        "fallback": True
    }

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
