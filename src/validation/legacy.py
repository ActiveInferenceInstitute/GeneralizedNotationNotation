#!/usr/bin/env python3
"""
Validation Legacy module for GNN Processing Pipeline.

This module provides legacy compatibility functions.
"""

from typing import Dict, Any

def semantic_validator(*args, **kwargs):
    """
    Legacy function name compatibility for semantic validation.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Semantic validation result
    """
    from .semantic_validator import process_semantic_validation
    return process_semantic_validation(*args, **kwargs)

def performance_profiler(*args, **kwargs):
    """
    Legacy function name compatibility for performance profiling.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Performance profiling result
    """
    from .performance_profiler import profile_performance
    return profile_performance(*args, **kwargs)

def consistency_checker(*args, **kwargs):
    """
    Legacy function name compatibility for consistency checking.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Consistency checking result
    """
    from .consistency_checker import check_consistency
    return check_consistency(*args, **kwargs)

def validate_semantic_fallback(content: str) -> Dict[str, Any]:
    """
    Fallback semantic validation when main validator is not available.
    
    Args:
        content: Content to validate
        
    Returns:
        Fallback validation result
    """
    return {
        "valid": True,
        "warnings": ["Semantic validation not available - using fallback"],
        "errors": [],
        "semantic_score": 0.8,
        "fallback": True
    }

def profile_performance_fallback(content: str) -> Dict[str, Any]:
    """
    Fallback performance profiling when main profiler is not available.
    
    Args:
        content: Content to profile
        
    Returns:
        Fallback performance result
    """
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
    """
    Fallback consistency checking when main checker is not available.
    
    Args:
        content: Content to check
        
    Returns:
        Fallback consistency result
    """
    return {
        "consistent": True,
        "consistency_score": 0.8,
        "inconsistencies": [],
        "warnings": ["Consistency checking not available - using fallback"],
        "fallback": True
    }
