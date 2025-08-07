#!/usr/bin/env python3
"""
Execute Legacy module for GNN Processing Pipeline.

This module provides legacy compatibility functions.
"""

import logging

logger = logging.getLogger(__name__)

def execute_pymdp_simulation_from_gnn(*args, **kwargs):
    """
    Execute PyMDP simulation from GNN specification (legacy function).
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Execution results
    """
    try:
        from .pymdp import execute_pymdp_simulation_from_gnn_impl
        return execute_pymdp_simulation_from_gnn_impl(*args, **kwargs)
    except ImportError:
        logger.warning("PyMDP executor not available")
        return {"error": "PyMDP executor not available"}

def validator(*args, **kwargs):
    """
    Legacy function name for execution environment validation.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Validation results
    """
    try:
        from .validator import validate_execution_environment_impl
        return validate_execution_environment_impl(*args, **kwargs)
    except ImportError:
        logger.warning("Execution validator not available")
        return {"error": "Execution validator not available"}

def pymdp(*args, **kwargs):
    """
    Legacy function name for PyMDP execution.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        PyMDP execution results
    """
    return execute_pymdp_simulation_from_gnn(*args, **kwargs)
