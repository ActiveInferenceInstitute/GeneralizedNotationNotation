#!/usr/bin/env python3
"""
Model Registry Legacy module for GNN Processing Pipeline.

This module provides legacy compatibility functions.
"""

def registry(*args, **kwargs):
    """
    Legacy function name compatibility for model registry operations.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Model registry operation result
    """
    from .registry import process_model_registry
    return process_model_registry(*args, **kwargs)
