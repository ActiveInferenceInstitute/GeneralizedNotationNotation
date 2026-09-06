#!/usr/bin/env python3
"""
Template Utils module for GNN Processing Pipeline.

This module provides template utility functions.
"""

from typing import Any


def get_version_info() -> Any:
    """
    Get version information for the template step.

    Returns:
        Dictionary with version information
    """
    return {
        "version": "1.0.0",
        "name": "Template Step",
        "description": "Standardized template for GNN pipeline steps",
        "author": "GNN Pipeline Team",
    }
