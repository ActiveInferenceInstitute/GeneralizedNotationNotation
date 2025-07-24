"""
Template Step Module

This module provides a standardized template for all pipeline steps.
It serves as a foundation for creating new pipeline steps with consistent structure.
"""

__version__ = "1.0.0"
__author__ = "GNN Pipeline Team"
__description__ = "Standardized template for GNN pipeline steps"

# Export main functionality
from .processor import (
    process_template_standardized,
    process_single_file,
    validate_file
)

# Version information
VERSION_INFO = {
    "version": __version__,
    "name": "Template Step",
    "description": __description__,
    "author": __author__
}

def get_version_info():
    """Get version information for the template step."""
    return VERSION_INFO 