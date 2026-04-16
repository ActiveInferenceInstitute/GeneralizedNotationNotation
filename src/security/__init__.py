"""
Security module for GNN Processing Pipeline.

This module provides security validation and access control for GNN models.
"""

__version__ = "1.6.0"
FEATURES = {
    "vulnerability_detection": True,
    "security_scoring": True,
    "access_control": True,
    "security_recommendations": True,
    "mcp_integration": False  # No mcp.py exists
}

# Import processor functions - single source of truth
from .processor import (
    calculate_security_score,
    check_vulnerabilities,
    generate_security_recommendations,
    generate_security_summary,
    perform_security_check,
    process_security,
)

__all__ = [
    'process_security',
    'perform_security_check',
    'check_vulnerabilities',
    'generate_security_recommendations',
    'calculate_security_score',
    'generate_security_summary',
    'FEATURES',
    '__version__'
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "security",
        "version": __version__,
        "description": "Security validation, vulnerability scanning, and access control",
        "features": FEATURES,
    }
