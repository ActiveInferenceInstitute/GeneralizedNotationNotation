"""
Security module for GNN Processing Pipeline.

This module provides security validation and access control for GNN models.
"""

__version__ = "1.1.3"
FEATURES = {
    "vulnerability_detection": True,
    "security_scoring": True,
    "access_control": True,
    "security_recommendations": True,
    "mcp_integration": False  # No mcp.py exists
}

# Import processor functions - single source of truth
from .processor import (
    process_security,
    perform_security_check,
    check_vulnerabilities,
    generate_security_recommendations,
    calculate_security_score,
    generate_security_summary
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
