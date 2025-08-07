"""
Security module for GNN Processing Pipeline.

This module provides security validation and access control for GNN models.
"""

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
    'generate_security_summary'
]
