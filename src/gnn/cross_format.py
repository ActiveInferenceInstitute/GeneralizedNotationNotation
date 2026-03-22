#!/usr/bin/env python3
"""
GNN Cross-Format Validation Strategy Module

This module provides an alias for the cross-format validation strategy
to maintain consistency with the modular architecture.
"""

# Import the existing CrossFormatValidator and related classes
try:
    from .cross_format_validator import (
        CrossFormatValidator,
    )

except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Cross-format validation not available: {e}")

    # Recovery implementation
    class CrossFormatValidator:
        def __init__(self):
            pass

        def configure(self, **kwargs):
            pass

        def validate(self, files):
            return {
                'success': False,
                'error': 'Cross-format validation not available',
                'files_validated': 0
            }
