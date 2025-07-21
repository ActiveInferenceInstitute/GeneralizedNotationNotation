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
        CrossFormatValidationResult,
        validate_cross_format_consistency,
        validate_schema_consistency
    )
    
    # Create alias for strategy pattern
    CrossFormatValidationStrategy = CrossFormatValidator
    
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Cross-format validation not available: {e}")
    
    # Fallback implementation
    class CrossFormatValidationStrategy:
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