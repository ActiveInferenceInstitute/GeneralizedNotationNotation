"""
Testing module for GNN

This module contains test files and performance benchmarks.
"""

# Testing module exports - simplified to avoid dependency issues

# Import only essential classes without heavy dependencies
try:
    from .test_round_trip import ComprehensiveTestReport, RoundTripResult
    from .simple_round_trip_test import SimpleTestResult
    
    __all__ = [
        'ComprehensiveTestReport', 
        'RoundTripResult',
        'SimpleTestResult'
    ]
    
    # Try to import GNNRoundTripTester separately
    try:
        from .test_round_trip import GNNRoundTripTester
        __all__.append('GNNRoundTripTester')
    except ImportError:
        pass
    
    # Note: Performance benchmarks temporarily disabled due to numpy dependency issues
    
except ImportError as e:
    # Fallback if some imports fail
    __all__ = []
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Some testing imports failed: {e}")
