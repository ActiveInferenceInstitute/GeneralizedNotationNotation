"""
Type Checker Module

This module provides functionality for validating GNN files to ensure 
they adhere to the specification and are correctly typed.
"""

from .checker import (
    GNNTypeChecker,
    TypeCheckResult,
    check_gnn_file,
    validate_syntax,
    estimate_resources
)
from .cli import main
from .resource_estimator import GNNResourceEstimator

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN type checking and validation"

# Feature availability flags
FEATURES = {
    'type_checking': True,
    'syntax_validation': True,
    'resource_estimation': True,
    'strict_mode': True,
    'report_generation': True
}

# Main API functions
__all__ = [
    'GNNTypeChecker',
    'TypeCheckResult',
    'check_gnn_file',
    'validate_syntax',
    'estimate_resources',
    'main',
    'GNNResourceEstimator',
    'FEATURES',
    '__version__'
]


def get_module_info():
    """Get comprehensive information about the type checker module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'validation_capabilities': [],
        'supported_checks': []
    }
    
    # Validation capabilities
    info['validation_capabilities'].extend([
        'Required sections validation',
        'State space variable checking',
        'Connection consistency validation',
        'Time specification validation',
        'Equation syntax checking',
        'Version and flags validation'
    ])
    
    # Supported checks
    info['supported_checks'].extend([
        'Type checking',
        'Syntax validation',
        'Resource estimation',
        'Model complexity analysis',
        'Parameterization analysis'
    ])
    
    return info


def get_validation_options() -> dict:
    """Get information about available validation options."""
    return {
        'validation_modes': {
            'strict': 'Enforce strict type checking rules',
            'lenient': 'Allow some flexibility in type checking',
            'basic': 'Basic validation only'
        },
        'check_types': {
            'sections': 'Check for required sections',
            'variables': 'Validate variable declarations',
            'connections': 'Check connection consistency',
            'equations': 'Validate equation syntax',
            'time_spec': 'Check time specifications'
        },
        'output_formats': {
            'json': 'JSON structured output',
            'markdown': 'Markdown report',
            'html': 'HTML report',
            'summary': 'Summary only'
        }
    } 