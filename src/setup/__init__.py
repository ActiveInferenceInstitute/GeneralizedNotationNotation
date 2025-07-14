"""
Setup package for GNN Processing Pipeline.

This package contains utility functions and shared resources for the pipeline.
"""

from typing import Dict, Any

from .utils import ensure_directory, find_gnn_files, get_output_paths
from .setup import (
    setup_environment,
    validate_setup,
    get_setup_info,
    cleanup_setup,
    setup_gnn_project,
    check_system_requirements,
    create_virtual_environment,
    install_dependencies,
    get_installed_package_versions
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN environment setup and management"

# Feature availability flags
FEATURES = {
    'environment_setup': True,
    'dependency_management': True,
    'virtual_environment': True,
    'system_validation': True,
    'project_initialization': True,
    'jax_installation': True,
    'mcp_integration': True
}

# Main API functions
__all__ = [
    # Utility functions
    'ensure_directory',
    'find_gnn_files',
    'get_output_paths',
    
    # Setup functions
    'setup_environment',
    'validate_setup',
    'get_setup_info',
    'cleanup_setup',
    'setup_gnn_project',
    'check_system_requirements',
    'create_virtual_environment',
    'install_dependencies',
    'get_installed_package_versions',
    'validate_system',
    'get_environment_info',
    
    # Metadata
    'FEATURES',
    '__version__'
]


def validate_system() -> Dict[str, Any]:
    """
    Validate the system requirements for GNN.
    
    Returns:
        Dictionary with system validation results
    """
    try:
        from .setup import check_system_requirements
        return {
            "success": check_system_requirements(),
            "message": "System validation completed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_module_info():
    """Get comprehensive information about the setup module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'setup_capabilities': [],
        'system_requirements': []
    }
    
    # Setup capabilities
    info['setup_capabilities'].extend([
        'Virtual environment creation',
        'Dependency installation',
        'System requirement validation',
        'JAX installation and testing',
        'Project structure initialization',
        'Environment cleanup'
    ])
    
    # System requirements
    info['system_requirements'].extend([
        'Python 3.9+',
        'pip package manager',
        'venv module',
        '1GB+ disk space'
    ])
    
    return info


def get_setup_options() -> dict:
    """Get information about available setup options."""
    return {
        'setup_modes': {
            'basic': 'Basic setup with core dependencies',
            'full': 'Full setup with all dependencies',
            'development': 'Development setup with dev dependencies',
            'minimal': 'Minimal setup for testing'
        },
        'environment_options': {
            'create_new': 'Create new virtual environment',
            'use_existing': 'Use existing virtual environment',
            'recreate': 'Recreate virtual environment'
        },
        'dependency_groups': {
            'core': 'Core GNN processing dependencies',
            'visualization': 'Visualization and plotting dependencies',
            'execution': 'Execution and simulation dependencies',
            'development': 'Development and testing dependencies'
        },
        'validation_levels': {
            'basic': 'Basic system requirement checks',
            'comprehensive': 'Comprehensive validation including JAX',
            'strict': 'Strict validation with all requirements'
        }
    } 


def get_environment_info() -> Dict[str, Any]:
    """Get environment info for the GNN setup."""
    from .setup import get_setup_info
    return get_setup_info() 