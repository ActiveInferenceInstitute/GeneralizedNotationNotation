"""
Setup package for GNN Processing Pipeline with UV support.

This package contains utility functions and shared resources for the pipeline,
using modern UV-based dependency management and environment setup.
"""

from typing import Dict, Any

from .utils import ensure_directory, find_gnn_files, get_output_paths
from .setup import (
    setup_uv_environment,
    validate_uv_setup,
    get_uv_setup_info,
    cleanup_uv_setup,
    setup_gnn_project,
    check_system_requirements,
    install_uv_dependencies,
    get_installed_package_versions,
    check_uv_availability,
    log_system_info,
    install_optional_dependencies,
    create_project_structure
)

# Import validator functions
from .validator import (
    validate_system,
    get_environment_info,
    get_uv_status
)

# Import utility functions
from .utils import (
    get_module_info,
    get_setup_options,
    setup_environment,
    install_dependencies
)

# Module metadata
__version__ = "2.0.0"
__author__ = "Active Inference Institute"
__description__ = "GNN environment setup and management with UV"

# Feature availability flags
FEATURES = {
    'uv_environment_setup': True,
    'uv_dependency_management': True,
    'uv_virtual_environment': True,
    'system_validation': True,
    'project_initialization': True,
    'jax_installation': True,
    'mcp_integration': True,
    'pyproject_toml_support': True,
    'lock_file_management': True
}

# Main API functions
__all__ = [
    # Utility functions
    'ensure_directory',
    'find_gnn_files',
    'get_output_paths',
    
    # UV-based setup functions
    'setup_uv_environment',
    'validate_uv_setup',
    'get_uv_setup_info',
    'cleanup_uv_setup',
    'setup_gnn_project',
    'check_system_requirements',
    'install_uv_dependencies',
    'get_installed_package_versions',
    'check_uv_availability',
    'log_system_info',
    'install_optional_dependencies',
    'create_project_structure',
    
    # Validator functions
    'validate_system',
    'get_environment_info',
    'get_uv_status',
    
    # Utility functions
    'get_module_info',
    'get_setup_options',
    'setup_environment',
    'install_dependencies',
    
    # Metadata
    'FEATURES',
    '__version__'
] 