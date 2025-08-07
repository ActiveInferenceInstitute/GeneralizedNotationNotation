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
    'setup_environment',  # Alias for setup_uv_environment
    'install_dependencies',  # Alias for install_uv_dependencies
    'validate_system',
    'get_environment_info',
    
    # Metadata
    'FEATURES',
    '__version__'
]


def validate_system() -> Dict[str, Any]:
    """
    Validate the system requirements for GNN with UV support.
    
    Returns:
        Dictionary with system validation results
    """
    try:
        from .setup import check_system_requirements, check_uv_availability
        return {
            "success": check_system_requirements() and check_uv_availability(),
            "message": "System validation completed with UV support"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_module_info():
    """Get comprehensive information about the setup module and its UV capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'setup_capabilities': [],
        'system_requirements': [],
        'uv_features': []
    }
    
    # UV-based setup capabilities
    info['setup_capabilities'].extend([
        'UV environment creation and management',
        'pyproject.toml-based dependency management',
        'Lock file generation and management',
        'System requirement validation',
        'JAX installation and testing',
        'Project structure initialization',
        'Environment cleanup',
        'Optional dependency group installation'
    ])
    
    # UV-specific features
    info['uv_features'].extend([
        'Fast dependency resolution',
        'Lock file for reproducible builds',
        'Built-in virtual environment management',
        'pyproject.toml support',
        'Optional dependency groups',
        'Development dependency management'
    ])
    
    # System requirements
    info['system_requirements'].extend([
        'Python 3.9+',
        'UV package manager',
        '1GB+ disk space',
        'Internet connection for package downloads'
    ])
    
    return info


def get_setup_options() -> dict:
    """Get information about available UV-based setup options."""
    return {
        'setup_modes': {
            'basic': 'Basic setup with core dependencies (uv sync)',
            'full': 'Full setup with all dependencies (uv sync --all-extras)',
            'development': 'Development setup with dev dependencies (uv sync --extra dev)',
            'minimal': 'Minimal setup for testing (uv sync --no-dev)'
        },
        'environment_options': {
            'create_new': 'Create new UV environment (uv init)',
            'use_existing': 'Use existing UV environment',
            'recreate': 'Recreate UV environment (rm -rf .venv && uv init)'
        },
        'dependency_groups': {
            'core': 'Core GNN processing dependencies (default)',
            'dev': 'Development and testing dependencies',
            'ml-ai': 'Machine Learning & AI dependencies',
            'llm': 'LLM Integration dependencies',
            'visualization': 'Advanced visualization dependencies',
            'audio': 'Audio processing dependencies',
            'graphs': 'Graph visualization dependencies',
            'research': 'Research tools dependencies',
            'all': 'All optional dependencies'
        },
        'validation_levels': {
            'basic': 'Basic system requirement checks',
            'comprehensive': 'Comprehensive validation including JAX',
            'strict': 'Strict validation with all requirements'
        },
        'uv_commands': {
            'sync': 'uv sync - Install dependencies from pyproject.toml',
            'add': 'uv add <package> - Add new dependency',
            'remove': 'uv remove <package> - Remove dependency',
            'run': 'uv run <command> - Run command in UV environment',
            'lock': 'uv lock - Generate lock file',
            'init': 'uv init - Initialize new UV project'
        }
    }


def get_environment_info() -> Dict[str, Any]:
    """Get environment info for the GNN setup with UV support."""
    from .setup import get_uv_setup_info
    return get_uv_setup_info()


def get_uv_status() -> Dict[str, Any]:
    """
    Get comprehensive UV status information.
    
    Returns:
        Dictionary with UV status details
    """
    try:
        from .setup import check_uv_availability
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent.parent
        status = {
            "uv_available": check_uv_availability(),
            "project_root": str(project_root),
            "pyproject_toml_exists": (project_root / "pyproject.toml").exists(),
            "uv_lock_exists": (project_root / "uv.lock").exists(),
            "venv_exists": (project_root / ".venv").exists(),
            "python_version": None,
            "installed_packages": {}
        }
        
        # Get Python version if venv exists
        if status["venv_exists"]:
            venv_python = project_root / ".venv" / "bin" / "python"
            if venv_python.exists():
                import subprocess
                result = subprocess.run(
                    [str(venv_python), "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    status["python_version"] = result.stdout.strip()
        
        return status
        
    except Exception as e:
        return {
            "error": str(e),
            "uv_available": False
        }

def setup_environment(*args, **kwargs):
    """Alias for setup_uv_environment for backward compatibility."""
    from .setup import setup_uv_environment
    return setup_uv_environment(*args, **kwargs)

def install_dependencies(*args, **kwargs):
    """Alias for install_uv_dependencies for backward compatibility."""
    from .setup import install_uv_dependencies
    return install_uv_dependencies(*args, **kwargs) 