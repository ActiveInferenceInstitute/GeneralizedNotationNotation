#!/usr/bin/env python3
"""
Setup Utils module for GNN Processing Pipeline.

This module provides setup utility functions.
"""

from typing import Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_directory(directory_path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to ensure
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        return directory_path
    except Exception as e:
        logger.error(f"Failed to ensure directory {directory_path}: {e}")
        return False

def find_gnn_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Find GNN files in the specified directory.
    
    Args:
        directory: Directory to search for GNN files
        
    Returns:
        List of GNN file paths found
    """
    gnn_files = []
    try:
        iterator = directory.rglob("*.md") if recursive else directory.glob("*.md")
        for file_path in iterator:
            if file_path.is_file():
                gnn_files.append(file_path)
    except Exception as e:
        logger.error(f"Failed to find GNN files in {directory}: {e}")
    
    return gnn_files

def get_output_paths(base_output_dir: Path) -> Dict[str, Path]:
    """
    Get standard output paths for the pipeline.
    
    Args:
        base_output_dir: Base output directory
        
    Returns:
        Dictionary of output paths
    """
    return {
        "setup": base_output_dir / "setup_artifacts",
        "tests": base_output_dir / "test_results",
        "gnn": base_output_dir / "gnn_output",
        "validation": base_output_dir / "validation_results",
        "export": base_output_dir / "export_results",
        "visualization": base_output_dir / "visualization_results",
        "ontology": base_output_dir / "ontology_results",
        "render": base_output_dir / "render_results",
        "execute": base_output_dir / "execute_results",
        "llm": base_output_dir / "llm_results",
        "ml_integration": base_output_dir / "ml_integration_results",
        "audio": base_output_dir / "audio_results",
        "analysis": base_output_dir / "analysis_results",
        "integration": base_output_dir / "integration_results",
        "security": base_output_dir / "security_results",
        "research": base_output_dir / "research_results",
        "website": base_output_dir / "website_results",
        "report": base_output_dir / "report_results",
        "mcp": base_output_dir / "mcp_results"
    }

def get_module_info():
    """
    Get comprehensive information about the setup module and its UV capabilities.
    
    Returns:
        Dictionary with module information
    """
    return {
        'version': "2.0.0",
        'description': "GNN environment setup and management with UV",
        'features': {
            'uv_environment_setup': True,
            'uv_dependency_management': True,
            'uv_virtual_environment': True,
            'system_validation': True,
            'project_initialization': True,
            'jax_installation': True,
            'mcp_integration': True,
            'pyproject_toml_support': True,
            'lock_file_management': True
        },
        'setup_capabilities': [
            'UV environment setup',
            'Dependency management',
            'Virtual environment creation',
            'System validation',
            'Project initialization',
            'JAX installation',
            'MCP integration',
            'PyProject.toml support',
            'Lock file management'
        ],
        'processing_methods': [
            'UV installation',
            'Environment validation',
            'Dependency resolution',
            'Project structure creation'
        ],
        'processing_capabilities': [
            'Setup UV environment',
            'Install dependencies',
            'Validate system requirements',
            'Create project structure',
            'Manage virtual environments',
            'Handle lock files'
        ],
        'supported_formats': [
            'PyProject.toml',
            'requirements.txt',
            'poetry.lock',
            'uv.lock'
        ]
    }

def get_setup_options() -> dict:
    """
    Get setup options and configuration.
    
    Returns:
        Dictionary with setup options
    """
    return {
        'environment_types': ['uv', 'venv', 'conda', 'pip'],
        'python_versions': ['3.8', '3.9', '3.10', '3.11', '3.12'],
        'dependency_sources': ['pyproject.toml', 'requirements.txt', 'poetry.lock'],
        'setup_modes': ['minimal', 'standard', 'full', 'development'],
        'validation_levels': ['basic', 'comprehensive', 'strict'],
        'installation_methods': ['uv', 'pip', 'conda', 'poetry'],
        'project_templates': ['basic', 'advanced', 'research', 'production'],
        'output_formats': ['json', 'yaml', 'toml', 'markdown']
    }

def setup_environment(*args, **kwargs):
    """
    Setup environment (alias for setup_uv_environment).
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Setup result
    """
    try:
        from .setup import setup_uv_environment
        return setup_uv_environment(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        return False

def install_dependencies(*args, **kwargs):
    """
    Install dependencies using UV.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Installation result
    """
    try:
        from .setup import install_uv_dependencies
        return install_uv_dependencies(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_uv_project_status(project_root: Path) -> Dict[str, Any]:
    """
    Check the status of a UV project.

    Args:
        project_root: Path to project root directory

    Returns:
        Dictionary with UV project status information
    """
    try:
        from .setup import validate_uv_setup
        status = validate_uv_setup(project_root=project_root)
        return status
    except Exception as e:
        logger.error(f"Failed to check UV project status: {e}")
        return {
            "success": False,
            "error": str(e),
            "system_requirements": False,
            "uv_environment": False,
            "dependencies": False,
            "jax_installation": False,
            "overall_status": False
        }

def get_uv_environment_info() -> Dict[str, Any]:
    """
    Get information about the current UV environment.

    Returns:
        Dictionary with UV environment information
    """
    try:
        from .setup import get_uv_setup_info
        env_info = get_uv_setup_info()
        return env_info
    except Exception as e:
        logger.error(f"Failed to get UV environment info: {e}")
        return {
            "success": False,
            "error": str(e),
            "environment_exists": False,
            "python_version": None,
            "dependencies_installed": False
        }

def setup_uv_project_structure(base_path: Path) -> bool:
    """
    Set up UV project structure.

    Args:
        base_path: Base path for project structure

    Returns:
        True if setup successful, False otherwise
    """
    try:
        from .setup import create_project_structure
        success = create_project_structure(base_path, logger)
        return success
    except Exception as e:
        logger.error(f"Failed to setup UV project structure: {e}")
        return False 