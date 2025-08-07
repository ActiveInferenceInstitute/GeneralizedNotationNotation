"""
Utility functions for the GNN Processing Pipeline with UV support.

This module provides utilities for UV-based environment management,
dependency handling, and project structure setup.
"""

import os
import sys
import platform
import subprocess
import json
import time
from pathlib import Path
import logging
try:
    from ..pipeline import get_output_dir_for_script
    from ..utils import log_step_start, log_step_success, log_step_warning, log_step_error
except ImportError:
    # Fallback for when running as standalone module
    def get_output_dir_for_script(script_name, base_output_dir):
        return Path(base_output_dir) / script_name.replace('.py', '')
    
    def log_step_start(logger, message):
        logger.info(f"🚀 {message}")
    
    def log_step_success(logger, message):
        logger.info(f"✅ {message}")
    
    def log_step_warning(logger, message):
        logger.warning(f"⚠️ {message}")
    
    def log_step_error(logger, message):
        logger.error(f"❌ {message}")
from typing import List, Dict, Tuple, Optional, Union, Any

def is_uv_environment_active() -> bool:
    """
    Check if running in a UV-managed environment.
    
    Returns:
        bool: True if running inside a UV environment, False otherwise.
    """
    # Check if we're in a UV-managed virtual environment
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_uv_environment_info() -> Dict[str, Union[bool, str, Path, None]]:
    """
    Get information about the current UV environment.
    
    Returns:
        Dict with the following keys:
            - is_active: Whether a UV environment is active
            - venv_path: Path to the venv if active, None otherwise
            - python_executable: Path to the Python executable in the venv
            - uv_executable: Path to the UV executable
    """
    result = {
        "is_active": is_uv_environment_active(),
        "venv_path": None,
        "python_executable": None,
        "uv_executable": None
    }
    
    if result["is_active"]:
        # Get venv path from sys.prefix
        result["venv_path"] = Path(sys.prefix)
        
        # Get Python executable path
        result["python_executable"] = Path(sys.executable)
        
        # Find UV executable
        try:
            uv_result = subprocess.run(
                ["which", "uv"] if platform.system() != "Windows" else ["where", "uv"],
                capture_output=True,
                text=True
            )
            if uv_result.returncode == 0:
                result["uv_executable"] = Path(uv_result.stdout.strip())
        except Exception:
            pass
    
    return result

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path (string or Path object)
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory creation fails or path exists but is not a directory
    """
    path = Path(directory)
    
    if path.exists() and not path.is_dir():
        raise OSError(f"Path exists but is not a directory: {path}")
    
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {path}: {e}")
    
    return path.resolve()

def find_gnn_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Find all GNN (.md) files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects for GNN files
        
    Raises:
        FileNotFoundError: If the specified directory does not exist
        ValueError: If the path exists but is not a directory
    """
    path = Path(directory)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
        
    pattern = "**/*.md" if recursive else "*.md"
    files = list(path.glob(pattern))
    
    # Verify these are GNN files, not just any Markdown files
    # Simple heuristic: Check if they contain GNN-specific headers
    gnn_files = []
    gnn_markers = ["ModelName:", "StateSpaceBlock:", "GNNVersionAndFlags:"]
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(4000)  # Read first 4KB to check headers
                
                # Check if any GNN marker is present
                if any(marker in content for marker in gnn_markers):
                    gnn_files.append(file_path)
        except Exception as e:
            # Just log and skip problematic files
            print(f"Warning: Could not read {file_path}: {e}")
    
    return gnn_files

def get_output_paths(base_output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Get standard output paths for the pipeline.
    
    Args:
        base_output_dir: Base output directory
        
    Returns:
        Dictionary of named output paths
        
    Raises:
        OSError: If directory creation fails
    """
    base_dir = ensure_directory(base_output_dir)
    
    # Create standard subdirectories
    paths = {
        "base": base_dir,
                    "type_check": ensure_directory(base_dir / "type_check"),
        "visualization": ensure_directory(base_dir / "visualization"),
        "exports": ensure_directory(base_dir / "gnn_exports"),
        "rendered": ensure_directory(base_dir / "gnn_rendered_simulators"),
        "logs": ensure_directory(base_dir / "logs"),
        "test_reports": ensure_directory(base_dir / "test_reports")
    }
    
    return paths

def check_uv_dependencies() -> Dict[str, bool]:
    """
    Check if required UV dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to boolean values indicating availability
    """
    result = {}
    
    # Check for UV itself
    try:
        uv_result = subprocess.run(
            ["uv", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        result["uv"] = uv_result.returncode == 0
    except Exception:
        result["uv"] = False
    
    # Check for Python packages via UV
    for package in ["pip", "venv"]:
        try:
            subprocess.run(
                [sys.executable, "-m", package, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            result[package] = True
        except Exception:
            result[package] = False
    
    # Check for external tools
    for tool in ["graphviz"]:
        try:
            process = subprocess.run(
                ["which", tool] if platform.system() != "Windows" else ["where", tool],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            result[tool] = process.returncode == 0
        except Exception:
            result[tool] = False
    
    return result 

def log_environment_info(logger):
    """Log comprehensive environment information including UV status."""
    import platform
    log_step_start(logger, "Logging environment information")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Operating system: {platform.platform()}")
    
    # Check UV availability
    try:
        uv_result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        if uv_result.returncode == 0:
            logger.info(f"UV version: {uv_result.stdout.strip()}")
        else:
            logger.warning("UV not available")
    except Exception as e:
        logger.warning(f"Could not check UV version: {e}")
    
    # Check if we're in a UV environment
    env_info = get_uv_environment_info()
    logger.info(f"UV environment active: {env_info['is_active']}")
    if env_info['venv_path']:
        logger.info(f"UV environment path: {env_info['venv_path']}")
    
    log_step_success(logger, "Environment information logged successfully")

def verify_directories(target_dir, output_dir, logger, find_gnn_files, verbose=False):
    """Verify directories and create output structure using UV standards."""
    log_step_start(logger, f"Verifying directories - target: {target_dir}, output: {output_dir}")
    target_path = Path(target_dir)
    if not target_path.is_dir():
        log_step_error(logger, f"Target directory '{target_dir}' does not exist or is not a directory")
        return False
    try:
        logger.debug(f"Searching for GNN files in {target_path}...")
        gnn_files = find_gnn_files(target_path, recursive=True)
        logger.debug(f"Found {len(gnn_files)} GNN .md files (recursively in target: {target_path.resolve()})")
        if not gnn_files and verbose:
            log_step_warning(logger, f"No GNN files found in {target_path}. This might be expected if you're planning to create them later.")
    except Exception as e:
        log_step_error(logger, f"Error scanning for GNN files in {target_path}: {e}")
    
    # Create output directory structure using centralized configuration
    try:
        logger.info("Creating output directory structure...")
        # Ensure output directory exists first
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the actual output paths using the centralized function
        output_paths = get_output_paths(output_dir)
        logger.info(f"Output directory structure verified/created: {output_path.resolve()}")
        
        structure_info = {name: str(path) for name, path in output_paths.items()}
        structure_file = output_path / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_info, f, indent=2)
        logger.debug(f"Directory structure info saved to: {structure_file}")
        log_step_success(logger, "Directory structure created successfully")
        return True
    except Exception as e:
        log_step_error(logger, f"Error creating output directories: {e}")
        return False

def list_installed_packages_uv(logger, output_dir=None, verbose=False):
    """Generate installed packages report using UV."""
    log_step_start(logger, "Generating installed packages report with UV")
    
    try:
        # Use UV to list packages
        uv_list_cmd = ["uv", "pip", "list", "--format=json"]
        result = subprocess.run(
            uv_list_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        packages = json.loads(result.stdout)
        package_dict = {pkg["name"]: pkg["version"] for pkg in packages}
        logger.info(f"Found {len(package_dict)} installed packages in the UV environment")
        
        if verbose:
            logger.info("Installed packages:")
            for name, version in sorted(package_dict.items()):
                logger.info(f"  - {name}: {version}")
        
        if output_dir:
            setup_dir = get_output_dir_for_script("1_setup.py", Path(output_dir))
            setup_dir.mkdir(parents=True, exist_ok=True)
            packages_file = setup_dir / "installed_packages_uv.json"
            json_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "uv_environment": str(get_uv_environment_info().get("venv_path")),
                "python_executable": str(get_uv_environment_info().get("python_executable")),
                "packages": package_dict
            }
            with open(packages_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.debug(f"Package list saved to: {packages_file}")
        
        log_step_success(logger, f"Successfully listed {len(package_dict)} packages with UV")
        
    except Exception as e:
        log_step_error(logger, f"Unexpected error listing packages with UV: {e}")

def check_uv_project_status(project_root: Path) -> Dict[str, Any]:
    """
    Check the status of a UV project.
    
    Args:
        project_root: Path to the project root
        
    Returns:
        Dictionary with project status information
    """
    status = {
        "project_root": str(project_root),
        "pyproject_toml_exists": False,
        "uv_lock_exists": False,
        "venv_exists": False,
        "uv_available": False,
        "python_version": None,
        "dependencies_installed": False
    }
    
    # Check if pyproject.toml exists
    pyproject_path = project_root / "pyproject.toml"
    status["pyproject_toml_exists"] = pyproject_path.exists()
    
    # Check if uv.lock exists
    lock_path = project_root / "uv.lock"
    status["uv_lock_exists"] = lock_path.exists()
    
    # Check if .venv exists
    venv_path = project_root / ".venv"
    status["venv_exists"] = venv_path.exists()
    
    # Check UV availability
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True
        )
        status["uv_available"] = result.returncode == 0
    except Exception:
        status["uv_available"] = False
    
    # Check Python version if venv exists
    if status["venv_exists"]:
        python_path = venv_path / "bin" / "python"
        if sys.platform == "win32":
            python_path = venv_path / "Scripts" / "python.exe"
        
        if python_path.exists():
            try:
                result = subprocess.run(
                    [str(python_path), "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    status["python_version"] = result.stdout.strip()
            except Exception:
                pass
    
    # Check if dependencies are installed
    if status["venv_exists"] and status["uv_available"]:
        try:
            result = subprocess.run(
                ["uv", "pip", "list"],
                capture_output=True,
                text=True
            )
            status["dependencies_installed"] = result.returncode == 0 and "numpy" in result.stdout
        except Exception:
            status["dependencies_installed"] = False
    
    return status

def setup_uv_project_structure(project_root: Path, logger) -> bool:
    """
    Set up a new UV project structure.
    
    Args:
        project_root: Path to the project root
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log_step_start(logger, "Setting up UV project structure")
        
        # Create basic project structure
        directories = [
            "src",
            "tests",
            "docs",
            "input/gnn_files",
            "output",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Create basic pyproject.toml if it doesn't exist
        pyproject_path = project_root / "pyproject.toml"
        if not pyproject_path.exists():
            logger.info("Creating basic pyproject.toml...")
            # This would be created by the main setup script
        
        log_step_success(logger, "UV project structure created successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Failed to setup UV project structure: {e}")
        return False 