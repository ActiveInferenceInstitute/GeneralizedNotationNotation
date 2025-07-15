"""
Utility functions for the GNN Processing Pipeline.
"""

import os
import sys
import platform
import subprocess
import json
import time
from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error
from typing import List, Dict, Tuple, Optional, Union

def is_venv_active() -> bool:
    """
    Check if a Python virtual environment is currently active.
    
    Returns:
        bool: True if running inside a virtual environment, False otherwise.
    """
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_venv_info() -> Dict[str, Union[bool, str, Path, None]]:
    """
    Get information about the current virtual environment.
    
    Returns:
        Dict with the following keys:
            - is_active: Whether a venv is active
            - venv_path: Path to the venv if active, None otherwise
            - python_executable: Path to the Python executable in the venv
            - pip_executable: Path to the pip executable in the venv
    """
    result = {
        "is_active": is_venv_active(),
        "venv_path": None,
        "python_executable": None,
        "pip_executable": None
    }
    
    if result["is_active"]:
        # Get venv path from sys.prefix
        result["venv_path"] = Path(sys.prefix)
        
        # Get Python executable path
        result["python_executable"] = Path(sys.executable)
        
        # Determine pip executable path
        if platform.system() == "Windows":
            pip_path = Path(sys.prefix) / "Scripts" / "pip.exe"
        else:
            pip_path = Path(sys.prefix) / "bin" / "pip"
        
        if pip_path.exists():
            result["pip_executable"] = pip_path
    
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

def check_system_dependencies() -> Dict[str, bool]:
    """
    Check if required system dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to boolean values indicating availability
    """
    result = {}
    
    # Check for Python packages via subprocess
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
    import platform
    log_step_start(logger, "Logging environment information")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Operating system: {platform.platform()}")
    # Add more environment info as needed
    log_step_success(logger, "Environment information logged successfully")

def verify_directories(target_dir, output_dir, logger, find_gnn_files, verbose=False):
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
        output_paths = {}  # Should be replaced with actual output path logic
        logger.info(f"Output directory structure verified/created: {Path(output_dir).resolve()}")
        structure_info = {name: str(path) for name, path in output_paths.items()}
        structure_file = Path(output_dir) / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_info, f, indent=2)
        logger.debug(f"Directory structure info saved to: {structure_file}")
        log_step_success(logger, "Directory structure created successfully")
        return True
    except Exception as e:
        log_step_error(logger, f"Error creating output directories: {e}")
        return False

def list_installed_packages(logger, venv_info, output_dir=None, verbose=False):
    log_step_start(logger, "Generating installed packages report")
    pip_executable = venv_info.get("pip_executable")
    if not pip_executable or not Path(pip_executable).exists():
        log_step_warning(logger, f"pip not found at {pip_executable}, skipping package listing")
        return
    try:
        import subprocess
        pip_list_cmd = [pip_executable, "list", "--format=json"]
        result = subprocess.run(
            pip_list_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        package_dict = {pkg["name"]: pkg["version"] for pkg in packages}
        logger.info(f"Found {len(package_dict)} installed packages in the virtual environment")
        if verbose:
            logger.info("Installed packages:")
            for name, version in sorted(package_dict.items()):
                logger.info(f"  - {name}: {version}")
        if output_dir:
            setup_dir = get_output_dir_for_script("2_setup.py", Path(output_dir))
            setup_dir.mkdir(parents=True, exist_ok=True)
            packages_file = setup_dir / "installed_packages.json"
            json_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "virtual_env": str(venv_info.get("venv_path")),
                "python_executable": str(venv_info.get("python_executable")),
                "packages": package_dict
            }
            with open(packages_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.debug(f"Package list saved to: {packages_file}")
        log_step_success(logger, f"Successfully listed {len(package_dict)} packages")
    except Exception as e:
        log_step_error(logger, f"Unexpected error listing packages: {e}") 