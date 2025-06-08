"""
Utility functions for the GNN Processing Pipeline.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
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
        "type_check": ensure_directory(base_dir / "gnn_type_check"),
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