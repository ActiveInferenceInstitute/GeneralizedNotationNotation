"""
Utility functions for the GNN Processing Pipeline.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def ensure_directory(directory: str or Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path (string or Path object)
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_gnn_files(directory: str or Path, recursive: bool = False) -> List[Path]:
    """
    Find all GNN (.md) files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects for GNN files
    """
    path = Path(directory)
    if not path.exists():
        return []
        
    pattern = "**/*.md" if recursive else "*.md"
    return list(path.glob(pattern))


def get_output_paths(base_output_dir: str or Path) -> Dict[str, Path]:
    """
    Get standard output paths for the pipeline.
    
    Args:
        base_output_dir: Base output directory
        
    Returns:
        Dictionary of named output paths
    """
    base_dir = ensure_directory(base_output_dir)
    
    # Create standard subdirectories
    paths = {
        "base": base_dir,
        "type_check": ensure_directory(base_dir / "gnn_type_check"),
        "visualization": ensure_directory(base_dir / "gnn_examples_visualization")
    }
    
    return paths 