"""
MCP (Model Context Protocol) integration for GNN Visualization module.

This module exposes GNN visualization functionality through the Model Context Protocol.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from .visualizer import GNNVisualizer
from .parser import GNNParser

# MCP Tools for Visualization Module

def visualize_file(file_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize a GNN file through MCP.
    
    Args:
        file_path: Path to the GNN file to visualize
        output_dir: Optional output directory to save visualizations
        
    Returns:
        Dictionary containing visualization results
    """
    try:
        visualizer = GNNVisualizer(output_dir=output_dir)
        output_path = visualizer.visualize_file(file_path)
        
        return {
            "success": True,
            "output_directory": output_path,
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

def visualize_directory(dir_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize all GNN files in a directory through MCP.
    
    Args:
        dir_path: Path to directory containing GNN files
        output_dir: Optional output directory to save visualizations
        
    Returns:
        Dictionary containing visualization results
    """
    try:
        visualizer = GNNVisualizer(output_dir=output_dir)
        output_path = visualizer.visualize_directory(dir_path)
        
        return {
            "success": True,
            "output_directory": output_path,
            "directory_path": dir_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory_path": dir_path
        }

def parse_gnn_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a GNN file without visualization through MCP.
    
    Args:
        file_path: Path to the GNN file to parse
        
    Returns:
        Dictionary containing parsed GNN data
    """
    try:
        parser = GNNParser()
        parsed_data = parser.parse_file(file_path)
        
        # Convert to serializable format
        serializable_data = {}
        for k, v in parsed_data.items():
            if k not in ['Variables', 'Edges']:  # Skip complex objects
                serializable_data[k] = str(v)
            else:
                serializable_data[k] = f"{len(v)} items"
        
        return {
            "success": True,
            "parsed_data": serializable_data,
            "file_path": file_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

# Resource retrievers

def get_visualization_results(uri: str) -> Dict[str, Any]:
    """
    Retrieve visualization results by URI.
    
    Args:
        uri: URI of the visualization results. Format: visualization://{output_directory}
        
    Returns:
        Dictionary containing visualization results
    """
    # Extract directory path from URI
    if not uri.startswith("visualization://"):
        raise ValueError(f"Invalid URI format: {uri}")
    
    dir_path = uri[16:]  # Remove 'visualization://' prefix
    dir_path = Path(dir_path)
    
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    # Collect visualization files
    visualization_files = []
    for file_path in dir_path.glob("*"):
        if file_path.is_file():
            visualization_files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size
            })
    
    return {
        "directory": str(dir_path),
        "files": visualization_files
    }

# MCP Registration Function

def register_tools(mcp):
    """Register visualization tools with the MCP."""
    
    # Register visualization tools
    mcp.register_tool(
        "visualize_gnn_file",
        visualize_file,
        {
            "file_path": {"type": "string", "description": "Path to the GNN file to visualize"},
            "output_dir": {"type": "string", "description": "Optional output directory"}
        },
        "Generate visualizations for a specific GNN file."
    )
    
    mcp.register_tool(
        "visualize_gnn_directory",
        visualize_directory,
        {
            "dir_path": {"type": "string", "description": "Path to directory containing GNN files"},
            "output_dir": {"type": "string", "description": "Optional output directory"}
        },
        "Visualize all GNN files in a directory"
    )
    
    mcp.register_tool(
        "parse_gnn_file",
        parse_gnn_file,
        {
            "file_path": {"type": "string", "description": "Path to the GNN file to parse"}
        },
        "Parse a GNN file without visualization"
    )
    
    # Register visualization resources
    mcp.register_resource(
        "visualization://{output_directory}",
        get_visualization_results,
        "Retrieve visualization results by output directory"
    ) 