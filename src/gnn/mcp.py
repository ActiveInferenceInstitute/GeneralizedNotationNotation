"""
MCP (Model Context Protocol) integration for GNN core documentation.

This module exposes GNN documentation files through the Model Context Protocol.
"""

import os
from pathlib import Path
from typing import Dict, Any, Literal

# MCP Tools for GNN Documentation Module

def get_gnn_documentation(doc_name: Literal["file_structure", "punctuation"]) -> Dict[str, Any]:
    """
    Retrieve content of a GNN documentation file.
    
    Args:
        doc_name: Name of the GNN document to retrieve. 
                  Allowed values: "file_structure", "punctuation".
        
    Returns:
        Dictionary containing document content or an error.
    """
    
    base_path = Path(__file__).parent
    
    if doc_name == "file_structure":
        file_to_read = base_path / "gnn_file_structure.md"
    elif doc_name == "punctuation":
        file_to_read = base_path / "gnn_punctuation.md"
    else:
        return {
            "success": False,
            "error": f"Invalid document name: {doc_name}. Allowed: 'file_structure', 'punctuation'."
        }
        
    if not file_to_read.exists():
        return {
            "success": False,
            "error": f"Documentation file not found: {file_to_read.name}"
        }
        
    try:
        content = file_to_read.read_text()
        return {
            "success": True,
            "doc_name": doc_name,
            "content": content
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading {file_to_read.name}: {str(e)}"
        }

# Resource retrievers
def _retrieve_gnn_doc_resource(uri: str) -> Dict[str, Any]:
    """
    Retrieve GNN documentation resource by URI.
    Example URI: gnn://documentation/file_structure
    """
    if not uri.startswith("gnn://documentation/"):
        raise ValueError(f"Invalid URI format for GNN documentation: {uri}")
    
    doc_name_part = uri.replace("gnn://documentation/", "")
    
    if doc_name_part not in ["file_structure", "punctuation"]:
        raise ValueError(f"Invalid document name in URI: {doc_name_part}")
        
    # Type casting for Literal
    doc_name_literal = doc_name_part # type: Literal["file_structure", "punctuation"]
    
    result = get_gnn_documentation(doc_name=doc_name_literal)
    if not result["success"]:
        raise ValueError(f"Failed to retrieve document {doc_name_part}: {result.get('error', 'Unknown error')}")
    return result


# MCP Registration Function
def register_tools(mcp_instance): # Changed 'mcp' to 'mcp_instance' for clarity
    """Register GNN documentation tools and resources with the MCP."""
    
    mcp_instance.register_tool(
        "get_gnn_documentation",
        get_gnn_documentation,
        {
            "doc_name": {
                "type": "string", 
                "description": "Name of the GNN document (e.g., 'file_structure', 'punctuation')",
                "enum": ["file_structure", "punctuation"] # Added enum for better schema
            }
        },
        "Retrieve the content of a GNN core documentation file (e.g., syntax, file structure)."
    )
    
    mcp_instance.register_resource(
        "gnn://documentation/{doc_name}", # Using a more specific URI template
        _retrieve_gnn_doc_resource,
        "Access GNN core documentation files like syntax and file structure definitions."
    ) 