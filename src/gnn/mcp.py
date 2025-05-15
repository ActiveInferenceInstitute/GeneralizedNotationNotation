"""
MCP (Model Context Protocol) integration for GNN core documentation.

This module exposes GNN documentation files through the Model Context Protocol.
"""

import os
from pathlib import Path
from typing import Dict, Any, Literal
import logging

logger = logging.getLogger(__name__)

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
        error_msg = f"Invalid document name: {doc_name}. Allowed: 'file_structure', 'punctuation'."
        logger.error(f"get_gnn_documentation: {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }
        
    if not file_to_read.exists():
        error_msg = f"Documentation file not found: {file_to_read.name} (expected at {file_to_read.resolve()})"
        logger.error(f"get_gnn_documentation: {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }
        
    try:
        content = file_to_read.read_text()
        return {
            "success": True,
            "doc_name": doc_name,
            "content": content
        }
    except Exception as e:
        error_msg = f"Error reading {file_to_read.name}: {str(e)}"
        logger.error(f"get_gnn_documentation: {error_msg}", exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }

# Resource retrievers
def _retrieve_gnn_doc_resource(uri: str) -> Dict[str, Any]:
    """
    Retrieve GNN documentation resource by URI.
    Example URI: gnn://documentation/file_structure
    """
    if not uri.startswith("gnn://documentation/"):
        error_msg = f"Invalid URI format for GNN documentation: {uri}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)
    
    doc_name_part = uri.replace("gnn://documentation/", "")
    
    if doc_name_part not in ["file_structure", "punctuation"]:
        error_msg = f"Invalid document name in URI: {doc_name_part}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)
        
    # Type casting for Literal
    doc_name_literal = doc_name_part # type: Literal["file_structure", "punctuation"]
    
    result = get_gnn_documentation(doc_name=doc_name_literal)
    if not result["success"]:
        error_msg = f"Failed to retrieve document {doc_name_part}: {result.get('error', 'Unknown error')}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)
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
    logger.info("GNN documentation module MCP tools and resources registered.") 