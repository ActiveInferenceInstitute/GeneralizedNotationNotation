"""
MCP (Model Context Protocol) integration for GNN core documentation and schema.

This module exposes GNN documentation files, schema definitions, and validation
capabilities through the Model Context Protocol.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Literal, Union
import logging

logger = logging.getLogger(__name__)

# MCP Tools for GNN Documentation and Schema Module

def get_gnn_documentation(doc_name: Literal["file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"]) -> Dict[str, Any]:
    """
    Retrieve content of a GNN documentation or schema file.
    
    Args:
        doc_name: Name of the GNN document to retrieve. 
                  Allowed values: "file_structure", "punctuation", "schema_json", "schema_yaml", "grammar".
        
    Returns:
        Dictionary containing document content or an error.
    """
    
    base_path = Path(__file__).parent
    
    file_map = {
        "file_structure": "gnn_file_structure.md",
        "punctuation": "gnn_punctuation.md",
        "schema_json": "gnn_schema.json",
        "schema_yaml": "gnn_schema.yaml", 
        "grammar": "gnn_grammar.ebnf"
    }
    
    if doc_name not in file_map:
        error_msg = f"Invalid document name: {doc_name}. Allowed: {', '.join(file_map.keys())}."
        logger.error(f"get_gnn_documentation: {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }
    
    file_to_read = base_path / file_map[doc_name]
        
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


def validate_gnn_content(content: str, validation_level: Literal["basic", "standard", "strict"] = "standard") -> Dict[str, Any]:
    """
    Validate GNN content against the schema.
    
    Args:
        content: GNN file content as string
        validation_level: Validation strictness level
        
    Returns:
        Dictionary containing validation results
    """
    try:
        from .schema_validator import GNNValidator, ValidationLevel, ValidationResult
        
        validator = GNNValidator()
        level_map = {
            "basic": ValidationLevel.BASIC,
            "standard": ValidationLevel.STANDARD,
            "strict": ValidationLevel.STRICT
        }
        
        # Create a temporary validation result
        result = ValidationResult(is_valid=True)
        validator._validate_structure(content, result)
        
        return {
            "success": True,
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "suggestions": result.suggestions,
            "validation_level": validation_level
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "Schema validator not available. Ensure schema_validator.py is properly installed."
        }
    except Exception as e:
        logger.error(f"validate_gnn_content: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Validation error: {str(e)}"
        }


def get_gnn_schema_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the GNN schema.
    
    Returns:
        Dictionary containing schema metadata and structure information
    """
    try:
        base_path = Path(__file__).parent
        
        # Load schema info from YAML if available
        yaml_path = base_path / "gnn_schema.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                schema_data = yaml.safe_load(f)
            
            return {
                "success": True,
                "schema_info": schema_data.get("schema_info", {}),
                "required_sections": schema_data.get("required_sections", []),
                "optional_sections": schema_data.get("optional_sections", []),
                "syntax_rules": schema_data.get("syntax_rules", {}),
                "active_inference_patterns": schema_data.get("active_inference", {}),
                "validation_levels": schema_data.get("validation_levels", {})
            }
        else:
            return {
                "success": False,
                "error": "Schema YAML file not found"
            }
            
    except Exception as e:
        logger.error(f"get_gnn_schema_info: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Error loading schema info: {str(e)}"
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
    
    allowed_docs = ["file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"]
    if doc_name_part not in allowed_docs:
        error_msg = f"Invalid document name in URI: {doc_name_part}"
        logger.error(f"_retrieve_gnn_doc_resource: {error_msg}")
        raise ValueError(error_msg)
        
    # Type casting for Literal
    doc_name_literal = doc_name_part # type: Literal["file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"]
    
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
                "description": "Name of the GNN document (e.g., 'file_structure', 'punctuation', 'schema_json', 'schema_yaml', 'grammar')",
                "enum": ["file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"]
            }
        },
        "Retrieve the content of a GNN documentation file, schema definition, or grammar specification."
    )
    
    mcp_instance.register_tool(
        "validate_gnn_content",
        validate_gnn_content,
        {
            "content": {
                "type": "string",
                "description": "GNN file content to validate"
            },
            "validation_level": {
                "type": "string",
                "description": "Validation strictness level",
                "enum": ["basic", "standard", "strict"],
                "default": "standard"
            }
        },
        "Validate GNN file content against the formal schema."
    )
    
    mcp_instance.register_tool(
        "get_gnn_schema_info",
        get_gnn_schema_info,
        {},
        "Get comprehensive information about the GNN schema structure and validation rules."
    )
    
    mcp_instance.register_resource(
        "gnn://documentation/{doc_name}", # Using a more specific URI template
        _retrieve_gnn_doc_resource,
        "Access GNN core documentation files like syntax and file structure definitions."
    )
    logger.info("GNN documentation module MCP tools and resources registered.") 