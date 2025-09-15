#!/usr/bin/env python3
"""
MCP (Model Context Protocol) integration for the Type Checker module.

This module provides MCP tool registration and implementation for type checking
functionality, enabling external tools and applications to interact with the
type checker through the Model Context Protocol.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .processor import GNNTypeChecker
from .analysis_utils import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity
)

logger = logging.getLogger(__name__)

# MCP Tool Definitions
MCP_TOOLS = [
    {
        "name": "validate_gnn_file",
        "description": "Validate a single GNN file for type consistency and syntax errors",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the GNN file to validate"
                },
                "strict_mode": {
                    "type": "boolean",
                    "description": "Enable strict validation mode",
                    "default": False
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output",
                    "default": False
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "analyze_variable_types",
        "description": "Analyze variable types and dimensions in a GNN model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "description": "List of variable dictionaries to analyze",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "data_type": {"type": "string"},
                            "dimensions": {"type": "array", "items": {"type": "integer"}}
                        }
                    }
                }
            },
            "required": ["variables"]
        }
    },
    {
        "name": "analyze_connections",
        "description": "Analyze connection patterns and complexity in a GNN model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "connections": {
                    "type": "array",
                    "description": "List of connection dictionaries to analyze",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "source_variables": {"type": "array", "items": {"type": "string"}},
                            "target_variables": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "required": ["connections"]
        }
    },
    {
        "name": "estimate_complexity",
        "description": "Estimate computational complexity for inference and learning",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type_analysis": {
                    "type": "object",
                    "description": "Results from variable type analysis"
                },
                "connection_analysis": {
                    "type": "object",
                    "description": "Results from connection analysis"
                }
            },
            "required": ["type_analysis", "connection_analysis"]
        }
    },
    {
        "name": "validate_gnn_directory",
        "description": "Validate all GNN files in a directory",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to directory containing GNN files"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save validation results"
                },
                "strict_mode": {
                    "type": "boolean",
                    "description": "Enable strict validation mode",
                    "default": False
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output",
                    "default": False
                }
            },
            "required": ["directory_path", "output_path"]
        }
    },
    {
        "name": "get_type_checker_info",
        "description": "Get information about the type checker module and its capabilities",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def register_mcp_tools() -> List[Dict[str, Any]]:
    """
    Register MCP tools for the type checker module.
    
    Returns:
        List of MCP tool definitions
    """
    logger.info("Registering MCP tools for type checker module")
    return MCP_TOOLS.copy()


def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an MCP tool for type checking functionality.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        
    Returns:
        Tool execution result
        
    Raises:
        ValueError: If tool name is not recognized
        TypeError: If arguments are invalid
    """
    logger.info(f"Executing MCP tool: {tool_name}")
    
    try:
        if tool_name == "validate_gnn_file":
            return _execute_validate_gnn_file(arguments)
        elif tool_name == "analyze_variable_types":
            return _execute_analyze_variable_types(arguments)
        elif tool_name == "analyze_connections":
            return _execute_analyze_connections(arguments)
        elif tool_name == "estimate_complexity":
            return _execute_estimate_complexity(arguments)
        elif tool_name == "validate_gnn_directory":
            return _execute_validate_gnn_directory(arguments)
        elif tool_name == "get_type_checker_info":
            return _execute_get_type_checker_info(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
            
    except Exception as e:
        logger.error(f"Error executing MCP tool {tool_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_validate_gnn_file(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute validate_gnn_file tool."""
    file_path = Path(arguments["file_path"])
    strict_mode = arguments.get("strict_mode", False)
    verbose = arguments.get("verbose", False)
    
    if not file_path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        checker = GNNTypeChecker(strict_mode=strict_mode, verbose=verbose)
        result = checker.validate_single_gnn_file(file_path, verbose)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_analyze_variable_types(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute analyze_variable_types tool."""
    variables = arguments["variables"]
    
    try:
        result = analyze_variable_types(variables)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_analyze_connections(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute analyze_connections tool."""
    connections = arguments["connections"]
    
    try:
        result = analyze_connections(connections)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_estimate_complexity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute estimate_complexity tool."""
    type_analysis = arguments["type_analysis"]
    connection_analysis = arguments["connection_analysis"]
    
    try:
        result = estimate_computational_complexity(type_analysis, connection_analysis)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_validate_gnn_directory(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute validate_gnn_directory tool."""
    directory_path = Path(arguments["directory_path"])
    output_path = Path(arguments["output_path"])
    strict_mode = arguments.get("strict_mode", False)
    verbose = arguments.get("verbose", False)
    
    if not directory_path.exists():
        return {
            "success": False,
            "error": f"Directory not found: {directory_path}",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        checker = GNNTypeChecker(strict_mode=strict_mode, verbose=verbose)
        success = checker.validate_gnn_files(directory_path, output_path, verbose)
        
        return {
            "success": success,
            "result": {
                "directory_path": str(directory_path),
                "output_path": str(output_path),
                "validation_successful": success
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }


def _execute_get_type_checker_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute get_type_checker_info tool."""
    return {
        "success": True,
        "result": {
            "module_name": "type_checker",
            "version": "1.0.0",
            "description": "Comprehensive type checking and validation for GNN files",
            "capabilities": [
                "GNN file validation",
                "Variable type analysis",
                "Connection pattern analysis",
                "Computational complexity estimation",
                "Resource requirement estimation",
                "Performance analysis"
            ],
            "supported_file_types": [".md", ".gnn", ".txt"],
            "validation_modes": ["standard", "strict"],
            "available_tools": [tool["name"] for tool in MCP_TOOLS],
            "timestamp": datetime.now().isoformat()
        },
        "timestamp": datetime.now().isoformat()
    }


def get_mcp_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the schema for a specific MCP tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool schema or None if not found
    """
    for tool in MCP_TOOLS:
        if tool["name"] == tool_name:
            return tool
    return None


def list_available_tools() -> List[str]:
    """
    List all available MCP tools.
    
    Returns:
        List of tool names
    """
    return [tool["name"] for tool in MCP_TOOLS]


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate arguments for an MCP tool.
    
    Args:
        tool_name: Name of the tool
        arguments: Arguments to validate
        
    Returns:
        Validation result with success status and any errors
    """
    tool_schema = get_mcp_tool_schema(tool_name)
    if not tool_schema:
        return {
            "valid": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    input_schema = tool_schema.get("inputSchema", {})
    required_fields = input_schema.get("required", [])
    
    # Check required fields
    missing_fields = []
    for field in required_fields:
        if field not in arguments:
            missing_fields.append(field)
    
    if missing_fields:
        return {
            "valid": False,
            "error": f"Missing required fields: {missing_fields}"
        }
    
    # Basic type validation
    properties = input_schema.get("properties", {})
    for field, value in arguments.items():
        if field in properties:
            expected_type = properties[field].get("type")
            if expected_type and not isinstance(value, _get_python_type(expected_type)):
                return {
                    "valid": False,
                    "error": f"Field '{field}' should be of type {expected_type}, got {type(value).__name__}"
                }
    
    return {
        "valid": True,
        "message": "Arguments are valid"
    }


def _get_python_type(json_type: str) -> type:
    """Convert JSON schema type to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict
    }
    return type_mapping.get(json_type, object)


# MCP Tool Registry
MCP_TOOL_REGISTRY = {
    "validate_gnn_file": _execute_validate_gnn_file,
    "analyze_variable_types": _execute_analyze_variable_types,
    "analyze_connections": _execute_analyze_connections,
    "estimate_complexity": _execute_estimate_complexity,
    "validate_gnn_directory": _execute_validate_gnn_directory,
    "get_type_checker_info": _execute_get_type_checker_info,
}


if __name__ == "__main__":
    # Test MCP integration
    print("Type Checker MCP Tools:")
    for tool in MCP_TOOLS:
        print(f"- {tool['name']}: {tool['description']}")
    
    print(f"\nTotal tools available: {len(MCP_TOOLS)}")