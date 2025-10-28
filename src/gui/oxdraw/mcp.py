"""
MCP Tools for oxdraw Integration

Registers Model Context Protocol tools for visual GNN model editing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

# Import core functions
from .processor import (
    process_oxdraw,
    check_oxdraw_installed,
    launch_oxdraw_editor,
    get_module_info
)
from .mermaid_converter import (
    convert_gnn_file_to_mermaid,
    gnn_to_mermaid
)
from .mermaid_parser import (
    convert_mermaid_file_to_gnn,
    mermaid_to_gnn
)


def register_mcp_tools():
    """
    Register oxdraw MCP tools.
    
    Tools:
    - oxdraw.convert_to_mermaid: Convert GNN file to Mermaid
    - oxdraw.convert_from_mermaid: Convert Mermaid back to GNN
    - oxdraw.launch_editor: Launch interactive oxdraw editor
    - oxdraw.check_installation: Check oxdraw CLI availability
    - oxdraw.get_info: Get module information
    """
    tools = []
    
    # Tool 1: Convert GNN to Mermaid
    tools.append({
        "name": "oxdraw.convert_to_mermaid",
        "description": "Convert a GNN Active Inference model to Mermaid flowchart format for visual editing in oxdraw",
        "input_schema": {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to GNN markdown file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output path for Mermaid file"
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include GNN metadata for bidirectional sync",
                    "default": True
                }
            },
            "required": ["gnn_file_path"]
        },
        "handler": tool_convert_to_mermaid
    })
    
    # Tool 2: Convert Mermaid to GNN
    tools.append({
        "name": "oxdraw.convert_from_mermaid",
        "description": "Convert Mermaid flowchart edited in oxdraw back to GNN format",
        "input_schema": {
            "type": "object",
            "properties": {
                "mermaid_file_path": {
                    "type": "string",
                    "description": "Path to Mermaid (.mmd) file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output path for GNN file"
                },
                "validate_ontology": {
                    "type": "boolean",
                    "description": "Validate ontology term mappings",
                    "default": False
                }
            },
            "required": ["mermaid_file_path"]
        },
        "handler": tool_convert_from_mermaid
    })
    
    # Tool 3: Launch oxdraw editor
    tools.append({
        "name": "oxdraw.launch_editor",
        "description": "Launch interactive oxdraw editor for visual GNN model construction",
        "input_schema": {
            "type": "object",
            "properties": {
                "mermaid_file_path": {
                    "type": "string",
                    "description": "Path to Mermaid file to edit"
                },
                "port": {
                    "type": "integer",
                    "description": "Port for oxdraw server",
                    "default": 5151
                },
                "host": {
                    "type": "string",
                    "description": "Host address for oxdraw server",
                    "default": "127.0.0.1"
                }
            },
            "required": ["mermaid_file_path"]
        },
        "handler": tool_launch_editor
    })
    
    # Tool 4: Check installation
    tools.append({
        "name": "oxdraw.check_installation",
        "description": "Check if oxdraw CLI is installed and available",
        "input_schema": {
            "type": "object",
            "properties": {}
        },
        "handler": tool_check_installation
    })
    
    # Tool 5: Get module info
    tools.append({
        "name": "oxdraw.get_info",
        "description": "Get oxdraw integration module information and capabilities",
        "input_schema": {
            "type": "object",
            "properties": {}
        },
        "handler": tool_get_info
    })
    
    return tools


def tool_convert_to_mermaid(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler: Convert GNN to Mermaid.
    
    Args:
        args: Tool arguments
        
    Returns:
        Tool result dictionary
    """
    try:
        gnn_file_path = Path(args["gnn_file_path"])
        output_path = Path(args.get("output_path")) if args.get("output_path") else None
        include_metadata = args.get("include_metadata", True)
        
        if not gnn_file_path.exists():
            return {
                "success": False,
                "error": f"GNN file not found: {gnn_file_path}"
            }
        
        mermaid_content = convert_gnn_file_to_mermaid(
            gnn_file_path,
            output_path,
            include_metadata=include_metadata
        )
        
        return {
            "success": True,
            "gnn_file": str(gnn_file_path),
            "output_file": str(output_path) if output_path else None,
            "mermaid_content": mermaid_content,
            "lines": len(mermaid_content.split('\n'))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def tool_convert_from_mermaid(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler: Convert Mermaid to GNN.
    
    Args:
        args: Tool arguments
        
    Returns:
        Tool result dictionary
    """
    try:
        mermaid_file_path = Path(args["mermaid_file_path"])
        output_path = Path(args.get("output_path")) if args.get("output_path") else None
        validate_ontology = args.get("validate_ontology", False)
        
        if not mermaid_file_path.exists():
            return {
                "success": False,
                "error": f"Mermaid file not found: {mermaid_file_path}"
            }
        
        gnn_model = convert_mermaid_file_to_gnn(
            mermaid_file_path,
            output_path,
            validate_ontology=validate_ontology
        )
        
        return {
            "success": True,
            "mermaid_file": str(mermaid_file_path),
            "output_file": str(output_path) if output_path else None,
            "model_name": gnn_model.get("model_name"),
            "variables": len(gnn_model.get("variables", {})),
            "connections": len(gnn_model.get("connections", []))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def tool_launch_editor(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler: Launch oxdraw editor.
    
    Args:
        args: Tool arguments
        
    Returns:
        Tool result dictionary
    """
    try:
        mermaid_file_path = Path(args["mermaid_file_path"])
        port = args.get("port", 5151)
        host = args.get("host", "127.0.0.1")
        
        if not check_oxdraw_installed():
            return {
                "success": False,
                "error": "oxdraw CLI not installed. Install with: cargo install oxdraw"
            }
        
        if not mermaid_file_path.exists():
            return {
                "success": False,
                "error": f"Mermaid file not found: {mermaid_file_path}"
            }
        
        success = launch_oxdraw_editor(
            mermaid_file=mermaid_file_path,
            port=port,
            host=host,
            background=True
        )
        
        if success:
            return {
                "success": True,
                "editor_url": f"http://{host}:{port}",
                "mermaid_file": str(mermaid_file_path)
            }
        else:
            return {
                "success": False,
                "error": "Failed to launch oxdraw editor"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def tool_check_installation(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler: Check oxdraw installation.
    
    Args:
        args: Tool arguments (unused)
        
    Returns:
        Tool result dictionary
    """
    installed = check_oxdraw_installed()
    
    return {
        "success": True,
        "oxdraw_installed": installed,
        "install_command": "cargo install oxdraw" if not installed else None
    }


def tool_get_info(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler: Get module info.
    
    Args:
        args: Tool arguments (unused)
        
    Returns:
        Tool result dictionary
    """
    info = get_module_info()
    
    return {
        "success": True,
        **info
    }

