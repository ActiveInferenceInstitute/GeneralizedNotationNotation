"""
Model Context Protocol (MCP) integration for the GNN Rendering module.

This module exposes the GNN rendering capabilities (e.g., to PyMDP, RxInfer.jl)
as tools that MCP-enabled clients (like LLMs) can consume, by registering
them with the main project MCP instance.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from collections.abc import Callable # Added import
import tempfile # For creating temporary files for rendered output

from pydantic import BaseModel, Field

# Assuming render.py is in the same directory or accessible in PYTHONPATH
from .render import render_gnn_spec # Main function from render.py

# Import from the project's actual MCP system
# Assuming this file (src/render/mcp.py) will be loaded by src/mcp/mcp.py,
# which adjusts sys.path, allowing 'from mcp import ...' or 'from src.mcp import ...'
# Based on 7_mcp.py, 'from src.mcp import ...' is the primary pattern.
try:
    from src.mcp import MCPTool # MCPTool class for defining tools
    # mcp_instance will be passed to register_tools, no need to import it directly here
except ImportError:
    # Fallback if path isn't set up as expected during direct linting/testing of this file,
    # though at runtime by the pipeline, src.mcp should be findable.
    # This helps with static analysis if this file is checked in isolation.
    # For runtime, the discover_modules in src/mcp/mcp.py handles path adjustments.
    logging.warning("Could not directly import MCPTool from src.mcp. This might be okay if loaded by the main MCP system.")
    # Define a placeholder if really needed for isolated testing, but ideally not hit at runtime.
    class MCPTool:
        def __init__(self, name: str, func: Callable, schema: Dict, description: str):
            pass # Minimal placeholder

logger = logging.getLogger(__name__)


# --- Tool Input/Output Models ---

class RenderGnnInput(BaseModel):
    gnn_specification: Union[Dict[str, Any], str] = Field(
        description="The GNN specification itself as a dictionary, or a string URI/path to a GNN spec file (e.g., JSON)."
    )
    target_format: Literal["pymdp", "rxinfer"] = Field(
        description="The target format to render the GNN specification to."
    )
    output_filename_base: Optional[str] = Field(
        None,
        description="Optional desired base name for the output file (e.g., 'my_model'). Extension is added automatically. If None, derived from GNN spec name or input file name."
    )
    render_options: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional dictionary of specific options for the chosen renderer (e.g., data_bindings for RxInfer)."
    )

class RenderGnnOutput(BaseModel):
    success: bool = Field(description="Whether the rendering was successful.")
    message: str = Field(description="A message describing the outcome of the rendering process.")
    artifact_uri: Optional[str] = Field(
        None, 
        description="URI to the generated rendered file, if successful and file was saved."
    )
    rendered_content_preview: Optional[str] = Field(
        None,
        description="A preview of the rendered content (e.g., first N lines). May be null if content is large or not applicable."
    )

class ListRenderTargetsInput(BaseModel):
    pass

class ListRenderTargetsOutput(BaseModel):
    targets: List[str] = Field(description="A list of supported rendering target formats.")

# --- Tool Implementations (handlers) ---
# These handlers are now designed to be called by the MCP framework.
# The 'context' argument is removed as it's not standard for simple tool functions
# unless the MCP framework specifically provides and requires it.
# If context is needed (e.g., for auth, user info), the MCP registration mechanism would handle it.

async def handle_render_gnn_spec(input_data: RenderGnnInput) -> RenderGnnOutput:
    """Handles a request to render a GNN specification to a target format."""
    logger.info(f"MCP Tool: Received request to render GNN to {input_data.target_format}")

    gnn_spec_dict: Optional[Dict[str, Any]] = None

    if isinstance(input_data.gnn_specification, dict):
        gnn_spec_dict = input_data.gnn_specification
        logger.debug("Received GNN specification directly as a dictionary.")
    elif isinstance(input_data.gnn_specification, str):
        gnn_file_path = Path(input_data.gnn_specification)
        logger.debug(f"Attempting to load GNN specification from path: {gnn_file_path}")
        if not gnn_file_path.is_file():
            return RenderGnnOutput(success=False, message=f"GNN specification file not found: {gnn_file_path}")
        try:
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                gnn_spec_dict = json.load(f)
            logger.info(f"Successfully loaded GNN specification from {gnn_file_path}")
        except json.JSONDecodeError as e:
            return RenderGnnOutput(success=False, message=f"Error decoding JSON from {gnn_file_path}: {e}")
        except Exception as e:
            return RenderGnnOutput(success=False, message=f"Failed to read GNN file {gnn_file_path}: {e}")
    else:
        return RenderGnnOutput(success=False, message="Invalid gnn_specification type. Must be dict or str path.")

    if gnn_spec_dict is None:
         return RenderGnnOutput(success=False, message="Could not obtain GNN specification dictionary.")

    temp_dir = Path(tempfile.mkdtemp(prefix="gnn_render_mcp_"))
    
    filename_base = input_data.output_filename_base
    if not filename_base:
        filename_base = gnn_spec_dict.get("name", "rendered_gnn_model")
        filename_base = filename_base.replace(" ", "_").lower()
    
    file_extension = ".py" if input_data.target_format == "pymdp" else ".jl"
    output_script_name = f"{filename_base}{file_extension}"
    temp_output_path = temp_dir / output_script_name

    logger.info(f"Rendering to temporary file: {temp_output_path}")

    success, message, artifacts = render_gnn_spec(
        gnn_spec=gnn_spec_dict,
        output_script_path=temp_output_path,
        target_format=input_data.target_format,
        render_options=input_data.render_options or {}
    )

    if success:
        artifact_uri = temp_output_path.as_uri() if artifacts else None
        content_preview = None
        try:
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                lines = [next(f) for _ in range(20)]
                content_preview = "".join(lines) + ("... (truncated)" if len(lines) == 20 else "")
        except Exception as e:
            logger.warning(f"Could not read rendered file for preview: {e}")
        
        return RenderGnnOutput(
            success=True, 
            message=message, 
            artifact_uri=artifact_uri,
            rendered_content_preview=content_preview
        )
    else:
        logger.error(f"MCP Tool: Rendering GNN failed. Message from renderer: {message}")
        return RenderGnnOutput(success=False, message=message)

async def handle_list_render_targets() -> ListRenderTargetsOutput:
    """Lists the supported rendering target formats."""
    logger.info("MCP Tool: Received request to list render targets.")
    supported_targets = ["pymdp", "rxinfer"]
    return ListRenderTargetsOutput(targets=supported_targets)

# --- MCP Tool Registration ---

def register_tools(mcp_instance_param): # Name changed to avoid conflict if mcp_instance is imported
    """
    Registers the rendering tools with the provided MCP instance.
    This function will be called by the main MCP module during discovery.
    """
    
    # Note on schemas: Pydantic models (RenderGnnInput, ListRenderTargetsOutput)
    # can be converted to JSON Schema for MCP. The MCP registration might handle this
    # automatically if it supports Pydantic, or a manual conversion step might be needed.
    # For simplicity here, we'll pass the Pydantic model itself, assuming the
    # MCP framework can derive or is given the schema.
    # If MCPTool expects a dict schema: input_schema=RenderGnnInput.schema()

    mcp_instance_param.register_tool(
        name="render_gnn_specification",
        func=handle_render_gnn_spec,
        schema=RenderGnnInput.model_json_schema(),
        description="Renders a GNN (Generalized Notation Notation) specification into an executable format for a target modeling environment like PyMDP or RxInfer.jl."
    )
    
    mcp_instance_param.register_tool(
        name="list_render_targets",
        func=handle_list_render_targets,
        schema=ListRenderTargetsInput.model_json_schema(),
        description="Lists the available target formats for GNN rendering (e.g., pymdp, rxinfer)."
    )
    
    logger.info("Render module MCP tools registered.")

# The standalone server setup (serve_render_mcp, if __name__ == "__main__") is removed.
# This module now only defines tools to be registered by the main MCP system.


# Removed the if __name__ == "__main__": block as this module is intended
# to be loaded by the main MCP system for tool registration, not run standalone as a server.
# For isolated testing of handlers, one would typically import them and call them directly
# with mock inputs, or use a dedicated test script that sets up a minimal MCP environment. 