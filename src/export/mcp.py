# Minimal mcp.py for the export module

import logging
from typing import Dict, Any, Callable # Added Callable
from pathlib import Path # For type hints and potential path ops if needed

# Import from the new specialized exporter modules
from .structured_data_exporters import (
    export_to_json_gnn,
    export_to_xml_gnn,
    export_to_python_pickle
)
from .graph_exporters import (
    export_to_gexf,
    export_to_graphml,
    export_to_json_adjacency_list,
    HAS_NETWORKX as GRAPH_EXPORTERS_HAVE_NETWORKX # Use this to check for NetworkX availability
)
from .text_exporters import (
    export_to_plaintext_summary,
    export_to_plaintext_dsl
)

# The GNN parser is still needed for the _handle_export function
# Assuming _gnn_model_to_dict is in format_exporters.py in the same directory
try:
    from .format_exporters import _gnn_model_to_dict
except ImportError as e:
    logging.getLogger(__name__).error(f"Could not import _gnn_model_to_dict from .format_exporters: {e}")
    _gnn_model_to_dict = None # Will cause tools to fail gracefully

logger = logging.getLogger(__name__)

# --- MCP Tool Wrapper Functions ---

def _handle_export(
    export_func: Callable[[Dict[str, Any], str], None], 
    gnn_file_path: str, 
    output_file_path: str, 
    format_name: str, 
    requires_nx: bool = False
) -> Dict[str, Any]:
    """Generic helper to run an export function and handle common exceptions."""
    if _gnn_model_to_dict is None:
        logger.error(f"Export to {format_name} failed: _gnn_model_to_dict parser not available.")
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": "GNN parser (_gnn_model_to_dict) not available. Cannot perform export."
        }
        
    if requires_nx and not GRAPH_EXPORTERS_HAVE_NETWORKX:
        logger.error(f"NetworkX library is not available. Cannot export to {format_name}.")
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": f"NetworkX library is required for {format_name} export but is not installed or available."
        }
        
    try:
        gnn_model = _gnn_model_to_dict(gnn_file_path)
        if gnn_model is None: # Parser might return None on critical failure
            raise ValueError("GNN parsing resulted in None, cannot proceed with export.")
            
        export_func(gnn_model, output_file_path)
        return {
            "success": True,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "message": f"Successfully exported GNN model from '{gnn_file_path}' to {format_name}: '{output_file_path}'"
        }
    except FileNotFoundError as fnfe:
        logger.error(f"Input GNN file not found ('{gnn_file_path}') for {format_name} export: {fnfe}")
        return {"success": False, "input_file": gnn_file_path, "error": f"Input file not found: {str(fnfe)}"}
    except ImportError as ie: 
        logger.error(f"ImportError during {format_name} export for '{gnn_file_path}': {ie}")
        return {"success": False, "input_file": gnn_file_path, "output_file": output_file_path, "error": f"Missing dependency for {format_name}: {str(ie)}"}
    except Exception as e:
        logger.error(f"Failed to export GNN to {format_name} for '{gnn_file_path}': {e}", exc_info=True)
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

def export_gnn_to_json_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_json_gnn, gnn_file_path, output_file_path, "JSON")

def export_gnn_to_xml_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_xml_gnn, gnn_file_path, output_file_path, "XML")

def export_gnn_to_plaintext_summary_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_plaintext_summary, gnn_file_path, output_file_path, "Plaintext Summary")

def export_gnn_to_plaintext_dsl_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_plaintext_dsl, gnn_file_path, output_file_path, "Plaintext DSL")

def export_gnn_to_gexf_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_gexf, gnn_file_path, output_file_path, "GEXF", requires_nx=True)

def export_gnn_to_graphml_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_graphml, gnn_file_path, output_file_path, "GraphML", requires_nx=True)

def export_gnn_to_json_adjacency_list_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_json_adjacency_list, gnn_file_path, output_file_path, "JSON Adjacency List", requires_nx=True)

def export_gnn_to_python_pickle_mcp(gnn_file_path: str, output_file_path: str) -> Dict[str, Any]:
    return _handle_export(export_to_python_pickle, gnn_file_path, output_file_path, "Python Pickle")

# --- MCP Registration ---

def register_tools(mcp_instance):
    """Registers all GNN export tools with the MCP instance."""
    
    base_schema = {
        "gnn_file_path": {"type": "string", "description": "Path to the input GNN Markdown file (.gnn.md)."},
        "output_file_path": {"type": "string", "description": "Path where the exported file will be saved."}
    }

    tools_to_register_spec = [
        ("export_gnn_to_json", export_gnn_to_json_mcp, export_to_json_gnn, "Exports a GNN model to JSON format.", False),
        ("export_gnn_to_xml", export_gnn_to_xml_mcp, export_to_xml_gnn, "Exports a GNN model to XML format.", False),
        ("export_gnn_to_plaintext_summary", export_gnn_to_plaintext_summary_mcp, export_to_plaintext_summary, "Exports a GNN model to a human-readable plain text summary.", False),
        ("export_gnn_to_plaintext_dsl", export_gnn_to_plaintext_dsl_mcp, export_to_plaintext_dsl, "Exports a GNN model back to its GNN DSL plain text format.", False),
        ("export_gnn_to_gexf", export_gnn_to_gexf_mcp, export_to_gexf, "Exports a GNN model to GEXF graph format (requires NetworkX).", True),
        ("export_gnn_to_graphml", export_gnn_to_graphml_mcp, export_to_graphml, "Exports a GNN model to GraphML graph format (requires NetworkX).", True),
        ("export_gnn_to_json_adjacency_list", export_gnn_to_json_adjacency_list_mcp, export_to_json_adjacency_list, "Exports a GNN model to JSON Adjacency List graph format (requires NetworkX).", True),
        ("export_gnn_to_python_pickle", export_gnn_to_python_pickle_mcp, export_to_python_pickle, "Serializes a GNN model to a Python pickle file.", False)
    ]

    for mcp_tool_name, mcp_wrapper_func, core_exporter_func, description, needs_nx_flag in tools_to_register_spec:
        if core_exporter_func is None: # Check if the core function itself is None (due to import issues in specialized modules)
             logger.warning(f"Skipping registration of MCP tool '{mcp_tool_name}': Its underlying core export function was not imported correctly from its specialized module.")
             continue
        if needs_nx_flag and not GRAPH_EXPORTERS_HAVE_NETWORKX:
            logger.warning(f"Skipping registration of MCP tool '{mcp_tool_name}': It requires NetworkX, which is not available.")
            continue

        mcp_instance.register_tool(
            name=mcp_tool_name,
            func=mcp_wrapper_func,
            schema=base_schema.copy(), 
            description=description
        )

# Remove the old get_mcp_interface if it exists, or ensure this file only defines the above.
# The main MCP loader will look for `register_tools`. 