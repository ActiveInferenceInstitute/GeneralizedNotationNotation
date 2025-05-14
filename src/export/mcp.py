# Minimal mcp.py for the export module

import logging
from typing import Dict, Any

# Attempt to import all necessary components from format_exporters
# Assuming format_exporters.py is in the same directory (package)
try:
    from .format_exporters import (
        _gnn_model_to_dict,
        export_to_json_gnn,
        export_to_xml_gnn,
        export_to_plaintext_summary,
        export_to_plaintext_dsl,
        export_to_gexf,
        export_to_graphml,
        export_to_json_adjacency_list,
        export_to_python_pickle,
        # Import nx to check its availability for relevant exporters
        nx 
    )
except ImportError as e:
    # This allows the module to load even if format_exporters or its dependencies are problematic,
    # though tools might fail at runtime. MCP registration can then indicate issues.
    logging.error(f"Could not import from .format_exporters: {e}")
    # Define placeholders if imports fail, so the rest of the file can be parsed,
    # but tools relying on these will not work.
    _gnn_model_to_dict = None
    export_to_json_gnn = None
    export_to_xml_gnn = None
    export_to_plaintext_summary = None
    export_to_plaintext_dsl = None
    export_to_gexf = None
    export_to_graphml = None
    export_to_json_adjacency_list = None
    export_to_python_pickle = None
    nx = None


logger = logging.getLogger(__name__)

# --- MCP Tool Wrapper Functions ---

def _handle_export(export_func, gnn_file_path: str, output_file_path: str, format_name: str, requires_nx: bool = False) -> Dict[str, Any]:
    """Generic helper to run an export function and handle common exceptions."""
    if export_func is None or _gnn_model_to_dict is None:
        missing_module = "format_exporters or its dependencies"
        logger.error(f"Export to {format_name} failed: {missing_module} not correctly imported.")
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": f"{missing_module} not available. Cannot perform export."
        }
        
    if requires_nx and nx is None:
        logger.error(f"NetworkX library is not available. Cannot export to {format_name}.")
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": f"NetworkX library is required for {format_name} export but is not installed or available."
        }
        
    try:
        gnn_model = _gnn_model_to_dict(gnn_file_path)
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
    except ImportError as ie: # Should be caught by nx check, but as a fallback
        logger.error(f"ImportError during {format_name} export for '{gnn_file_path}': {ie}")
        return {"success": False, "input_file": gnn_file_path, "output_file": output_file_path, "error": f"Missing dependency for {format_name}: {str(ie)}"}
    except Exception as e:
        logger.error(f"Failed to export GNN to {format_name} for '{gnn_file_path}': {e}", exc_info=True)
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": str(e)
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

    # Updated structure: (mcp_tool_name, mcp_wrapper_func, core_exporter_name_str, description)
    tools_to_register = [
        ("export_gnn_to_json", export_gnn_to_json_mcp, "export_to_json_gnn", "Exports a GNN model to JSON format."),
        ("export_gnn_to_xml", export_gnn_to_xml_mcp, "export_to_xml_gnn", "Exports a GNN model to XML format."),
        ("export_gnn_to_plaintext_summary", export_gnn_to_plaintext_summary_mcp, "export_to_plaintext_summary", "Exports a GNN model to a human-readable plain text summary."),
        ("export_gnn_to_plaintext_dsl", export_gnn_to_plaintext_dsl_mcp, "export_to_plaintext_dsl", "Exports a GNN model back to its GNN DSL plain text format."),
        ("export_gnn_to_gexf", export_gnn_to_gexf_mcp, "export_to_gexf", "Exports a GNN model to GEXF graph format (requires NetworkX)."),
        ("export_gnn_to_graphml", export_gnn_to_graphml_mcp, "export_to_graphml", "Exports a GNN model to GraphML graph format (requires NetworkX)."),
        ("export_gnn_to_json_adjacency_list", export_gnn_to_json_adjacency_list_mcp, "export_to_json_adjacency_list", "Exports a GNN model to JSON Adjacency List graph format (requires NetworkX)."),
        ("export_gnn_to_python_pickle", export_gnn_to_python_pickle_mcp, "export_to_python_pickle", "Serializes a GNN model to a Python pickle file.")
    ]

    for mcp_tool_name, mcp_wrapper_func, core_exporter_name_str, description in tools_to_register:
        if mcp_wrapper_func is not None and getattr(mcp_wrapper_func, '__name__', '') != '_handle_export': # Ensure the target function is valid
            # Check if the underlying export function (by its string name) was imported correctly and is not None
            if globals().get(core_exporter_name_str) is None:
                 logger.warning(f"Skipping registration of MCP tool '{mcp_tool_name}': Its underlying core export function '{core_exporter_name_str}' was not found or not imported correctly from format_exporters.py.")
                 continue # Skip registration if the core function is missing

            mcp_instance.register_tool(
                name=mcp_tool_name,
                func=mcp_wrapper_func,
                schema=base_schema.copy(), # Use a copy to avoid modification issues if schema varies later
                description=description
            )
        else:
            logger.warning(f"Skipping registration of MCP tool '{mcp_tool_name}': Its MCP wrapper function '{getattr(mcp_wrapper_func, '__name__', 'N/A')}' is not available (e.g. None or points to _handle_export, possibly due to an import error for the wrapper itself).")

# Remove the old get_mcp_interface if it exists, or ensure this file only defines the above.
# The main MCP loader will look for `register_tools`. 