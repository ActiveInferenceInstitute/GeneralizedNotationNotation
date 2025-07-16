"""
GNN Export Module

This package provides comprehensive tools for exporting GNN models to various formats,
including structured data (JSON, XML), graph formats (GEXF, GraphML), and text-based representations.
"""

# Core export functions
from .format_exporters import (
    _gnn_model_to_dict,
    export_to_json_gnn,
    export_to_xml_gnn,
    export_to_python_pickle,
    export_to_gexf,
    export_to_graphml,
    export_to_json_adjacency_list,
    export_to_plaintext_summary,
    export_to_plaintext_dsl,
    HAS_NETWORKX
)

# MCP integration
try:
    from .mcp import (
        register_tools,
        export_gnn_to_json_mcp,
        export_gnn_to_xml_mcp,
        export_gnn_to_plaintext_summary_mcp,
        export_gnn_to_plaintext_dsl_mcp,
        export_gnn_to_gexf_mcp,
        export_gnn_to_graphml_mcp,
        export_gnn_to_json_adjacency_list_mcp,
        export_gnn_to_python_pickle_mcp
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.1.0"
__author__ = "Active Inference Institute"
__description__ = "GNN model export to multiple formats"

# Feature availability flags
FEATURES = {
    'json_export': True,
    'xml_export': True,
    'pickle_export': True,
    'plaintext_export': True,
    'graph_export': HAS_NETWORKX,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
    # Core export functions
    '_gnn_model_to_dict',
    'export_to_json_gnn',
    'export_to_xml_gnn',
    'export_to_python_pickle',
    'export_to_plaintext_summary',
    'export_to_plaintext_dsl',
    
    # Graph export functions (if NetworkX available)
    'export_to_gexf',
    'export_to_graphml',
    'export_to_json_adjacency_list',
    
    # MCP integration (if available)
    'register_tools',
    'export_gnn_to_json_mcp',
    'export_gnn_to_xml_mcp',
    'export_gnn_to_plaintext_summary_mcp',
    'export_gnn_to_plaintext_dsl_mcp',
    'export_gnn_to_gexf_mcp',
    'export_gnn_to_graphml_mcp',
    'export_gnn_to_json_adjacency_list_mcp',
    'export_gnn_to_python_pickle_mcp',
    
    # Metadata
    'FEATURES',
    'HAS_NETWORKX',
    '__version__'
]

# Add conditional exports
if not HAS_NETWORKX:
    __all__.remove('export_to_gexf')
    __all__.remove('export_to_graphml')
    __all__.remove('export_to_json_adjacency_list')
    __all__.remove('export_gnn_to_gexf_mcp')
    __all__.remove('export_gnn_to_graphml_mcp')
    __all__.remove('export_gnn_to_json_adjacency_list_mcp')

if not MCP_AVAILABLE:
    __all__.remove('register_tools')
    __all__.remove('export_gnn_to_json_mcp')
    __all__.remove('export_gnn_to_xml_mcp')
    __all__.remove('export_gnn_to_plaintext_summary_mcp')
    __all__.remove('export_gnn_to_plaintext_dsl_mcp')
    __all__.remove('export_gnn_to_gexf_mcp')
    __all__.remove('export_gnn_to_graphml_mcp')
    __all__.remove('export_gnn_to_json_adjacency_list_mcp')
    __all__.remove('export_gnn_to_python_pickle_mcp')


def get_module_info():
    """Get comprehensive information about the export module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'available_formats': [],
        'graph_formats': [],
        'text_formats': [],
        'data_formats': []
    }
    
    # Available formats
    info['available_formats'].extend(['JSON', 'XML', 'Python Pickle', 'Plaintext Summary', 'Plaintext DSL'])
    info['data_formats'].extend(['JSON', 'XML', 'Python Pickle'])
    info['text_formats'].extend(['Plaintext Summary', 'Plaintext DSL'])
    
    if HAS_NETWORKX:
        info['available_formats'].extend(['GEXF', 'GraphML', 'JSON Adjacency List'])
        info['graph_formats'].extend(['GEXF', 'GraphML', 'JSON Adjacency List'])
    
    return info


def export_gnn_model(gnn_file_path: str, output_format: str, output_file_path: str = None) -> dict:
    """
    Export a GNN model to the specified format.
    
    Args:
        gnn_file_path: Path to the input GNN file
        output_format: Target format ('json', 'xml', 'pickle', 'gexf', 'graphml', 'summary', 'dsl')
        output_file_path: Output file path (auto-generated if None)
    
    Returns:
        Dictionary with export result information
    """
    import os
    from pathlib import Path
    
    # Auto-generate output path if not provided
    if output_file_path is None:
        input_path = Path(gnn_file_path)
        output_file_path = str(input_path.parent / f"{input_path.stem}.{output_format}")
    
    # Map format names to functions
    format_functions = {
        'json': export_to_json_gnn,
        'xml': export_to_xml_gnn,
        'pickle': export_to_python_pickle,
        'summary': export_to_plaintext_summary,
        'dsl': export_to_plaintext_dsl
    }
    
    # Add graph formats if NetworkX is available
    if HAS_NETWORKX:
        format_functions.update({
            'gexf': export_to_gexf,
            'graphml': export_to_graphml,
            'adjacency': export_to_json_adjacency_list
        })
    
    if output_format.lower() not in format_functions:
        return {
            "success": False,
            "error": f"Unsupported format: {output_format}. Available formats: {list(format_functions.keys())}"
        }
    
    try:
        # Parse GNN model
        gnn_model = _gnn_model_to_dict(gnn_file_path)
        
        # Export to target format
        export_func = format_functions[output_format.lower()]
        export_func(gnn_model, output_file_path)
        
        return {
            "success": True,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "format": output_format,
            "message": f"Successfully exported to {output_format}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "input_file": gnn_file_path,
            "output_file": output_file_path,
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_supported_formats() -> dict:
    """Get information about all supported export formats."""
    formats = {
        'data_formats': {
            'json': 'Structured JSON representation',
            'xml': 'XML document format',
            'pickle': 'Python pickle serialization'
        },
        'text_formats': {
            'summary': 'Human-readable plain text summary',
            'dsl': 'GNN DSL plain text format'
        }
    }
    
    if HAS_NETWORKX:
        formats['graph_formats'] = {
            'gexf': 'GEXF graph format',
            'graphml': 'GraphML format',
            'adjacency': 'JSON adjacency list'
        }
    
    return formats
