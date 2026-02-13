"""
GNN module for GNN Processing Pipeline.

This module provides GNN file discovery, parsing, and validation capabilities.
"""

from .processor import (
    process_gnn_directory_lightweight,
    _extract_sections_lightweight,
    _extract_variables_lightweight,
    discover_gnn_files,
    parse_gnn_file,
    validate_gnn_structure,
    process_gnn_directory,
    generate_gnn_report,
    get_module_info
)

from .multi_format_processor import (
    process_gnn_multi_format
)

from .parser import (
    ValidationLevel,
    ParsedGNN,
    GNNParsingSystem,
    GNNFormat,
    GNNFormalParser,
    ParsedGNNFormal,
    parse_gnn_formal,
    validate_gnn_syntax_formal,
    get_parse_tree_visualization,
    parsers,
    validate_gnn,
    _convert_parse_result_to_parsed_gnn
)

__version__ = "1.1.3"
# Ensure tests see MCP feature presence consistently
FEATURES = {
    "file_discovery": True,
    "content_parsing": True,
    "structure_validation": True,
    "report_generation": True,
    "core_validation": True,
    "mcp_integration": True,
}

def process_gnn(*args, **kwargs):
    return process_gnn_directory(*args, **kwargs)

def validate_gnn_file(content: str):
    """Validate a GNN file's content using the real parser validate_gnn."""
    is_valid, errors = validate_gnn(content)
    return {"is_valid": is_valid, "errors": errors}

__all__ = [
    # Processor functions
    'process_gnn_directory_lightweight',
    '_extract_sections_lightweight',
    '_extract_variables_lightweight',
    'discover_gnn_files',
    'parse_gnn_file',
    'validate_gnn_structure',
    'process_gnn_directory',
    'generate_gnn_report',
    'get_module_info',

    # Multi-format processor
    'process_gnn_multi_format',
    
    # Parser classes and functions
    'ValidationLevel',
    'ParsedGNN',
    'GNNParsingSystem',
    'GNNFormat',
    'GNNFormalParser',
    'ParsedGNNFormal',
    'parse_gnn_formal',
    'validate_gnn_syntax_formal',
    'get_parse_tree_visualization',
    'parsers',
    'validate_gnn',
    '_convert_parse_result_to_parsed_gnn',
    '__version__',
    'FEATURES',
    'process_gnn',
    'validate_gnn_file'
] 