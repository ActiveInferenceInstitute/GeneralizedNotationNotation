"""
GNN module for GNN Processing Pipeline.

This module provides GNN file discovery, parsing, and validation capabilities.
"""

from .multi_format_processor import process_gnn_multi_format
from .parser import (
    GNNFormalParser,
    ParsedGNNFormal,
    get_parse_tree_visualization,
    parse_gnn_formal,
    validate_gnn,
    validate_gnn_syntax_formal,
)
from .parsers.common import GNNFormat
from .parsers.system import (
    GNNParsingSystem,  # canonical 23-format registry (23 parsers, 22 serializers; PNML is parse-only)
)
from .processor import (
    discover_gnn_files,
    generate_gnn_report,
    get_module_info,
    parse_gnn_file,
    process_gnn_directory,
    process_gnn_directory_lightweight,
    validate_gnn_structure,
)

# Canonical domain types live in types.py
from .types import ParsedGNN, ValidationLevel

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

def validate_gnn_file(source, *, is_content: bool = False):
    """Validate a GNN file or content string.

    Args:
        source: File path (str or Path) or raw GNN content string.
        is_content: If True, treat source as raw content regardless of type.

    Returns:
        Dict with keys ``is_valid`` (bool) and ``errors`` (list[str]).
    """
    from pathlib import Path as _Path
    if not is_content and isinstance(source, (str, _Path)) and _Path(source).exists():
        content = _Path(source).read_text(encoding="utf-8")
    else:
        content = str(source)
    is_valid, errors = validate_gnn(content)
    return {"is_valid": is_valid, "errors": errors}

__all__ = [
    # Processor functions
    'process_gnn_directory_lightweight',
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
    'validate_gnn',
    '__version__',
    'FEATURES',
    'validate_gnn_file'
]
