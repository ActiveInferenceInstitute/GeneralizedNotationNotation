"""
GNN module for GNN Processing Pipeline.

This module provides GNN file discovery, parsing, and validation capabilities.

Importing ``gnn`` is intentionally LIGHT (lazy PEP 562 re-exports): no
submodule executes at import time, so the package can be imported — and the
headless POMDP extractor (``gnn.pomdp_extractor``) used — without pulling in
the full pipeline stack or heavy module-scope dependencies (psutil,
matplotlib). Names resolve through ``__getattr__`` on first access.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static re-export surface for type checkers: mirrors the pre-lazy eager
    # imports so mypy resolves names without executing submodules at runtime.
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
    from .parsers.system import GNNParsingSystem
    from .processor import (
        discover_gnn_files,
        generate_gnn_report,
        get_module_info,
        parse_gnn_file,
        process_gnn_directory,
        process_gnn_directory_lightweight,
        validate_gnn_structure,
    )
    from .types import ParsedGNN, ValidationLevel

__version__ = "1.6.0"

# Ensure tests see MCP feature presence consistently
FEATURES: dict[str, Any] = {
    "file_discovery": True,
    "content_parsing": True,
    "structure_validation": True,
    "report_generation": True,
    "core_validation": True,
    "mcp_integration": True,
}

# Explicit name -> source submodule map for every re-export. Resolving a name
# imports only that one submodule.
_EXPORT_MAP: dict[str, str] = {
    # multi_format_processor
    "process_gnn_multi_format": "multi_format_processor",
    # parser
    "GNNFormalParser": "parser",
    "ParsedGNNFormal": "parser",
    "get_parse_tree_visualization": "parser",
    "parse_gnn_formal": "parser",
    "validate_gnn": "parser",
    "validate_gnn_syntax_formal": "parser",
    # parsers.common
    "GNNFormat": "parsers.common",
    # parsers.system — canonical 23-format registry (23 parsers, 22 serializers;
    # PNML is parse-only)
    "GNNParsingSystem": "parsers.system",
    # processor
    "discover_gnn_files": "processor",
    "generate_gnn_report": "processor",
    "get_module_info": "processor",
    "parse_gnn_file": "processor",
    "process_gnn_directory": "processor",
    "process_gnn_directory_lightweight": "processor",
    "validate_gnn_structure": "processor",
    # types — canonical domain types
    "ParsedGNN": "types",
    "ValidationLevel": "types",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve a re-exported name (PEP 562).

    The submodule owning ``name`` is imported on first access and the value is
    cached in the module globals. Any ImportError raised while importing the
    owning submodule propagates unchanged — no silent fallback.
    """
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include re-exported names in ``dir(gnn)`` alongside module globals."""
    return sorted(set(globals()) | set(_EXPORT_MAP))


def validate_gnn_file(source: Any, *, is_content: bool = False) -> Any:
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
    from .parser import validate_gnn as _validate_gnn

    is_valid, errors = _validate_gnn(content)
    return {"is_valid": is_valid, "errors": errors}


__all__: list[Any] = [
    # Processor functions
    "process_gnn_directory_lightweight",
    "discover_gnn_files",
    "parse_gnn_file",
    "validate_gnn_structure",
    "process_gnn_directory",
    "generate_gnn_report",
    "get_module_info",
    # Multi-format processor
    "process_gnn_multi_format",
    # Parser classes and functions
    "ValidationLevel",
    "ParsedGNN",
    "GNNParsingSystem",
    "GNNFormat",
    "GNNFormalParser",
    "ParsedGNNFormal",
    "parse_gnn_formal",
    "validate_gnn_syntax_formal",
    "get_parse_tree_visualization",
    "validate_gnn",
    "__version__",
    "FEATURES",
    "validate_gnn_file",
]
