"""Lazy public processing surface; lightweight callers need only the standard library."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gnn.processing.core_processor import (
        GNNProcessor,
        ProcessingContext,
        ProcessingPhase,
    )
    from gnn.processing.discovery import (
        NON_MODEL_MARKDOWN_FILENAMES,
        NON_MODEL_MARKDOWN_SUFFIXES,
        DiscoveryResult,
        FileDiscoveryStrategy,
        is_model_source_path,
    )
    from gnn.processing.multi_format_processor import process_gnn_multi_format
    from gnn.processing.processor import (
        check_gnn_file_structure,
        discover_gnn_files,
        generate_gnn_report,
        get_module_info,
        parse_gnn_file,
        process_gnn_directory,
        process_gnn_directory_lightweight,
    )

_EXPORT_MAP = {
    "GNNProcessor": "gnn.processing.core_processor",
    "ProcessingContext": "gnn.processing.core_processor",
    "ProcessingPhase": "gnn.processing.core_processor",
    "NON_MODEL_MARKDOWN_FILENAMES": "gnn.processing.discovery",
    "NON_MODEL_MARKDOWN_SUFFIXES": "gnn.processing.discovery",
    "DiscoveryResult": "gnn.processing.discovery",
    "FileDiscoveryStrategy": "gnn.processing.discovery",
    "is_model_source_path": "gnn.processing.discovery",
    "process_gnn_multi_format": "gnn.processing.multi_format_processor",
    "check_gnn_file_structure": "gnn.processing.processor",
    "discover_gnn_files": "gnn.processing.processor",
    "generate_gnn_report": "gnn.processing.processor",
    "get_module_info": "gnn.processing.processor",
    "parse_gnn_file": "gnn.processing.processor",
    "process_gnn_directory": "gnn.processing.processor",
    "process_gnn_directory_lightweight": "gnn.processing.processor",
}


def __getattr__(name: str) -> Any:
    """Resolve only the requested owner; preserve its genuine import failures."""
    from importlib import import_module

    module = _EXPORT_MAP.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include public lazy exports without loading their owners."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "check_gnn_file_structure",
    "DiscoveryResult",
    "FileDiscoveryStrategy",
    "GNNProcessor",
    "NON_MODEL_MARKDOWN_FILENAMES",
    "NON_MODEL_MARKDOWN_SUFFIXES",
    "ProcessingContext",
    "ProcessingPhase",
    "discover_gnn_files",
    "generate_gnn_report",
    "get_module_info",
    "is_model_source_path",
    "parse_gnn_file",
    "process_gnn_directory",
    "process_gnn_directory_lightweight",
    "process_gnn_multi_format",
]
