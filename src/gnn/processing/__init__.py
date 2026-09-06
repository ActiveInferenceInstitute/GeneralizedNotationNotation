"""GNN processing engine: discovery, parsing surface, orchestration, multi-format.

Consolidates the processing concern:

- ``processor``       — lightweight file discovery, parsing, validation, reporting
- ``core_processor``  - ``GNNProcessor`` five-phase orchestration engine
- ``multi_format_processor`` — step-3 multi-format serialization orchestrator
- ``discovery``       — corpus filtering predicates and the discovery strategy
"""

from gnn.processing.core_processor import (
    GNNProcessor,
    ProcessingContext,
    ProcessingPhase,
    create_processor,
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
    discover_gnn_files,
    generate_gnn_report,
    get_module_info,
    parse_gnn_file,
    process_gnn_directory,
    process_gnn_directory_lightweight,
    validate_gnn_structure,
)

__all__ = [
    "DiscoveryResult",
    "FileDiscoveryStrategy",
    "GNNProcessor",
    "NON_MODEL_MARKDOWN_FILENAMES",
    "NON_MODEL_MARKDOWN_SUFFIXES",
    "ProcessingContext",
    "ProcessingPhase",
    "create_processor",
    "discover_gnn_files",
    "generate_gnn_report",
    "get_module_info",
    "is_model_source_path",
    "parse_gnn_file",
    "process_gnn_directory",
    "process_gnn_directory_lightweight",
    "process_gnn_multi_format",
    "validate_gnn_structure",
]
