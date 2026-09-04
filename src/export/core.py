"""Provides helper functions: export_gnn_files.

Public functions: export_gnn_files
"""

import logging
from pathlib import Path
from typing import Any, Tuple, cast

from pipeline import get_output_dir_for_script
from utils import log_step_error, log_step_start, log_step_success, log_step_warning

# Import format exporters
try:
    from .format_exporters import (
        HAS_NETWORKX,
        _gnn_model_to_dict,
        export_to_gexf,
        export_to_graphml,
        export_to_json_adjacency_list,
        export_to_json_gnn,
        export_to_plaintext_dsl,
        export_to_plaintext_summary,
        export_to_python_pickle,
        export_to_xml_gnn,
    )

    FORMAT_EXPORTERS_LOADED = True
except ImportError:
    FORMAT_EXPORTERS_LOADED = False


def _writer_success(result: Any) -> bool:
    """Normalize a format_exporters writer result to a plain success flag.

    Writers in :mod:`export.format_exporters` return ``Tuple[bool, str]``;
    the legacy ``formatters`` writers return a bare ``bool``. Treat anything
    truthy-but-not-a-tuple as success, and any tuple whose first element is
    truthy as success. This closes a silent-failure path where the tuple
    ``(False, "NetworkX not available…")`` was being logged as a successful
    export by ``export_gnn_files``.
    """
    if isinstance(result, tuple):
        return bool(result[0]) if result else False
    return bool(result)


def _writer_error(result: Any) -> str:
    """Extract the error message from a tuple-returning writer, else ``""``."""
    if isinstance(result, tuple) and len(result) > 1:
        return str(result[1])
    return ""


def export_gnn_files(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Export GNN files to multiple formats.

    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional export options

    Returns:
        True if export succeeded, False otherwise
    """
    log_step_start(logger, f"Exporting GNN files from: {target_dir}")

    if not FORMAT_EXPORTERS_LOADED:
        log_step_error(logger, "Format exporters not available")
        return False

    # Use centralized output directory configuration
    export_output_dir = get_output_dir_for_script("7_export.py", output_dir)
    export_output_dir.mkdir(parents=True, exist_ok=True)

    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))

    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        return True

    logger.info(f"Found {len(gnn_files)} GNN files to export")

    success_count = 0
    total_files = len(gnn_files)

    # Per-format writer dispatch table. Each entry is (label, writer, path_template, needs_networkx).
    # Writers come from format_exporters and return Tuple[bool, str]; we normalize via
    # _writer_success and capture the error message on failure. Graph formats are only
    # dispatched when NetworkX is available; the others are pure-stdlib and always run.
    writer_table: list[tuple[str, Any, str, bool]] = [
        ("JSON", export_to_json_gnn, "{stem}.json", False),
        ("XML", export_to_xml_gnn, "{stem}.xml", False),
        ("Summary", export_to_plaintext_summary, "{stem}_summary.txt", False),
        ("DSL", export_to_plaintext_dsl, "{stem}_dsl.txt", False),
        ("GEXF", export_to_gexf, "{stem}.gexf", True),
        ("GraphML", export_to_graphml, "{stem}.graphml", True),
        ("Adjacency", export_to_json_adjacency_list, "{stem}_adjacency.json", True),
        ("Pickle", export_to_python_pickle, "{stem}.pkl", False),
    ]

    for gnn_file in gnn_files:
        try:
            logger.debug(f"Processing file: {gnn_file}")

            # Parse GNN file to dictionary
            gnn_dict = _gnn_model_to_dict(str(gnn_file))

            # Create file-specific output directory
            file_output_dir = export_output_dir / gnn_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            export_success = True
            export_errors: list[str] = []
            for label, writer, template, needs_nx in writer_table:
                if needs_nx and not HAS_NETWORKX:
                    continue
                out_path = template.format(stem=gnn_file.stem)
                out_target = file_output_dir / out_path
                try:
                    result = writer(gnn_dict, str(out_target))
                except Exception as e:  # defensive: writers shouldn't raise
                    export_success = False
                    export_errors.append(f"{label}: {e}")
                    continue
                if not _writer_success(result):
                    export_success = False
                    msg = _writer_error(result)
                    export_errors.append(
                        f"{label}: {msg}" if msg else f"{label}: writer returned False"
                    )

            if export_success:
                success_count += 1
                logger.debug(f"Successfully exported {gnn_file.name}")
            else:
                log_step_warning(
                    logger,
                    f"Some exports failed for {gnn_file.name}: {'; '.join(export_errors)}",
                )

        except Exception as e:
            log_step_error(logger, f"Failed to export {gnn_file}: {e}")

    # Log summary
    logger.info(
        f"Export completed: {success_count}/{total_files} files exported successfully"
    )

    if success_count == total_files:
        log_step_success(logger, "All GNN files exported successfully")
        return True
    elif success_count > 0:
        log_step_warning(
            logger, f"Partial success: {success_count}/{total_files} files exported"
        )
        return False
    else:
        log_step_error(logger, "No files were exported successfully")
        return False
