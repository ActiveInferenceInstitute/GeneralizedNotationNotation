#!/usr/bin/env python3
"""
Integration Processor module for GNN Processing Pipeline.

This module composes the pure parsing/graph primitives in
``integration/parsing.py`` and ``integration/graph.py`` into the single
pipeline entry point :func:`process_integration`, which also orchestrates the
meta-analysis submodule when Step 12 execution outputs exist.

Output contract (unchanged since Step 17's introduction):

- ``<output_dir>/integration_results/integration_results.json``
- ``<output_dir>/integration_results/integration_summary.md``
- ``<output_dir>/integration_results/meta_analysis/`` (when execution
  outputs are found; failures here are warnings, never step failures)
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from utils.pipeline_template import log_step_error, log_step_start, log_step_success

from .graph import build_system_graph, verify_references
from .parsing import discover_gnn_files

logger = logging.getLogger(__name__)


def _locate_pipeline_dir(output_dir: Path, dirname: str) -> Path | None:
    """Locate a sibling pipeline output directory (e.g. ``12_execute_output``).

    Checks ``output_dir.parent/<dirname>`` first, then a few conventional
    fallbacks relative to ``output_dir`` and the working directory. Returns
    ``None`` when no candidate exists.
    """
    candidates = [
        output_dir.parent / dirname,
        output_dir / ".." / dirname,
        Path("output") / dirname,
    ]
    for candidate in candidates:
        try:
            if candidate.resolve().exists():
                return candidate.resolve()
        except OSError:  # pragma: no cover - defensive
            continue
    return None


def _render_summary(
    scanned_file_count: int,
    stats: dict[str, int],
    issues: list[str],
    meta_analysis_results: dict[str, Any] | None,
) -> str:
    """Render the human-readable ``integration_summary.md`` content."""
    node_count = stats.get("nodes", 0)
    edge_count = stats.get("edges", 0)
    summary = "# System Integration Report\n\n"
    summary += f"Scanned {scanned_file_count} files.\n\n"
    summary += f"- **Graph Nodes**: {node_count}\n"
    summary += f"- **Graph Edges**: {edge_count}\n"
    if stats.get("cycles", 0) > 0:
        summary += f"- **Cycles Detected**: {stats['cycles']}\n"
    if stats.get("isolated_nodes", 0) > 0:
        summary += f"- **Isolated Nodes**: {stats['isolated_nodes']}\n"
    if issues:
        summary += f"\n## Issues ({len(issues)})\n\n"
        for issue in issues:
            summary += f"- {issue}\n"
    else:
        summary += "\nNo issues detected.\n"

    if meta_analysis_results:
        summary += "\n## Meta-Analysis\n\n"
        summary += f"- **Sweep Records**: {meta_analysis_results['records']}\n"
        summary += f"- **Visualizations**: {len(meta_analysis_results['plots'])}\n"
        summary += (
            f"- **Report**: [{Path(meta_analysis_results['report']).name}]"
            f"({meta_analysis_results['report']})\n"
        )
        vj = meta_analysis_results.get("validation_json")
        sj = meta_analysis_results.get("statistics_json")
        if vj:
            summary += f"- **Validation JSON**: [{Path(vj).name}]({vj})\n"
        if sj:
            summary += f"- **Aggregate statistics JSON**: [{Path(sj).name}]({sj})\n"
    return summary


def process_integration(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process integration for GNN files.

    This module performs system-level consistency checks, builds a dependency
    graph of components, and detects circular dependencies or isolated
    components.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    step_logger = logging.getLogger("integration")

    try:
        log_step_start(step_logger, "Processing integration")

        # Create results directory (with integration_results subdirectory)
        results_dir = output_dir / "integration_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "processed_files": 0,
            "success": True,
            "errors": [],
            "system_graph_stats": {},
            "issues": [],
        }

        # Search for GNN files in multiple locations
        gnn_files = discover_gnn_files(target_dir)
        results["processed_files"] = len(gnn_files)

        if verbose:
            step_logger.debug(
                "Processing %d GNN files for integration analysis", len(gnn_files)
            )

        # Build system graph + structural analysis
        analysis = build_system_graph(gnn_files, logger=step_logger, verbose=verbose)
        results["system_graph_stats"] = analysis.stats.to_dict()
        results["issues"].extend(analysis.issues)

        # Verify cross-references
        results["issues"].extend(
            verify_references(gnn_files, analysis.component_locations, step_logger)
        )

        # ── Meta-Analysis: Runtime & Simulation Sweep Analysis ──────────────
        meta_analysis_results = None
        try:
            from .meta_analysis import run_meta_analysis

            execute_output_dir = _locate_pipeline_dir(output_dir, "12_execute_output")
            if execute_output_dir is not None:
                meta_output = results_dir / "meta_analysis"
                render_output_dir = _locate_pipeline_dir(output_dir, "11_render_output")

                meta_analysis_results = run_meta_analysis(
                    execute_output_dir=execute_output_dir,
                    output_dir=meta_output,
                    render_output_dir=render_output_dir,
                    logger=step_logger,
                    verbose=verbose,
                )
                if meta_analysis_results:
                    results["meta_analysis"] = meta_analysis_results
                    val_json = meta_analysis_results.get("validation_json")
                    stats_json = meta_analysis_results.get("statistics_json")
                    step_logger.info(
                        "Meta-analysis: %s records, %s plots; validation=%s; statistics=%s",
                        meta_analysis_results["records"],
                        len(meta_analysis_results["plots"]),
                        val_json,
                        stats_json,
                    )
            else:
                step_logger.info("No execution outputs found — skipping meta-analysis")
        except Exception as e:
            step_logger.warning(f"Meta-analysis failed (non-fatal): {e}")

        # Save results
        results_file = results_dir / "integration_results.json"
        with open(results_file, "w") as output_file:
            json.dump(results, output_file, indent=2, default=str)

        # Generate summary
        summary = _render_summary(
            len(gnn_files),
            results["system_graph_stats"],
            results["issues"],
            meta_analysis_results,
        )
        summary_path = results_dir / "integration_summary.md"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=summary_path.parent, delete=False
        ) as tmp_f:
            tmp_f.write(summary)
        os.replace(tmp_f.name, str(summary_path))

        if results["success"]:
            log_step_success(
                step_logger,
                f"Integration processing completed: {results['system_graph_stats'].get('nodes', 0)} nodes, "
                f"{results['system_graph_stats'].get('edges', 0)} edges",
            )
        else:
            log_step_error(step_logger, "integration processing failed")

        return bool(results["success"])

    except Exception as e:
        log_step_error(step_logger, "integration processing failed", error=str(e))
        return False
