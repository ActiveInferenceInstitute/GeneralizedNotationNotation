"""
Advanced Visualization Processor - Core Processing Logic

This module provides advanced visualization capabilities including:
- 3D network visualizations
- Interactive dashboards
- Statistical analysis plots
- Multi-format export support
- Comprehensive error handling and fallback mechanisms

Implementation is split across sub-modules for maintainability:
- _shared: Dataclasses, validation, and utility functions (no circular imports)
- network_viz: 3D visualization, interactive dashboards, network metrics, D2 diagrams
- statistical_viz: Statistical plots, matrix correlations, Plotly dashboards

This file re-exports all public names for backward compatibility.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime

# Import matplotlib for plotting (with fallback for headless environments)
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

# Import performance tracker with fallback
try:
    from utils.performance_tracker import PerformanceTracker
except ImportError:
    try:
        from utils import performance_tracker
        PerformanceTracker = performance_tracker.PerformanceTracker
    except (ImportError, AttributeError):
        # Fallback: simple performance tracker
        class PerformanceTracker:
            def __init__(self):
                self.timings = {}

            def start_timing(self, name: str):
                self.timings[name] = time.time()

            def stop_timing(self, name: str) -> float:
                if name in self.timings:
                    duration = time.time() - self.timings[name]
                    del self.timings[name]
                    return duration
                return 0.0

# Re-export shared items for backward compatibility
from ._shared import (
    AdvancedVisualizationAttempt,
    AdvancedVisualizationResults,
    _normalize_connection_format,
    _calculate_semantic_positions,
    _generate_fallback_report,
    validate_visualization_data,
)


class SafeAdvancedVisualizationManager:
    """Context manager for safe advanced visualization with automatic cleanup"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.tracker = PerformanceTracker()
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.warning(f"Advanced visualization encountered error: {exc_val}")
        return False  # Don't suppress exceptions


# Re-export from network_viz
from .network_viz import (
    _generate_3d_visualization,
    _generate_interactive_dashboard,
    _generate_network_metrics,
    _generate_pomdp_transition_analysis,
    _generate_policy_visualization,
    _generate_d2_visualizations_safe,
    _generate_pipeline_d2_diagrams_safe,
)

# Re-export from statistical_viz
from .statistical_viz import (
    _generate_statistical_plots,
    _generate_matrix_correlations,
    _generate_interactive_plotly_dashboard,
)


# Global seaborn availability flag
SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None

def _check_dependencies(logger: logging.Logger) -> Dict[str, bool]:
    """Check availability of visualization dependencies"""
    global MATPLOTLIB_AVAILABLE, SEABORN_AVAILABLE
    dependencies = {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "plotly": False,
        "seaborn": SEABORN_AVAILABLE,
        "bokeh": False,
        "numpy": False
    }

    if not MATPLOTLIB_AVAILABLE:
        logger.info("matplotlib not available - some visualizations will be skipped")

    # Check plotly
    try:
        import plotly
        dependencies["plotly"] = True
    except ImportError:
        logger.info("plotly not available - interactive visualizations will be limited")

    # Check seaborn (already checked globally)
    if not SEABORN_AVAILABLE:
        logger.debug("seaborn not available - will use matplotlib fallback")

    # Check bokeh
    try:
        import bokeh
        dependencies["bokeh"] = True
    except ImportError:
        logger.debug("bokeh not available - will use plotly fallback")

    # Check numpy
    if np is not None:
        dependencies["numpy"] = True
    else:
        logger.info("numpy not available - numeric visualizations will be limited")

    return dependencies


def _load_gnn_models(target_dir: Path, logger: logging.Logger, base_output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Load GNN models from processing results"""
    from pipeline.config import get_output_dir_for_script

    # Get GNN output directory
    if base_output_dir is None:
        base_output_dir = Path("output")
    gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)

    logger.info(f"Looking for GNN output in: {gnn_output_dir}")

    # Check for double-nested directory structure
    results_file = gnn_output_dir / "gnn_processing_results.json"
    logger.info(f"Looking for results file: {results_file} (exists: {results_file.exists()})")

    if not results_file.exists():
        # Try nested structure
        nested_results_file = gnn_output_dir / "3_gnn_output" / "gnn_processing_results.json"
        logger.info(f"Looking for nested results file: {nested_results_file} (exists: {nested_results_file.exists()})")
        if nested_results_file.exists():
            results_file = nested_results_file
            gnn_output_dir = gnn_output_dir / "3_gnn_output"

    if not results_file.exists():
        logger.warning(f"GNN processing results not found at {results_file}")
        # Try to find any parsed JSON files in the GNN output directory
        parsed_files = list(gnn_output_dir.glob("**/*_parsed.json"))
        logger.info(f"Found {len(parsed_files)} parsed files in {gnn_output_dir}")
        if parsed_files:
            logger.info(f"Found {len(parsed_files)} parsed files, loading directly")
            models = {}
            for parsed_file in parsed_files:
                logger.info(f"Processing parsed file: {parsed_file}")
                try:
                    with open(parsed_file) as f:
                        model_data = json.load(f)
                    model_name = parsed_file.stem.replace("_parsed", "")
                    models[model_name] = model_data
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {parsed_file}: {e}")
            return models
        return {}

    # Load results
    try:
        with open(results_file) as f:
            processing_results = json.load(f)

        models = {}
        # The results file uses "processed_files" not "results"
        processed_files = processing_results.get("processed_files", [])
        logger.info(f"Found {len(processed_files)} processed files in results")

        for result in processed_files:
            if result.get("parse_success"):  # Note: it's "parse_success" not "parsing_success"
                parsed_model_file = result.get("parsed_model_file")
                if parsed_model_file and parsed_model_file.endswith("_parsed.json"):
                    # Extract model name from file path
                    model_name = parsed_model_file.split("/")[-1].replace("_parsed.json", "")

                    # Construct full path to parsed file
                    parsed_file = Path(parsed_model_file)

                    if parsed_file.exists():
                        try:
                            with open(parsed_file) as f:
                                model_data = json.load(f)
                            models[model_name] = model_data
                            logger.info(f"Loaded parsed model: {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load model {model_name} from {parsed_file}: {e}")
                    else:
                        # Try to resolve relative to gnn_output_dir
                        # The JSON contains "output_directory" which is the root for these files usually
                        json_out_dir = processing_results.get("output_directory")
                        if json_out_dir and str(parsed_model_file).startswith(str(json_out_dir)):
                            rel_path = str(parsed_model_file)[len(str(json_out_dir)):].lstrip("/")
                            parsed_file = gnn_output_dir / rel_path

                        if parsed_file.exists():
                            try:
                                with open(parsed_file) as f:
                                    model_data = json.load(f)
                                models[model_name] = model_data
                                logger.info(f"Loaded parsed model (path resolved): {model_name}")
                            except Exception as e:
                                logger.warning(f"Failed to load model {model_name} from {parsed_file}: {e}")
                        else:
                            logger.warning(f"Parsed model file not found: {parsed_file}")
            else:
                logger.warning(f"Skipping failed parse result: {result.get('file_name', 'unknown')}")

        return models

    except Exception as e:
        logger.error(f"Failed to load GNN models: {e}")
        return {}


def _save_results(output_dir: Path, results: AdvancedVisualizationResults, logger: logging.Logger):
    """Save visualization results to JSON with detailed skipped feature tracking"""
    # Categorize skipped visualizations by reason
    skipped_by_reason = {}
    for attempt in results.attempts:
        if attempt.status == "skipped":
            reason = attempt.error_message or "Unknown reason"
            if reason not in skipped_by_reason:
                skipped_by_reason[reason] = []
            skipped_by_reason[reason].append(f"{attempt.viz_type}:{attempt.model_name}")

    # Build the summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_attempts": results.total_attempts,
        "successful": results.successful,
        "failed": results.failed,
        "skipped": results.skipped,
        "total_duration_ms": results.total_duration_ms,
        "output_files": results.output_files,
        "warnings": results.warnings,
        "errors": results.errors,
        "skipped_features": {
            "count": results.skipped,
            "by_reason": skipped_by_reason,
            "details": [
                {
                    "feature": f"{a.viz_type}:{a.model_name}",
                    "reason": a.error_message or "Unknown",
                    "fallback_available": a.fallback_used
                }
                for a in results.attempts
                if a.status == "skipped"
            ]
        },
        "attempts": [
            {
                "viz_type": a.viz_type,
                "model_name": a.model_name,
                "status": a.status,
                "duration_ms": a.duration_ms,
                "output_files": a.output_files,
                "error_message": a.error_message,
                "fallback_used": a.fallback_used
            }
            for a in results.attempts
        ]
    }

    output_file = output_dir / "advanced_viz_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved advanced visualization summary: {output_file}")

    # Log detailed skipped feature report if there are skipped items
    if skipped_by_reason:
        logger.info(f"Skipped visualization features ({results.skipped} total):")
        for reason, features in skipped_by_reason.items():
            logger.info(f"  - {reason}: {len(features)} feature(s)")
            for feature in features[:3]:  # Show first 3 examples
                logger.debug(f"    * {feature}")
            if len(features) > 3:
                logger.debug(f"    ... and {len(features)-3} more")


def process_advanced_viz_standardized_impl(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    viz_type: str = "all",
    interactive: bool = True,
    export_formats: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """
    Main advanced visualization processing function.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for visualizations
        logger: Logger instance
        viz_type: Type of visualization ("all", "3d", "interactive", "dashboard")
        interactive: Enable interactive visualizations
        export_formats: List of export formats ["html", "json", "png"]
        **kwargs: Additional arguments

    Returns:
        True if processing succeeded (with possible warnings)
    """
    logger.info("=" * 80)
    logger.info("ADVANCED VISUALIZATION PROCESSING")
    logger.info("=" * 80)

    # Initialize results
    results = AdvancedVisualizationResults()

    # Set default export formats
    if export_formats is None:
        export_formats = ["html", "json"]

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required dependencies
    dependencies_available = _check_dependencies(logger)

    try:
        with SafeAdvancedVisualizationManager(logger) as manager:
            # Load GNN processing results
            gnn_models = _load_gnn_models(target_dir, logger, output_dir.parent if output_dir.name.endswith("_output") else output_dir)

            if not gnn_models:
                logger.warning("No GNN models found for advanced visualization")
                results.warnings.append("No GNN models found")
                _save_results(output_dir, results, logger)
                return True  # Not a failure, just no data

            # Process each model
            for model_name, model_data in gnn_models.items():
                logger.info(f"Processing advanced visualizations for: {model_name}")

                # Helper to track attempt results
                def _track(attempt):
                    results.attempts.append(attempt)
                    results.total_attempts += 1
                    if attempt.status == "success":
                        results.successful += 1
                        results.output_files.extend(attempt.output_files)
                    elif attempt.status == "failed":
                        results.failed += 1
                        results.errors.append(attempt.error_message or "Unknown error")
                    else:
                        results.skipped += 1

                # Generate visualizations based on type
                if viz_type in ["all", "3d"]:
                    _track(_generate_3d_visualization(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    ))

                if viz_type in ["all", "statistical"]:
                    _track(_generate_statistical_plots(
                        model_name, model_data, output_dir,
                        dependencies_available, logger
                    ))

                if viz_type in ["all", "statistical"]:
                    _track(_generate_matrix_correlations(
                        model_name, model_data, output_dir,
                        dependencies_available, logger
                    ))

                if viz_type in ["all", "pomdp"]:
                    _track(_generate_pomdp_transition_analysis(
                        model_name, model_data, output_dir,
                        dependencies_available, logger
                    ))

                if viz_type in ["all", "pomdp"]:
                    _track(_generate_policy_visualization(
                        model_name, model_data, output_dir,
                        dependencies_available, logger
                    ))

                if viz_type in ["all", "interactive"] and interactive:
                    _track(_generate_interactive_plotly_dashboard(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    ))

                if viz_type in ["all", "dashboard"] and interactive:
                    _track(_generate_interactive_dashboard(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    ))

                if viz_type in ["all", "network"]:
                    _track(_generate_network_metrics(
                        model_name, model_data, output_dir,
                        dependencies_available, logger
                    ))

                # Generate D2 diagrams for each model
                if viz_type in ["all", "d2", "diagrams"]:
                    attempt = _generate_d2_visualizations_safe(
                        model_data, output_dir, logger
                    )
                    results.attempts.append(attempt)
                    results.total_attempts += 1
                    if attempt.status == "success":
                        results.successful += 1
                        results.output_files.extend(attempt.output_files)
                    elif attempt.status == "failed":
                        results.failed += 1
                        if attempt.error_message:
                            results.errors.append(attempt.error_message)
                    else:
                        results.skipped += 1
                        # D2 CLI is optional - don't add warnings for missing CLI
                        if attempt.error_message and "D2 CLI" not in attempt.error_message:
                            results.warnings.append(attempt.error_message)

            # Generate pipeline-level D2 diagrams (once for all models)
            if viz_type in ["all", "d2", "diagrams", "pipeline"]:
                attempt = _generate_pipeline_d2_diagrams_safe(output_dir, logger)
                results.attempts.append(attempt)
                results.total_attempts += 1
                if attempt.status == "success":
                    results.successful += 1
                    results.output_files.extend(attempt.output_files)
                elif attempt.status == "failed":
                    results.failed += 1
                    if attempt.error_message:
                        results.errors.append(attempt.error_message)
                else:
                    results.skipped += 1
                    # D2 CLI is optional - don't add warnings for missing CLI
                    if attempt.error_message and "D2 CLI" not in attempt.error_message:
                        results.warnings.append(attempt.error_message)

        # Save results
        _save_results(output_dir, results, logger)

        # Log summary
        logger.info(f"Advanced visualization complete:")
        logger.info(f"  Total attempts: {results.total_attempts}")
        logger.info(f"  Successful: {results.successful}")
        logger.info(f"  Failed: {results.failed}")
        logger.info(f"  Skipped: {results.skipped}")
        logger.info(f"  Output files: {len(results.output_files)}")

        # Return success if:
        # 1. At least some visualizations succeeded, OR
        # 2. No attempts were made (no data), OR
        # 3. Only failures are skipped optional features (no actual errors)
        return (
            results.successful > 0 or
            results.total_attempts == 0 or
            (results.failed == 0 and results.skipped > 0)
        )

    except Exception as e:
        logger.error(f"Advanced visualization processing failed: {e}")
        results.errors.append(str(e))
        _save_results(output_dir, results, logger)
        return False
