"""
Advanced Visualization Processor - Core Processing Logic

This module provides advanced visualization capabilities including:
- 3D network visualizations
- Interactive dashboards
- Statistical analysis plots
- Multi-format export support
- Comprehensive error handling and fallback mechanisms
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
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


@dataclass
class AdvancedVisualizationAttempt:
    """Track individual visualization attempts"""
    viz_type: str
    model_name: str
    status: str  # "success", "failed", "skipped"
    duration_ms: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass
class AdvancedVisualizationResults:
    """Aggregate results for advanced visualization processing"""
    total_attempts: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    attempts: List[AdvancedVisualizationAttempt] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


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


def process_advanced_viz_standardized_impl(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    viz_type: str = "all",
    interactive: bool = False,
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
                
                # Generate visualizations based on type
                if viz_type in ["all", "3d"]:
                    attempt = _generate_3d_visualization(
                        model_name, model_data, output_dir, 
                        export_formats, dependencies_available, logger
                    )
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
                
                if viz_type in ["all", "dashboard"] and interactive:
                    attempt = _generate_interactive_dashboard(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
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
                
                # Generate D2 diagrams for each model
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


def _check_dependencies(logger: logging.Logger) -> Dict[str, bool]:
    """Check availability of visualization dependencies"""
    global MATPLOTLIB_AVAILABLE
    dependencies = {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "plotly": False,
        "seaborn": False,
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

    # Check seaborn
    global SEABORN_AVAILABLE
    try:
        import seaborn
        dependencies["seaborn"] = True
        SEABORN_AVAILABLE = True
    except ImportError:
        logger.debug("seaborn not available - will use matplotlib fallback")
        SEABORN_AVAILABLE = False

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
                        logger.warning(f"Parsed model file not found: {parsed_file}")
            else:
                logger.warning(f"Skipping failed parse result: {result.get('file_name', 'unknown')}")

        return models

    except Exception as e:
        logger.error(f"Failed to load GNN models: {e}")
        return {}


def _generate_3d_visualization(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate 3D network visualization"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="3d",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        # Generate 3D visualization using matplotlib for now (can be enhanced with plotly later)
        if MATPLOTLIB_AVAILABLE and plt and np:
            # Create a 3D scatter plot representing the model structure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Extract model components for visualization
            variables = model_data.get("variables", [])
            connections = model_data.get("connections", [])

            # Validate data quality before proceeding
            validation_results = validate_visualization_data(model_data, logger)

            if not validation_results["overall_valid"]:
                logger.warning(f"Poor data quality for 3D visualization of {model_name}")
                if validation_results["errors"]:
                    logger.error(f"Errors: {validation_results['errors']}")
                if validation_results["warnings"]:
                    logger.warning(f"Warnings: {validation_results['warnings']}")
                attempt.status = "failed"
                attempt.error_message = f"Data validation failed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
                return attempt

            if variables:
                # Create meaningful 3D positions for variables based on their semantic relationships
                n_vars = len(variables)

                # Use a force-directed layout algorithm for better positioning
                # Position variables based on their types and dimensions
                positions = _calculate_semantic_positions(variables, connections)

                # Color code by variable type with better color mapping
                type_color_map = {
                    'likelihood_matrix': '#FF6B6B',  # Red
                    'transition_matrix': '#4ECDC4',  # Teal
                    'preference_vector': '#45B7D1',  # Blue
                    'prior_vector': '#96CEB4',       # Green
                    'hidden_state': '#FECA57',       # Yellow
                    'observation': '#FF9FF3',        # Pink
                    'policy': '#A8E6CF',             # Mint
                    'action': '#DCE9BE',             # Light green
                    'unknown': '#CCCCCC'             # Gray
                }

                # Calculate node sizes based on dimensions and connections
                node_sizes = []
                node_colors = []

                for var in variables:
                    var_name = var.get("name", "unknown")
                    var_type = var.get("var_type", "unknown")
                    dimensions = var.get("dimensions", [])

                    # Size based on dimensionality and importance
                    base_size = 50
                    if isinstance(dimensions, list) and len(dimensions) > 0:
                        # Larger matrices get bigger nodes
                        dim_product = 1
                        for dim in dimensions[:2]:  # Use first two dimensions for size calculation
                            if isinstance(dim, (int, float)):
                                dim_product *= dim
                        size_multiplier = min(3, max(0.5, dim_product / 10))
                        base_size *= size_multiplier

                    node_sizes.append(base_size)
                    node_colors.append(type_color_map.get(var_type, '#CCCCCC'))

                # Plot variables as 3D points with proper labels
                for i, (pos, color, size) in enumerate(zip(positions, node_colors, node_sizes)):
                    ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.8, edgecolors='black')
                    ax.text(pos[0], pos[1], pos[2], variables[i].get("name", f"Var{i}"),
                           fontsize=8, ha='center', va='center', fontweight='bold')

                # Add real connections from the model data
                if connections:
                    for conn_info in connections:
                        # Normalize connection format
                        normalized_conn = _normalize_connection_format(conn_info)
                        source_vars = normalized_conn.get("source_variables", [])
                        target_vars = normalized_conn.get("target_variables", [])

                        for source_var in source_vars:
                            for target_var in target_vars:
                                if source_var != target_var:
                                    # Find positions of source and target
                                    source_idx = None
                                    target_idx = None

                                    for idx, var in enumerate(variables):
                                        if var.get("name") == source_var:
                                            source_idx = idx
                                        if var.get("name") == target_var:
                                            target_idx = idx

                                    if source_idx is not None and target_idx is not None:
                                        source_pos = positions[source_idx]
                                        target_pos = positions[target_idx]

                                        # Draw connection line
                                        ax.plot([source_pos[0], target_pos[0]],
                                               [source_pos[1], target_pos[1]],
                                               [source_pos[2], target_pos[2]],
                                               'gray', alpha=0.5, linewidth=2)

                                        # Add arrow head at target
                                        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                                                 c='red', s=30, marker='>', alpha=0.7)

                ax.set_xlabel('X Dimension')
                ax.set_ylabel('Y Dimension')
                ax.set_zlabel('Z Dimension')
                ax.set_title(f'3D Model Structure: {model_name}')

                plt.tight_layout()

                # Save the 3D visualization
                output_file = output_dir / f"{model_name}_3d_visualization.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()

                attempt.status = "success"
                attempt.output_files.append(str(output_file))
                logger.info(f"Generated 3D visualization: {output_file}")
            else:
                logger.info(f"No variables found for 3D visualization of {model_name}")
                attempt.status = "skipped"

        elif not dependencies.get("plotly"):
            logger.info(f"Skipping 3D visualization for {model_name} (plotly not available)")
            attempt.fallback_used = True
            # Generate fallback HTML report
            _generate_fallback_report(model_name, "3d", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(str(output_dir / f"{model_name}_3d_fallback.html"))
        else:
            logger.info(f"3D visualization for {model_name} - using matplotlib fallback")
            attempt.fallback_used = True
            # Generate fallback HTML report
            _generate_fallback_report(model_name, "3d", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(str(output_dir / f"{model_name}_3d_fallback.html"))

    except Exception as e:
        logger.error(f"Failed to generate 3D visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_interactive_dashboard(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate interactive dashboard"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="dashboard",
        model_name=model_name,
        status="skipped"
    )
    
    start_time = time.time()
    
    try:
        if not (dependencies.get("plotly") or dependencies.get("bokeh")):
            logger.info(f"Skipping dashboard for {model_name} (no interactive libraries available)")
            attempt.fallback_used = True
            # Generate fallback HTML report
            _generate_fallback_report(model_name, "dashboard", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(str(output_dir / f"{model_name}_dashboard_fallback.html"))
        else:
            # TODO: Implement actual dashboard
            logger.info(f"Interactive dashboard for {model_name} not yet implemented")
            attempt.status = "skipped"
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000
    
    return attempt


def _normalize_connection_format(conn_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize connection format to handle both old and new formats."""
    if "source_variables" in conn_info and "target_variables" in conn_info:
        # New format with arrays
        return conn_info
    elif "source" in conn_info and "target" in conn_info:
        # Old format with single values
        return {
            "source_variables": [conn_info["source"]],
            "target_variables": [conn_info["target"]],
            **{k: v for k, v in conn_info.items() if k not in ["source", "target"]}
        }
    else:
        # Unknown format - return as-is
        return conn_info

def _calculate_semantic_positions(variables: List[Dict], connections: List[Dict]) -> np.ndarray:
    """
    Calculate meaningful 3D positions for variables based on semantic relationships.

    Args:
        variables: List of variable dictionaries
        connections: List of connection dictionaries

    Returns:
        Array of 3D positions for each variable
    """
    if not variables:
        return np.array([])

    n_vars = len(variables)
    positions = np.zeros((n_vars, 3))

    # Create a simple force-directed layout based on connections
    # Start with random positions
    np.random.seed(42)  # For reproducible results
    positions = np.random.rand(n_vars, 3) * 10

    # Build connection matrix
    var_names = [var.get("name", f"var_{i}") for i, var in enumerate(variables)]
    connection_matrix = np.zeros((n_vars, n_vars))

    for conn_info in connections:
        # Normalize connection format
        normalized_conn = _normalize_connection_format(conn_info)
        source_vars = normalized_conn.get("source_variables", [])
        target_vars = normalized_conn.get("target_variables", [])

        for source_var in source_vars:
            for target_var in target_vars:
                if source_var != target_var:
                    source_idx = None
                    target_idx = None

                    for idx, name in enumerate(var_names):
                        if name == source_var:
                            source_idx = idx
                        if name == target_var:
                            target_idx = idx

                    if source_idx is not None and target_idx is not None:
                        connection_matrix[source_idx, target_idx] = 1

    # Simple force-directed layout simulation (simplified)
    for _ in range(50):  # 50 iterations
        forces = np.zeros_like(positions)

        # Repulsive forces between all nodes
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += (diff / distance) * (1 / distance)

        # Attractive forces along connections
        for i in range(n_vars):
            for j in range(n_vars):
                if connection_matrix[i, j] > 0:
                    diff = positions[j] - positions[i]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += diff * (distance / 10)  # Spring force

        # Update positions
        positions += forces * 0.01

    # Normalize positions to a reasonable range
    positions = (positions - positions.min()) / (positions.max() - positions.min()) * 10

    return positions

# Removed _generate_statistical_plots (moved to Step 16 analysis)


def _generate_fallback_report(
    model_name: str,
    viz_type: str,
    output_dir: Path,
    model_data: Dict,
    logger: logging.Logger
):
    """Generate fallback HTML report when advanced libraries unavailable"""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{model_name} - {viz_type.upper()} Visualization (Fallback)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        .data {{ background: #fff; border: 1px solid #ddd; padding: 10px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{model_name} - {viz_type.upper()} Visualization</h1>
    <div class="info">
        <p><strong>Note:</strong> Advanced visualization libraries not available. 
        Showing basic model information instead.</p>
    </div>
    <div class="data">
        <h2>Model Structure</h2>
        <pre>{json.dumps(model_data, indent=2)}</pre>
    </div>
</body>
</html>"""
    
    output_file = output_dir / f"{model_name}_{viz_type}_fallback.html"
    with open(output_file, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated fallback report: {output_file}")


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
                logger.debug(f"    • {feature}")
            if len(features) > 3:
                logger.debug(f"    ... and {len(features)-3} more")


def validate_visualization_data(model_data: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate that visualization data is complete and meaningful.

    Args:
        model_data: Parsed model data
        logger: Logger instance

    Returns:
        Validation results dictionary
    """
    validation_results = {
        "overall_valid": True,
        "warnings": [],
        "errors": [],
        "data_quality": {},
        "recommendations": []
    }

    try:
        # Check basic structure
        if not isinstance(model_data, dict):
            validation_results["errors"].append("Model data is not a dictionary")
            validation_results["overall_valid"] = False
            return validation_results

        # Check for required keys
        required_keys = ["variables", "connections"]
        for key in required_keys:
            if key not in model_data:
                validation_results["warnings"].append(f"Missing key: {key}")
            elif not model_data[key]:
                validation_results["warnings"].append(f"Empty data for key: {key}")

        # Validate variables
        variables = model_data.get("variables", [])
        if not variables:
            validation_results["errors"].append("No variables found in model")
            validation_results["overall_valid"] = False
        else:
            validation_results["data_quality"]["total_variables"] = len(variables)

            # Check variable structure
            valid_vars = 0
            for var in variables:
                if isinstance(var, dict) and "name" in var and "var_type" in var:
                    valid_vars += 1
                else:
                    validation_results["warnings"].append(f"Invalid variable structure: {var}")

            validation_results["data_quality"]["valid_variables"] = valid_vars
            validation_results["data_quality"]["variable_validity_rate"] = valid_vars / len(variables)

            if valid_vars < len(variables) * 0.8:  # Less than 80% valid
                validation_results["warnings"].append("Low variable validity rate")

        # Validate connections
        connections = model_data.get("connections", [])
        if not connections:
            validation_results["warnings"].append("No connections found in model")
        else:
            validation_results["data_quality"]["total_connections"] = len(connections)

            # Check connection structure
            valid_connections = 0
            for conn in connections:
                if isinstance(conn, dict) and ("source_variables" in conn or "target_variables" in conn):
                    valid_connections += 1
                else:
                    validation_results["warnings"].append(f"Invalid connection structure: {conn}")

            validation_results["data_quality"]["valid_connections"] = valid_connections
            validation_results["data_quality"]["connection_validity_rate"] = valid_connections / len(connections)

        # Check for POMDP-specific structure
        pomdp_indicators = {
            "likelihood_matrix": 0,
            "transition_matrix": 0,
            "preference_vector": 0,
            "prior_vector": 0,
            "hidden_state": 0,
            "observation": 0,
            "policy": 0
        }

        for var in variables:
            if isinstance(var, dict):
                var_type = var.get("var_type", "")
                for indicator in pomdp_indicators:
                    if indicator in var_type:
                        pomdp_indicators[indicator] += 1

        validation_results["data_quality"]["pomdp_indicators"] = pomdp_indicators

        # Validate if this looks like a proper POMDP model
        pomdp_score = sum(pomdp_indicators.values())
        if pomdp_score >= 3:  # At least 3 POMDP indicators
            validation_results["data_quality"]["is_pomdp_model"] = True
            validation_results["data_quality"]["pomdp_completeness"] = pomdp_score / len(pomdp_indicators)
        else:
            validation_results["data_quality"]["is_pomdp_model"] = False
            validation_results["warnings"].append("Model does not appear to be a complete POMDP")

        # Generate recommendations
        if validation_results["data_quality"].get("variable_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append("Review variable parsing - high invalidity rate")

        if validation_results["data_quality"].get("connection_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append("Review connection parsing - high invalidity rate")

        if pomdp_score < 3:
            validation_results["recommendations"].append("Model may not be a complete POMDP - check GNN structure")

        # Overall assessment
        if validation_results["errors"]:
            validation_results["overall_valid"] = False
        elif len(validation_results["warnings"]) > 2:
            validation_results["overall_valid"] = False
            validation_results["warnings"].append("Too many warnings - data quality may be poor")

        if logger:
            logger.info(f"Validation completed: {validation_results['overall_valid']} (errors: {len(validation_results['errors'])}, warnings: {len(validation_results['warnings'])})")

    except Exception as e:
        validation_results["errors"].append(f"Validation error: {e}")
        validation_results["overall_valid"] = False
        if logger:
            logger.error(f"Validation failed: {e}")

    return validation_results


# Removed _generate_state_transitions (moved to Step 16 analysis)


# Removed _generate_belief_evolution (moved to Step 16 analysis)


# Removed _generate_policy_visualization (moved to Step 16 analysis)


# Removed _generate_matrix_correlations (moved to Step 16 analysis)


# Removed _generate_timeline_visualization (moved to Step 16 analysis)


# Removed _generate_state_space_analysis (moved to Step 16 analysis)


# Removed _generate_belief_flow_visualization (moved to Step 16 analysis)


def _generate_d2_visualizations_safe(
    model_data: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """
    Generate D2 diagram visualizations for GNN models.
    
    Args:
        model_data: Parsed GNN model data
        output_dir: Output directory for visualizations
        logger: Logger instance
        
    Returns:
        AdvancedVisualizationAttempt tracking the generation
    """
    model_name = model_data.get("model_name", "unknown_model")
    attempt = AdvancedVisualizationAttempt(
        viz_type="d2_diagrams",
        model_name=model_name,
        status="in_progress"
    )
    
    start_time = time.time()
    
    try:
        # Import D2 visualizer
        try:
            from .d2_visualizer import D2Visualizer
            d2_available = True
        except ImportError:
            logger.warning("D2 visualizer module not available")
            attempt.status = "skipped"
            attempt.error_message = "D2 visualizer not available"
            return attempt
        
        logger.info(f"Generating D2 diagrams for {model_name}...")
        
        # Create D2 visualizer
        visualizer = D2Visualizer(logger=logger)
        
        if not visualizer.d2_available:
            logger.warning("D2 CLI not available. Install from https://d2lang.com")
            attempt.status = "skipped"
            attempt.error_message = "D2 CLI not installed"
            attempt.fallback_used = True
            return attempt
        
        # Create D2 output directory
        d2_output_dir = output_dir / "d2_diagrams" / model_name
        d2_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all applicable D2 diagrams
        results = visualizer.generate_all_diagrams_for_model(
            model_data,
            d2_output_dir,
            formats=["svg", "png"]
        )
        
        # Process results
        successful = 0
        for result in results:
            if result.success:
                successful += 1
                for output_file in result.output_files:
                    attempt.output_files.append(str(output_file))
                logger.info(f"Generated D2 diagram: {result.diagram_name}")
            else:
                logger.warning(f"Failed D2 diagram {result.diagram_name}: {result.error_message}")
        
        if successful > 0:
            attempt.status = "success"
            logger.info(f"Generated {successful} D2 diagrams for {model_name}")
        else:
            attempt.status = "failed"
            attempt.error_message = "No D2 diagrams generated successfully"
        
    except Exception as e:
        logger.error(f"Failed to generate D2 visualizations for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000
    
    return attempt


def _generate_pipeline_d2_diagrams_safe(
    output_dir: Path,
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """
    Generate D2 diagrams for GNN pipeline architecture.
    
    Args:
        output_dir: Output directory for visualizations
        logger: Logger instance
        
    Returns:
        AdvancedVisualizationAttempt tracking the generation
    """
    attempt = AdvancedVisualizationAttempt(
        viz_type="d2_pipeline_diagrams",
        model_name="gnn_pipeline",
        status="in_progress"
    )
    
    start_time = time.time()
    
    try:
        # Import D2 visualizer
        try:
            from .d2_visualizer import D2Visualizer
            d2_available = True
        except ImportError:
            logger.warning("D2 visualizer module not available")
            attempt.status = "skipped"
            attempt.error_message = "D2 visualizer not available"
            return attempt
        
        logger.info("Generating pipeline D2 diagrams...")
        
        # Create D2 visualizer
        visualizer = D2Visualizer(logger=logger)
        
        if not visualizer.d2_available:
            logger.warning("D2 CLI not available. Install from https://d2lang.com")
            attempt.status = "skipped"
            attempt.error_message = "D2 CLI not installed"
            attempt.fallback_used = True
            return attempt
        
        # Create D2 output directory
        d2_output_dir = output_dir / "d2_diagrams" / "pipeline"
        d2_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate pipeline flow diagram
        flow_spec = visualizer.generate_pipeline_flow_diagram(include_frameworks=True)
        flow_result = visualizer.compile_d2_diagram(
            flow_spec,
            d2_output_dir,
            formats=["svg", "png"]
        )
        
        if flow_result.success:
            for output_file in flow_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated pipeline flow diagram")
        
        # Generate framework mapping diagram
        framework_spec = visualizer.generate_framework_mapping_diagram()
        framework_result = visualizer.compile_d2_diagram(
            framework_spec,
            d2_output_dir,
            formats=["svg", "png"]
        )
        
        if framework_result.success:
            for output_file in framework_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated framework mapping diagram")
        
        # Generate Active Inference concepts diagram
        concepts_spec = visualizer.generate_active_inference_concepts_diagram()
        concepts_result = visualizer.compile_d2_diagram(
            concepts_spec,
            d2_output_dir,
            formats=["svg", "png"]
        )
        
        if concepts_result.success:
            for output_file in concepts_result.output_files:
                attempt.output_files.append(str(output_file))
            logger.info("Generated Active Inference concepts diagram")
        
        # Check overall success
        total_results = [flow_result, framework_result, concepts_result]
        successful = sum(1 for r in total_results if r.success)
        
        if successful > 0:
            attempt.status = "success"
            logger.info(f"Generated {successful} pipeline D2 diagrams")
        else:
            attempt.status = "failed"
            attempt.error_message = "No pipeline D2 diagrams generated successfully"
        
    except Exception as e:
        logger.error(f"Failed to generate pipeline D2 diagrams: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000
    
    return attempt

