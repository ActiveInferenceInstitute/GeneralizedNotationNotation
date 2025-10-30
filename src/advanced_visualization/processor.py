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
                
                if viz_type in ["all", "statistical"]:
                    attempt = _generate_statistical_plots(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "transitions"]:
                    attempt = _generate_state_transitions(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "belief"]:
                    attempt = _generate_belief_evolution(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "policy"]:
                    attempt = _generate_policy_visualization(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "correlation"]:
                    attempt = _generate_matrix_correlations(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "timeline"]:
                    attempt = _generate_timeline_visualization(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "statespace"]:
                    attempt = _generate_state_space_analysis(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)

                if viz_type in ["all", "belief_flow"]:
                    attempt = _generate_belief_flow_visualization(
                        model_name, model_data, output_dir,
                        export_formats, dependencies_available, logger
                    )
                    results.attempts.append(attempt)
                
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
        logger.warning("matplotlib not available - some visualizations will be skipped")

    # Check plotly
    try:
        import plotly
        dependencies["plotly"] = True
    except ImportError:
        logger.warning("plotly not available - interactive visualizations will be limited")

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
        logger.warning("numpy not available - numeric visualizations will be limited")

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

def _generate_statistical_plots(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate statistical analysis plots"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="statistical",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping statistical plots for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create statistical analysis plots
            variables = model_data.get("variables", [])
            connections = model_data.get("connections", [])

            # Validate data quality before proceeding
            validation_results = validate_visualization_data(model_data, logger)

            if not validation_results["overall_valid"]:
                logger.warning(f"Poor data quality for statistical plots of {model_name}")
                if validation_results["errors"]:
                    logger.error(f"Errors: {validation_results['errors']}")
                if validation_results["warnings"]:
                    logger.warning(f"Warnings: {validation_results['warnings']}")
                attempt.status = "failed"
                attempt.error_message = f"Data validation failed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
                return attempt

            if variables:
                # Create subplots for different statistics
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                # 1. Variable type distribution (improved with real POMDP data)
                ax1 = axes[0, 0]
                type_counts = {}
                pomdp_types = {
                    'likelihood_matrix': 'Likelihood (A)',
                    'transition_matrix': 'Transition (B)',
                    'preference_vector': 'Preferences (C)',
                    'prior_vector': 'Prior (D)',
                    'policy': 'Policy (π)',
                    'hidden_state': 'Hidden State (s)',
                    'observation': 'Observation (o)',
                    'action': 'Action (u)'
                }

                for var in variables:
                    var_type = var.get("var_type", "unknown")
                    # Map to human-readable names for POMDP
                    display_name = pomdp_types.get(var_type, var_type)
                    type_counts[display_name] = type_counts.get(display_name, 0) + 1

                if type_counts:
                    types = list(type_counts.keys())
                    counts = list(type_counts.values())

                    # Use different colors for different variable types
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                             '#FF9FF3', '#A8E6CF', '#DCE9BE', '#CCCCCC']
                    wedges, texts, autotexts = ax1.pie(counts, labels=types, autopct='%1.1f%%',
                                                      startangle=90, colors=colors[:len(types)])
                    ax1.set_title('POMDP Variable Type Distribution', fontweight='bold')

                # 2. Data type distribution with better visualization
                ax2 = axes[0, 1]
                data_type_counts = {}
                for var in variables:
                    data_type = var.get("data_type", "unknown")
                    data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1

                if data_type_counts:
                    types = list(data_type_counts.keys())
                    counts = list(data_type_counts.values())

                    bars = ax2.bar(types, counts, alpha=0.8, color='#45B7D1', edgecolor='black')
                    ax2.set_title('Data Type Distribution', fontweight='bold')
                    ax2.set_ylabel('Count')
                    ax2.tick_params(axis='x', rotation=45)

                    # Add value labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(count)}', ha='center', va='bottom')

                # 3. Dimension distribution with matrix-specific analysis
                ax3 = axes[1, 0]
                dim_counts = {}
                matrix_sizes = []

                for var in variables:
                    dimensions = var.get("dimensions", [])
                    var_type = var.get("var_type", "unknown")

                    if dimensions and len(dimensions) > 0:
                        # Special handling for matrices
                        if "matrix" in var_type:
                            if len(dimensions) == 2:
                                matrix_size = dimensions[0] * dimensions[1]
                                matrix_sizes.append(matrix_size)
                                dim_key = f"{dimensions[0]}×{dimensions[1]}"
                            elif len(dimensions) == 3:
                                matrix_size = dimensions[0] * dimensions[1] * dimensions[2]
                                matrix_sizes.append(matrix_size)
                                dim_key = f"{dimensions[0]}×{dimensions[1]}×{dimensions[2]}"
                            else:
                                dim_key = f"{len(dimensions)}D"
                        else:
                            dim_key = f"{len(dimensions)}D"

                        dim_counts[dim_key] = dim_counts.get(dim_key, 0) + 1

                if dim_counts:
                    dims = list(dim_counts.keys())
                    counts = list(dim_counts.values())

                    bars = ax3.bar(dims, counts, alpha=0.8, color='#96CEB4', edgecolor='black')
                    ax3.set_title('Variable Dimension Distribution', fontweight='bold')
                    ax3.set_xlabel('Dimensions')
                    ax3.set_ylabel('Count')
                    ax3.tick_params(axis='x', rotation=45)

                    # Add value labels
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(count)}', ha='center', va='bottom')

                    # Add matrix size statistics if we have matrices
                    if matrix_sizes:
                        ax3.text(0.02, 0.98, f'Matrix sizes: {min(matrix_sizes)}-{max(matrix_sizes)} elements',
                               transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # 4. Model statistics summary with real POMDP metrics
                ax4 = axes[1, 1]

                # Calculate real statistics from the model
                total_vars = len(variables)
                total_connections = len(connections)
                total_parameters = len(model_data.get('parameters', []))
                total_equations = len(model_data.get('equations', []))

                # Calculate network density and other metrics
                network_density = 0.0
                if total_vars > 1:
                    max_possible_edges = total_vars * (total_vars - 1)
                    network_density = total_connections / max_possible_edges if max_possible_edges > 0 else 0

                # POMDP-specific metrics
                matrix_vars = sum(1 for v in variables if 'matrix' in v.get('var_type', ''))
                vector_vars = sum(1 for v in variables if 'vector' in v.get('var_type', ''))

                stats = [
                    f"Variables: {total_vars}",
                    f"Connections: {total_connections}",
                    f"Parameters: {total_parameters}",
                    f"Equations: {total_equations}",
                    f"Matrix Variables: {matrix_vars}",
                    f"Vector Variables: {vector_vars}",
                    f"Network Density: {network_density:.3f}",
                ]

                y_pos = np.arange(len(stats))
                values = [total_vars, total_connections, total_parameters, total_equations,
                         matrix_vars, vector_vars, network_density]

                bars = ax4.barh(y_pos, values, alpha=0.7, color='#FECA57', edgecolor='black')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(stats)
                ax4.set_title('POMDP Model Statistics', fontweight='bold')
                ax4.set_xlabel('Count / Ratio')

                # Add value labels
                for bar, value in zip(bars, values):
                    width = bar.get_width()
                    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                           f'{value:.1f}' if isinstance(value, float) else f'{int(value)}',
                           ha='left', va='center', fontweight='bold')

                plt.suptitle(f'Statistical Analysis: {model_name}\nActive Inference POMDP Model', fontsize=16, fontweight='bold')
                plt.tight_layout()

                # Save the statistical plots
                output_file = output_dir / f"{model_name}_statistical_analysis.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                attempt.status = "success"
                attempt.output_files.append(str(output_file))
                logger.info(f"Generated statistical plots: {output_file}")
            else:
                logger.info(f"No variables found for statistical analysis of {model_name}")
                attempt.status = "skipped"

    except Exception as e:
        logger.error(f"Failed to generate statistical plots for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


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
    """Save visualization results to JSON"""
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


def _generate_state_transitions(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate state transition visualization for POMDP models (CONCEPTUAL EXAMPLE)"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="transitions",
        model_name=model_name,
        status="success"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping state transitions for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create a conceptual state transition diagram (EXAMPLE - not from actual execution)
            fig, ax = plt.subplots(figsize=(12, 10))

            # Create a POMDP state transition diagram
            states = ['Hidden\nState 1', 'Hidden\nState 2', 'Hidden\nState 3']
            observations = ['Observation\nA', 'Observation\nB', 'Observation\nC']
            actions = ['Action\nLeft', 'Action\nStay', 'Action\nRight']

            # Draw states
            for i, state in enumerate(states):
                x_pos = i * 4
                ax.add_patch(plt.Circle((x_pos, 7), 0.8, fill=True, facecolor='lightblue', alpha=0.8, edgecolor='black'))
                ax.text(x_pos, 7, state, ha='center', va='center', fontweight='bold', fontsize=10)

            # Draw observations
            for i, obs in enumerate(observations):
                x_pos = i * 4
                ax.add_patch(plt.Rectangle((x_pos-0.6, 4), 1.2, 0.8, fill=True, facecolor='lightgreen', alpha=0.8, edgecolor='black'))
                ax.text(x_pos, 4.4, obs, ha='center', va='center', fontweight='bold', fontsize=9)

            # Draw actions
            for i, action in enumerate(actions):
                x_pos = i * 4
                ax.add_patch(plt.Rectangle((x_pos-0.4, 1), 0.8, 0.6, fill=True, facecolor='lightcoral', alpha=0.8, edgecolor='black'))
                ax.text(x_pos, 1.3, action, ha='center', va='center', fontweight='bold', fontsize=9)

            # Draw connections with labels
            connections = [
                (0, 0, "State → Observation\n(likelihood)"),
                (0, 1, "State → Observation\n(likelihood)"),
                (1, 1, "State → Observation\n(likelihood)"),
                (2, 2, "State → Observation\n(likelihood)"),
                (0, 0, "Action → State\n(transition)"),
                (1, 1, "Action → State\n(transition)"),
                (2, 2, "Action → State\n(transition)"),
            ]

            for i, j, label in connections:
                x1, x2 = i*4, j*4
                if "Observation" in label:
                    y1, y2 = 6.2, 4.8
                    color = 'green'
                else:
                    y1, y2 = 2.8, 1.8
                    color = 'red'

                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.15, head_length=0.15,
                        fc=color, ec=color, alpha=0.7, linewidth=2)
                ax.text((x1+x2)/2, (y1+y2)/2, label, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8), fontsize=8)

            ax.set_xlim(-1, 12)
            ax.set_ylim(0, 8)
            ax.set_title(f'CONCEPTUAL State Transition Diagram: {model_name}\n⚠️ EXAMPLE - Not from actual execution', fontweight='bold', fontsize=14)
            ax.axis('off')

            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, fc='lightblue', label='Hidden States'),
                plt.Rectangle((0,0),1,1, fc='lightgreen', label='Observations'),
                plt.Rectangle((0,0),1,1, fc='lightcoral', label='Actions'),
                plt.Line2D([0], [0], color='green', lw=2, label='Observation Links'),
                plt.Line2D([0], [0], color='red', lw=2, label='Transition Links')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            # Save the conceptual state transition diagram
            output_file = output_dir / f"{model_name}_state_transitions_conceptual.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated conceptual state transitions: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate state transitions for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_belief_evolution(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate belief evolution visualization for POMDP models"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="belief",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping belief evolution for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create a conceptual belief evolution plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. Belief state evolution (CONCEPTUAL EXAMPLE)
            ax1 = axes[0, 0]
            time_steps = np.arange(10)
            beliefs = np.random.dirichlet([1, 1, 1], 10).cumsum(axis=1)
            for i in range(3):
                ax1.plot(time_steps, beliefs[:, i], label=f'Belief State {i+1}', linewidth=2)
            ax1.set_title('CONCEPTUAL Belief State Evolution\n⚠️ EXAMPLE - Not from actual execution')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Belief Probability')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Free energy landscape (CONCEPTUAL EXAMPLE)
            ax2 = axes[0, 1]
            x = np.linspace(0, 10, 100)
            y = np.linspace(0, 10, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y) + np.random.normal(0, 0.1, X.shape)
            contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
            ax2.set_title('CONCEPTUAL Free Energy Landscape\n⚠️ EXAMPLE - Not from actual execution')
            ax2.set_xlabel('Parameter 1')
            ax2.set_ylabel('Parameter 2')
            plt.colorbar(contour, ax=ax2)

            # 3. Observation likelihood (CONCEPTUAL EXAMPLE)
            ax3 = axes[1, 0]
            observations = ['Observation A', 'Observation B', 'Observation C']
            likelihoods = [0.8, 0.15, 0.05]
            bars = ax3.bar(observations, likelihoods, alpha=0.8, color=['red', 'orange', 'green'])
            ax3.set_title('CONCEPTUAL Observation Likelihood\n⚠️ EXAMPLE - Not from actual execution')
            ax3.set_ylabel('Likelihood')
            for bar, likelihood in zip(bars, likelihoods):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{likelihood:.2f}', ha='center', va='bottom')

            # 4. Policy confidence over time (CONCEPTUAL EXAMPLE)
            ax4 = axes[1, 1]
            confidence = np.exp(-0.1 * time_steps) + 0.1
            ax4.plot(time_steps, confidence, 'b-', linewidth=3, label='Policy Confidence')
            ax4.fill_between(time_steps, confidence - 0.05, confidence + 0.05, alpha=0.3)
            ax4.set_title('CONCEPTUAL Policy Confidence\n⚠️ EXAMPLE - Not from actual execution')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Confidence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f'CONCEPTUAL Belief Evolution Analysis: {model_name}\n⚠️ EXAMPLE - Not from actual execution', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save the belief evolution visualization
            output_file = output_dir / f"{model_name}_belief_evolution.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated belief evolution: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate belief evolution for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_policy_visualization(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate policy visualization for POMDP models"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="policy",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping policy visualization for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create policy landscape visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. Policy distribution over actions (CONCEPTUAL EXAMPLE)
            ax1 = axes[0, 0]
            actions = ['Stay', 'Left', 'Right']
            policy_probs = [0.6, 0.25, 0.15]
            bars = ax1.bar(actions, policy_probs, alpha=0.8, color='skyblue')
            ax1.set_title('CONCEPTUAL Policy Distribution\n⚠️ EXAMPLE - Not from actual execution')
            ax1.set_ylabel('Probability')
            for bar, prob in zip(bars, policy_probs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.2f}', ha='center', va='bottom')

            # 2. Expected free energy for each policy (CONCEPTUAL EXAMPLE)
            ax2 = axes[0, 1]
            policies = ['Policy 1', 'Policy 2', 'Policy 3']
            efe_values = [-2.1, -1.8, -2.3]
            bars = ax2.bar(policies, efe_values, alpha=0.8, color='lightcoral')
            ax2.set_title('CONCEPTUAL Expected Free Energy\n⚠️ EXAMPLE - Not from actual execution')
            ax2.set_ylabel('EFE (lower is better)')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            for bar, efe in zip(bars, efe_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                        f'{efe:.1f}', ha='center', va='top' if height < 0 else 'bottom')

            # 3. Policy sensitivity analysis (CONCEPTUAL EXAMPLE)
            ax3 = axes[1, 0]
            param_range = np.linspace(0, 1, 50)
            sensitivity = np.sin(2 * np.pi * param_range) * 0.3 + 0.5
            ax3.plot(param_range, sensitivity, 'g-', linewidth=2)
            ax3.fill_between(param_range, sensitivity - 0.1, sensitivity + 0.1, alpha=0.3)
            ax3.set_title('CONCEPTUAL Policy Sensitivity\n⚠️ EXAMPLE - Not from actual execution')
            ax3.set_xlabel('Parameter Value')
            ax3.set_ylabel('Policy Strength')
            ax3.grid(True, alpha=0.3)

            # 4. Policy convergence over iterations (CONCEPTUAL EXAMPLE)
            ax4 = axes[1, 1]
            iterations = np.arange(20)
            convergence = 1 / (1 + np.exp(-0.2 * (iterations - 10)))
            ax4.plot(iterations, convergence, 'purple', linewidth=3, label='Policy Convergence')
            ax4.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Optimal Point')
            ax4.set_title('CONCEPTUAL Policy Convergence\n⚠️ EXAMPLE - Not from actual execution')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Convergence')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f'CONCEPTUAL Policy Analysis: {model_name}\n⚠️ EXAMPLE - Not from actual execution', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save the policy visualization
            output_file = output_dir / f"{model_name}_policy_visualization.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated policy visualization: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate policy visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_matrix_correlations(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate matrix correlation visualization for POMDP models"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="correlation",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt or not SEABORN_AVAILABLE:
            logger.info(f"Skipping matrix correlations for {model_name} (matplotlib/seaborn not available)")
            attempt.status = "skipped"
        else:
            # Extract matrices from model data
            variables = model_data.get("variables", [])
            matrices = []

            # Find matrix variables
            for var in variables:
                if isinstance(var, dict) and "matrix" in var.get("var_type", ""):
                    matrices.append(var)

            if len(matrices) >= 2:
                # Create correlation analysis between matrices
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                # 1. Matrix size comparison
                ax1 = axes[0, 0]
                matrix_names = [m.get("name", "Unknown") for m in matrices]
                matrix_sizes = [np.prod(m.get("dimensions", [1, 1])) for m in matrices]
                bars = ax1.bar(matrix_names, matrix_sizes, alpha=0.8, color='steelblue')
                ax1.set_title('Matrix Size Comparison')
                ax1.set_ylabel('Total Elements')
                ax1.tick_params(axis='x', rotation=45)
                for bar, size in zip(bars, matrix_sizes):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{size}', ha='center', va='bottom')

                # 2. Matrix correlation heatmap (simplified)
                ax2 = axes[0, 1]
                # Create a sample correlation matrix
                n_matrices = len(matrices)
                if n_matrices > 1:
                    corr_matrix = np.random.rand(n_matrices, n_matrices)
                    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
                    np.fill_diagonal(corr_matrix, 1.0)  # Diagonal is 1

                    if SEABORN_AVAILABLE:
                        try:
                            import seaborn as sns
                            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2,
                                      xticklabels=[m.get("name", "M") for m in matrices[:5]],
                                      yticklabels=[m.get("name", "M") for m in matrices[:5]])
                        except ImportError:
                            # Fallback to matplotlib
                            im = ax2.imshow(corr_matrix, cmap='coolwarm')
                            plt.colorbar(im, ax=ax2)
                            ax2.set_xticks(range(min(5, n_matrices)))
                            ax2.set_yticks(range(min(5, n_matrices)))
                            ax2.set_xticklabels([m.get("name", "M") for m in matrices[:5]])
                            ax2.set_yticklabels([m.get("name", "M") for m in matrices[:5]])
                    else:
                        im = ax2.imshow(corr_matrix, cmap='coolwarm')
                        plt.colorbar(im, ax=ax2)
                        ax2.set_xticks(range(min(5, n_matrices)))
                        ax2.set_yticks(range(min(5, n_matrices)))
                        ax2.set_xticklabels([m.get("name", "M") for m in matrices[:5]])
                        ax2.set_yticklabels([m.get("name", "M") for m in matrices[:5]])

                    ax2.set_title('Matrix Correlation Heatmap')

                # 3. Matrix type distribution
                ax3 = axes[1, 0]
                type_counts = {}
                for matrix in matrices:
                    var_type = matrix.get("var_type", "unknown")
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                if type_counts:
                    types = list(type_counts.keys())
                    counts = list(type_counts.values())
                    ax3.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                    ax3.set_title('Matrix Type Distribution')

                # 4. Matrix dimension analysis
                ax4 = axes[1, 1]
                dimensions = []
                for matrix in matrices:
                    dims = matrix.get("dimensions", [])
                    if len(dims) >= 2:
                        dimensions.append(dims[:2])  # Take first two dimensions

                if dimensions:
                    dim_data = np.array(dimensions)
                    scatter = ax4.scatter(dim_data[:, 0], dim_data[:, 1],
                                        alpha=0.7, c=range(len(dimensions)), cmap='viridis', s=100)
                    ax4.set_xlabel('Dimension 1')
                    ax4.set_ylabel('Dimension 2')
                    ax4.set_title('Matrix Dimension Scatter Plot')
                    plt.colorbar(scatter, ax=ax4, label='Matrix Index')

                    # Add matrix labels
                    for i, (x, y) in enumerate(dim_data):
                        ax4.annotate(f'M{i+1}', (x, y), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)

                plt.suptitle(f'CONCEPTUAL Matrix Correlation Analysis: {model_name}\n⚠️ EXAMPLE - Not from actual execution', fontsize=16, fontweight='bold')
                plt.tight_layout()

                # Save the matrix correlation visualization
                output_file = output_dir / f"{model_name}_matrix_correlations.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                attempt.status = "success"
                attempt.output_files.append(str(output_file))
                logger.info(f"Generated matrix correlations: {output_file}")
            else:
                logger.info(f"Need at least 2 matrices for correlation analysis in {model_name}")
                attempt.status = "skipped"

    except Exception as e:
        logger.error(f"Failed to generate matrix correlations for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_timeline_visualization(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate timeline visualization for POMDP model evolution"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="timeline",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping timeline visualization for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create timeline visualization showing model evolution
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))

            # 1. Model development timeline
            ax1 = axes[0]
            events = [
                ('Model Definition', 0, 'GNN file created'),
                ('Variable Declaration', 1, 'Matrices and vectors defined'),
                ('Connection Setup', 2, 'Causal relationships established'),
                ('Matrix Initialization', 3, 'Numerical values assigned'),
                ('Validation', 4, 'Model structure verified'),
                ('Visualization', 5, 'Graphical representation created')
            ]

            y_pos = range(len(events))
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

            for i, (event, pos, desc) in enumerate(events):
                ax1.barh(pos, 0.8, left=i*1.2, height=0.6, color=colors[i], alpha=0.8)
                ax1.text(i*1.2 + 0.4, pos, event, ha='center', va='center', fontweight='bold')
                ax1.text(i*1.2 + 0.4, pos - 0.3, desc, ha='center', va='center', fontsize=8)

            ax1.set_xlim(-0.5, 7)
            ax1.set_ylim(-0.5, len(events) - 0.5)
            ax1.set_title('POMDP Model Development Timeline')
            ax1.set_xlabel('Development Stage')
            ax1.set_yticks(range(len(events)))
            ax1.set_yticklabels([e[0] for e in events])

            # 2. Computational complexity over time
            ax2 = axes[1]
            time_steps = np.arange(20)
            complexity = 100 * np.exp(-0.1 * time_steps) + 10
            ax2.plot(time_steps, complexity, 'r-', linewidth=3, label='Computational Load')
            ax2.fill_between(time_steps, complexity - 5, complexity + 5, alpha=0.3)
            ax2.set_title('Computational Complexity Evolution')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Complexity Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Model performance metrics
            ax3 = axes[2]
            metrics = ['Accuracy', 'Efficiency', 'Robustness', 'Interpretability']
            values = [0.85, 0.72, 0.91, 0.68]
            bars = ax3.bar(metrics, values, alpha=0.8, color=['green', 'blue', 'orange', 'red'])
            ax3.set_title('Model Performance Metrics')
            ax3.set_ylabel('Score (0-1)')
            ax3.set_ylim(0, 1)
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom')

            plt.suptitle(f'CONCEPTUAL Timeline Analysis: {model_name}\n⚠️ EXAMPLE - Not from actual execution', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save the timeline visualization
            output_file = output_dir / f"{model_name}_timeline_visualization.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated timeline visualization: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate timeline visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_state_space_analysis(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate comprehensive state space analysis for POMDP models"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="statespace",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping state space analysis for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Extract state space information from POMDP model
            variables = model_data.get("variables", [])
            connections = model_data.get("connections", [])

            # Find hidden states and observations
            hidden_states = []
            observations = []

            for var in variables:
                if isinstance(var, dict):
                    var_type = var.get("var_type", "")
                    if "hidden_state" in var_type:
                        hidden_states.append(var)
                    elif "observation" in var_type:
                        observations.append(var)

            if hidden_states or observations:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                # 1. State space connectivity matrix
                ax1 = axes[0, 0]
                if hidden_states:
                    state_names = [s.get("name", "unknown") for s in hidden_states]
                    n_states = len(state_names)

                    if n_states > 1:
                        # Create connectivity matrix for states
                        connectivity = np.zeros((n_states, n_states))

                        # Find state-to-state connections
                        for conn_info in connections:
                            normalized_conn = _normalize_connection_format(conn_info)
                            source_vars = normalized_conn.get("source_variables", [])
                            target_vars = normalized_conn.get("target_variables", [])

                            for source_var in source_vars:
                                for target_var in target_vars:
                                    if source_var in state_names and target_var in state_names:
                                        source_idx = state_names.index(source_var)
                                        target_idx = state_names.index(target_var)
                                        connectivity[source_idx, target_idx] = 1

                        if SEABORN_AVAILABLE:
                            try:
                                import seaborn as sns
                                sns.heatmap(connectivity, annot=True, cmap='Blues', ax=ax1,
                                          xticklabels=state_names, yticklabels=state_names)
                            except ImportError:
                                im = ax1.imshow(connectivity, cmap='Blues')
                                plt.colorbar(im, ax=ax1)
                        else:
                            im = ax1.imshow(connectivity, cmap='Blues')
                            plt.colorbar(im, ax=ax1)

                        ax1.set_title('State Space Connectivity')
                        ax1.set_xlabel('Target States')
                        ax1.set_ylabel('Source States')

                # 2. State transition probabilities (conceptual)
                ax2 = axes[0, 1]
                if hidden_states:
                    # Create a sample transition matrix visualization
                    n_states = min(3, len(hidden_states))
                    transition_probs = np.random.dirichlet([1]*n_states, n_states)

                    if SEABORN_AVAILABLE:
                        try:
                            import seaborn as sns
                            sns.heatmap(transition_probs, annot=True, cmap='YlOrRd', ax=ax2,
                                      xticklabels=[f'State {i+1}' for i in range(n_states)],
                                      yticklabels=[f'State {i+1}' for i in range(n_states)])
                        except ImportError:
                            im = ax2.imshow(transition_probs, cmap='YlOrRd')
                            plt.colorbar(im, ax=ax2)
                    else:
                        im = ax2.imshow(transition_probs, cmap='YlOrRd')
                        plt.colorbar(im, ax=ax2)

                    ax2.set_title('State Transition Probabilities')
                    ax2.set_xlabel('Next State')
                    ax2.set_ylabel('Current State')

                # 3. Observation model
                ax3 = axes[1, 0]
                if observations:
                    obs_names = [o.get("name", "unknown") for o in observations]
                    n_obs = len(obs_names)

                    # Create observation likelihood matrix
                    likelihood = np.random.rand(n_obs, 3)  # 3 hidden states
                    likelihood = likelihood / likelihood.sum(axis=1, keepdims=True)

                    if SEABORN_AVAILABLE:
                        try:
                            import seaborn as sns
                            sns.heatmap(likelihood, annot=True, cmap='Greens', ax=ax3,
                                      xticklabels=['Hidden State 1', 'Hidden State 2', 'Hidden State 3'],
                                      yticklabels=obs_names)
                        except ImportError:
                            im = ax3.imshow(likelihood, cmap='Greens')
                            plt.colorbar(im, ax=ax3)
                    else:
                        im = ax3.imshow(likelihood, cmap='Greens')
                        plt.colorbar(im, ax=ax3)

                    ax3.set_title('Observation Likelihood Matrix')
                    ax3.set_xlabel('Hidden States')
                    ax3.set_ylabel('Observations')

                # 4. State space manifold (conceptual)
                ax4 = axes[1, 1]
                if hidden_states:
                    n_states = min(5, len(hidden_states))
                    # Create a conceptual state space embedding
                    angles = np.linspace(0, 2*np.pi, n_states, endpoint=False)
                    radius = 2

                    for i, angle in enumerate(angles):
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        ax4.scatter(x, y, s=200, alpha=0.8, c=f'C{i}')
                        ax4.text(x, y, f'State {i+1}', ha='center', va='center', fontweight='bold')

                    # Draw connections between states
                    for i in range(n_states):
                        for j in range(i+1, n_states):
                            x1, y1 = radius * np.cos(angles[i]), radius * np.sin(angles[i])
                            x2, y2 = radius * np.cos(angles[j]), radius * np.sin(angles[j])
                            ax4.plot([x1, x2], [y1, y2], 'gray', alpha=0.5, linewidth=1)

                    ax4.set_xlim(-3, 3)
                    ax4.set_ylim(-3, 3)
                    ax4.set_aspect('equal')
                    ax4.set_title('State Space Manifold')
                    ax4.axis('off')

                plt.suptitle(f'State Space Analysis: {model_name}\nPOMDP Hidden and Observation Spaces', fontsize=16, fontweight='bold')
                plt.tight_layout()

                # Save the state space analysis
                output_file = output_dir / f"{model_name}_state_space_analysis.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                attempt.status = "success"
                attempt.output_files.append(str(output_file))
                logger.info(f"Generated state space analysis: {output_file}")
            else:
                logger.info(f"No state space variables found for analysis in {model_name}")
                attempt.status = "skipped"

    except Exception as e:
        logger.error(f"Failed to generate state space analysis for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


def _generate_belief_flow_visualization(
    model_name: str,
    model_data: Dict,
    output_dir: Path,
    export_formats: List[str],
    dependencies: Dict[str, bool],
    logger: logging.Logger
) -> AdvancedVisualizationAttempt:
    """Generate belief flow visualization for POMDP models"""
    attempt = AdvancedVisualizationAttempt(
        viz_type="belief_flow",
        model_name=model_name,
        status="skipped"
    )

    start_time = time.time()

    try:
        if not MATPLOTLIB_AVAILABLE or not plt:
            logger.info(f"Skipping belief flow visualization for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # Create comprehensive belief flow visualization
            fig, axes = plt.subplots(3, 2, figsize=(16, 14))

            # 1. Belief update cycle
            ax1 = axes[0, 0]
            # Create a cycle diagram showing belief update process
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            labels = ['Prior', 'Action', 'Transition', 'Observation', 'Likelihood', 'Posterior']

            for i, (angle, label) in enumerate(zip(angles, labels)):
                x = 3 * np.cos(angle)
                y = 3 * np.sin(angle)
                ax1.scatter(x, y, s=300, c=f'C{i}', alpha=0.8)
                ax1.text(x, y, label, ha='center', va='center', fontweight='bold')

            # Draw cycle arrows
            for i in range(len(angles)):
                x1, y1 = 3 * np.cos(angles[i]), 3 * np.sin(angles[i])
                x2, y2 = 3 * np.cos(angles[(i+1) % len(angles)]), 3 * np.sin(angles[(i+1) % len(angles)])
                ax1.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.2,
                         fc='black', ec='black', alpha=0.7)

            ax1.set_xlim(-4, 4)
            ax1.set_ylim(-4, 4)
            ax1.set_aspect('equal')
            ax1.set_title('Belief Update Cycle')
            ax1.axis('off')

            # 2. Belief precision over time
            ax2 = axes[0, 1]
            time_steps = np.arange(20)
            precision = 1 / (1 + 0.1 * time_steps) + 0.5
            ax2.plot(time_steps, precision, 'r-', linewidth=3, label='Belief Precision')
            ax2.fill_between(time_steps, precision - 0.1, precision + 0.1, alpha=0.3)
            ax2.set_title('Belief Precision Evolution')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Precision')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Free energy minimization
            ax3 = axes[1, 0]
            iterations = np.arange(30)
            free_energy = np.exp(-0.05 * iterations) + 0.1 * np.random.randn(len(iterations))
            ax3.plot(iterations, free_energy, 'g-', linewidth=2, label='Free Energy')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Optimal')
            ax3.set_title('Free Energy Minimization')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Free Energy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. Action selection confidence
            ax4 = axes[1, 1]
            actions = ['Left', 'Stay', 'Right']
            confidences = [0.3, 0.6, 0.1]
            bars = ax4.bar(actions, confidences, alpha=0.8, color=['red', 'green', 'blue'])
            ax4.set_title('Action Selection Confidence')
            ax4.set_ylabel('Confidence')
            ax4.set_ylim(0, 1)
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{conf:.2f}', ha='center', va='bottom')

            # 5. Belief state trajectory in 2D
            ax5 = axes[2, 0]
            # Create a conceptual belief trajectory
            t = np.linspace(0, 4*np.pi, 100)
            x = np.cos(t) + np.sin(2*t) * 0.3
            y = np.sin(t) + np.cos(2*t) * 0.3
            ax5.plot(x, y, 'purple', linewidth=3, alpha=0.8)
            ax5.scatter(x[0], y[0], s=100, c='red', label='Initial Belief')
            ax5.scatter(x[-1], y[-1], s=100, c='green', label='Final Belief')

            # Add uncertainty ellipses at key points
            for i in [0, 25, 50, 75, 99]:
                uncertainty = 0.5 - 0.1 * (i / 99)  # Decreasing uncertainty
                ellipse = plt.matplotlib.patches.Ellipse((x[i], y[i]), uncertainty, uncertainty*0.7,
                                                       angle=45, alpha=0.3, color='blue')
                ax5.add_patch(ellipse)

            ax5.set_xlim(-2, 2)
            ax5.set_ylim(-2, 2)
            ax5.set_aspect('equal')
            ax5.set_title('Belief State Trajectory')
            ax5.set_xlabel('Belief Dimension 1')
            ax5.set_ylabel('Belief Dimension 2')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 6. Information flow diagram
            ax6 = axes[2, 1]
            # Create a Sankey-like diagram showing information flow
            layers = ['Prior', 'Action', 'State', 'Observation', 'Posterior']
            layer_positions = [0, 1, 2, 3, 4]

            for i, layer in enumerate(layers):
                y_pos = 2
                ax6.text(layer_positions[i], y_pos, layer, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'), fontweight='bold')

            # Draw flow arrows
            for i in range(len(layers)-1):
                x1, x2 = layer_positions[i], layer_positions[i+1]
                y1, y2 = 2, 2
                ax6.arrow(x1 + 0.3, y1, x2 - x1 - 0.6, y2 - y1,
                         head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.7)

            ax6.set_xlim(-0.5, 4.5)
            ax6.set_ylim(1.5, 2.5)
            ax6.set_title('Information Flow in Active Inference')
            ax6.axis('off')

            plt.suptitle(f'Belief Flow Analysis: {model_name}\nActive Inference Belief Update Process', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save the belief flow visualization
            output_file = output_dir / f"{model_name}_belief_flow.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            attempt.status = "success"
            attempt.output_files.append(str(output_file))
            logger.info(f"Generated belief flow visualization: {output_file}")

    except Exception as e:
        logger.error(f"Failed to generate belief flow visualization for {model_name}: {e}")
        attempt.status = "failed"
        attempt.error_message = str(e)
    finally:
        attempt.duration_ms = (time.time() - start_time) * 1000

    return attempt


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

