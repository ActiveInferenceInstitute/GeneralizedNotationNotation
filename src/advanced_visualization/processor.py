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
                    results.total_attempts += 1
                    if attempt.status == "success":
                        results.successful += 1
                        results.output_files.extend(attempt.output_files)
                    elif attempt.status == "failed":
                        results.failed += 1
                        results.errors.append(attempt.error_message or "Unknown error")
                    else:
                        results.skipped += 1
        
        # Save results
        _save_results(output_dir, results, logger)
        
        # Log summary
        logger.info(f"Advanced visualization complete:")
        logger.info(f"  Total attempts: {results.total_attempts}")
        logger.info(f"  Successful: {results.successful}")
        logger.info(f"  Failed: {results.failed}")
        logger.info(f"  Skipped: {results.skipped}")
        logger.info(f"  Output files: {len(results.output_files)}")
        
        # Return success if at least some visualizations succeeded
        return results.successful > 0 or results.total_attempts == 0
        
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
    try:
        import seaborn
        dependencies["seaborn"] = True
    except ImportError:
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

            if variables:
                # Create 3D positions for variables based on their properties
                n_vars = len(variables)
                positions = np.random.rand(n_vars, 3) * 10  # Random 3D positions for demo

                # Color code by variable type
                colors = []
                sizes = []
                for var in variables:
                    var_type = var.get("var_type", "unknown")
                    if "matrix" in var_type:
                        colors.append('red')
                        sizes.append(100)
                    elif "vector" in var_type:
                        colors.append('blue')
                        sizes.append(80)
                    else:
                        colors.append('green')
                        sizes.append(60)

                # Plot variables as 3D points
                for i, (pos, color, size) in enumerate(zip(positions, colors, sizes)):
                    ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.7)
                    ax.text(pos[0], pos[1], pos[2], variables[i].get("name", f"Var{i}"),
                           fontsize=8, ha='center', va='center')

                # Add some connections between variables (simplified)
                for i in range(min(5, n_vars)):
                    for j in range(i+1, min(i+3, n_vars)):
                        ax.plot([positions[i][0], positions[j][0]],
                               [positions[i][1], positions[j][1]],
                               [positions[i][2], positions[j][2]], 'gray', alpha=0.3)

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

            if variables:
                # Create subplots for different statistics
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # 1. Variable type distribution
                ax1 = axes[0, 0]
                type_counts = {}
                for var in variables:
                    var_type = var.get("var_type", "unknown")
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                if type_counts:
                    types = list(type_counts.keys())
                    counts = list(type_counts.values())
                    ax1.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Variable Type Distribution')

                # 2. Data type distribution
                ax2 = axes[0, 1]
                data_type_counts = {}
                for var in variables:
                    data_type = var.get("data_type", "unknown")
                    data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1

                if data_type_counts:
                    types = list(data_type_counts.keys())
                    counts = list(data_type_counts.values())
                    ax2.bar(types, counts, alpha=0.7)
                    ax2.set_title('Data Type Distribution')
                    ax2.tick_params(axis='x', rotation=45)

                # 3. Dimension distribution
                ax3 = axes[1, 0]
                dim_counts = {}
                for var in variables:
                    dimensions = var.get("dimensions", [])
                    if dimensions:
                        dim_key = f"{len(dimensions)}D"
                        dim_counts[dim_key] = dim_counts.get(dim_key, 0) + 1

                if dim_counts:
                    dims = list(dim_counts.keys())
                    counts = list(dim_counts.values())
                    ax3.bar(dims, counts, alpha=0.7, color='green')
                    ax3.set_title('Dimension Distribution')
                    ax3.set_xlabel('Dimensions')
                    ax3.set_ylabel('Count')

                # 4. Model statistics summary
                ax4 = axes[1, 1]
                stats = [
                    f"Variables: {len(variables)}",
                    f"Connections: {len(connections)}",
                    f"Parameters: {len(model_data.get('parameters', []))}",
                    f"Equations: {len(model_data.get('equations', []))}"
                ]

                y_pos = np.arange(len(stats))
                ax4.barh(y_pos, [len(variables), len(connections),
                                len(model_data.get('parameters', [])),
                                len(model_data.get('equations', []))],
                         alpha=0.7, color='orange')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(stats)
                ax4.set_title('Model Statistics')
                ax4.set_xlabel('Count')

                plt.suptitle(f'Statistical Analysis: {model_name}', fontsize=16)
                plt.tight_layout()

                # Save the statistical plots
                output_file = output_dir / f"{model_name}_statistical_analysis.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
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

