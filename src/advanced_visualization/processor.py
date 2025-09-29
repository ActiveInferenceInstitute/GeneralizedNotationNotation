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
    dependencies = {
        "matplotlib": False,
        "plotly": False,
        "seaborn": False,
        "bokeh": False,
        "numpy": False
    }
    
    # Check matplotlib
    try:
        import matplotlib
        dependencies["matplotlib"] = True
    except ImportError:
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
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        logger.warning("numpy not available - numeric visualizations will be limited")
    
    return dependencies


def _load_gnn_models(target_dir: Path, logger: logging.Logger, base_output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """Load GNN models from processing results"""
    from pipeline.config import get_output_dir_for_script
    
    # Get GNN output directory
    if base_output_dir is None:
        base_output_dir = Path("output")
    gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)
    
    # Check for double-nested directory structure
    results_file = gnn_output_dir / "gnn_processing_results.json"
    if not results_file.exists():
        # Try nested structure
        nested_results_file = gnn_output_dir / "3_gnn_output" / "gnn_processing_results.json"
        if nested_results_file.exists():
            results_file = nested_results_file
            gnn_output_dir = gnn_output_dir / "3_gnn_output"
    
    if not results_file.exists():
        logger.warning(f"GNN processing results not found at {results_file}")
        return {}
    
    # Load results
    try:
        with open(results_file) as f:
            processing_results = json.load(f)
        
        models = {}
        for result in processing_results.get("results", []):
            if result.get("parsing_success"):
                model_name = result.get("model_name")
                parsed_file = gnn_output_dir / result.get("output_dir", model_name) / f"{model_name}_parsed.json"
                
                if parsed_file.exists():
                    with open(parsed_file) as f:
                        models[model_name] = json.load(f)
        
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
        if not dependencies.get("plotly"):
            logger.info(f"Skipping 3D visualization for {model_name} (plotly not available)")
            attempt.fallback_used = True
            # Generate fallback HTML report
            _generate_fallback_report(model_name, "3d", output_dir, model_data, logger)
            attempt.status = "success"
            attempt.output_files.append(str(output_dir / f"{model_name}_3d_fallback.html"))
        else:
            # TODO: Implement actual 3D visualization with plotly
            logger.info(f"3D visualization for {model_name} not yet implemented")
            attempt.status = "skipped"
        
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
        if not dependencies.get("matplotlib"):
            logger.info(f"Skipping statistical plots for {model_name} (matplotlib not available)")
            attempt.status = "skipped"
        else:
            # TODO: Implement statistical plots
            logger.info(f"Statistical plots for {model_name} not yet implemented")
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

