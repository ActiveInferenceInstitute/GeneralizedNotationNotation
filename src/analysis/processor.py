#!/usr/bin/env python3
"""
Analysis processor module for GNN analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.logging.logging_utils import log_step_error, log_step_start, log_step_success

from .analyzer import (
    calculate_complexity_metrics,
    generate_analysis_summary,
    generate_matrix_visualizations,
    perform_model_comparisons,
    perform_statistical_analysis,
    run_performance_benchmarks,
    visualize_simulation_results,
)


def aggregate_simulation_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple simulations.
    
    Args:
        results_list: List of simulation result dictionaries
        
    Returns:
        Aggregated results dictionary
    """
    aggregated = {
        "count": len(results_list),
        "metrics": {},
        "frameworks_used": set()
    }

    # Extract common metrics
    metrics_to_gather = ["execution_time", "free_energy_final", "steps_completed"]

    for res in results_list:
        if "framework" in res:
            aggregated["frameworks_used"].add(res["framework"])

        for metric in metrics_to_gather:
            if metric not in aggregated["metrics"]:
                aggregated["metrics"][metric] = []

            if metric in res:
                aggregated["metrics"][metric].append(res[metric])
            elif "metrics" in res and metric in res["metrics"]:
                aggregated["metrics"][metric].append(res["metrics"][metric])

    # Convert sets to lists
    aggregated["frameworks_used"] = list(aggregated["frameworks_used"])

    return aggregated


def generate_summary_statistics(aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from aggregated data.
    
    Args:
        aggregated_data: Aggregated data dictionary
        
    Returns:
        Dictionary with statistical summaries (mean, std, min, max)
    """
    stats = {}

    for metric, values in aggregated_data.get("metrics", {}).items():
        if not values:
            continue

        # Filter out None values
        valid_values = [v for v in values if v is not None and isinstance(v, (int, float))]

        if not valid_values:
            continue

        stats[metric] = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "count": len(valid_values)
        }

    return stats

def process_analysis(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with comprehensive analysis.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("analysis")

    try:
        log_step_start(logger, "Processing analysis")
        logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "statistical_analysis": [],
            "complexity_metrics": [],
            "performance_benchmarks": [],
            "model_comparisons": [],
            "visualization_files": []
        }

        # Resolve execution results directory once at the top so all branches can reference it
        try:
            from pipeline.config import get_output_dir_for_script
            execution_dir = get_output_dir_for_script("12_execute.py", output_dir.parent)
        except ImportError:
            execution_dir = output_dir.parent / "12_execute_output"

        # Find GNN files
        gnn_files = list(target_dir.rglob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for analysis")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform statistical analysis
                    stats_analysis = perform_statistical_analysis(gnn_file, verbose)
                    results["statistical_analysis"].append(stats_analysis)

                    # Calculate complexity metrics
                    complexity = calculate_complexity_metrics(gnn_file, verbose)
                    results["complexity_metrics"].append(complexity)

                    # Run performance benchmarks
                    benchmarks = run_performance_benchmarks(gnn_file, verbose)
                    results["performance_benchmarks"].append(benchmarks)

                    # Generate matrix visualizations (moved from Step 8)
                    matrix_viz = generate_matrix_visualizations({"matrices": stats_analysis.get("matrices", [])}, results_dir, gnn_file.stem)
                    results["visualization_files"].extend(matrix_viz)

                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")

            # 2. Post-Simulation Analysis: Load execution results if available
            # (execution_dir resolved above, before the gnn_files check)
            # Log the exact path being searched for debugging
            logger.info(f"Looking for execution results in: {execution_dir}")

            if execution_dir.exists():
                logger.info("Found execution results directory, performing post-simulation analysis")

                # First, check for execution summary file (generated by 12_execute step)
                # New location: summaries/ subfolder; recovery: root for backward compatibility
                execution_summary_file = execution_dir / "summaries" / "execution_summary.json"
                if not execution_summary_file.exists():
                    execution_summary_file = execution_dir / "execution_summary.json"
                logger.info(f"Looking for execution summary at: {execution_summary_file}")

                if execution_summary_file.exists():
                    logger.info(f"Found execution summary: {execution_summary_file}")
                    try:
                        with open(execution_summary_file, 'r') as f:
                            execution_results_data = json.load(f)

                        # Generate empirical visualizations from execution summary
                        empirical_viz = visualize_simulation_results(execution_results_data, results_dir)
                        results["visualization_files"].extend(empirical_viz)
                        logger.info(f"Generated {len(empirical_viz)} empirical visualizations from execution summary")
                    except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
                        logger.warning(f"Failed to load execution summary: {e}")
                else:
                    logger.warning(f"Execution summary not found at {execution_summary_file}")

                # Perform detailed post-simulation analysis for each model
                try:
                    from .post_simulation import analyze_execution_results

                    for gnn_file in gnn_files:
                        model_name = gnn_file.stem
                        logger.info(f"Analyzing execution results for {model_name}")

                        # Find execution results for this model
                        model_execution_dir = execution_dir / model_name
                        if model_execution_dir.exists():
                            post_sim_analysis = analyze_execution_results(
                                model_execution_dir,
                                model_name=model_name
                            )

                            analysis_cross_fw_dir = results_dir / "cross_framework"
                            analysis_cross_fw_dir.mkdir(parents=True, exist_ok=True)
                            analysis_file = analysis_cross_fw_dir / f"{model_name}_post_simulation_analysis.json"
                            with open(analysis_file, 'w') as f:
                                json.dump(post_sim_analysis, f, indent=2, default=convert_numpy_types)

                            results["post_simulation_analysis"] = results.get("post_simulation_analysis", [])
                            results["post_simulation_analysis"].append({
                                "model_name": model_name,
                                "analysis_file": str(analysis_file),
                                "framework_count": len(post_sim_analysis.get("framework_results", {})),
                                "has_comparison": "cross_framework_comparison" in post_sim_analysis
                            })

                            logger.info(f"Post-simulation analysis completed for {model_name}")
                        else:
                            logger.debug(f"No model-specific execution directory for {model_name}")

                    # Aggregate results and generate statistics
                    if results.get("post_simulation_analysis"):
                        aggregated = aggregate_simulation_results(results["post_simulation_analysis"])
                        stats = generate_summary_statistics(aggregated)
                        results["overall_statistics"] = stats
                        logger.info("Generated overall summary statistics")
                except Exception as e:
                    logger.error(f"Failed to process post-simulation analysis: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

                # 2.5-2.11: Generate per-framework visualizations from execution logs
                _FRAMEWORK_ANALYZERS = [
                    ("pymdp",              "pymdp",              "PyMDP"),
                    ("activeinference_jl", "activeinference_jl", "ActiveInference.jl"),
                    ("discopy",            "discopy",            "DisCoPy"),
                    ("jax",                "jax",                "JAX"),
                    ("rxinfer",            "rxinfer",            "RxInfer"),
                    ("pytorch",            "pytorch",            "PyTorch"),
                    ("numpyro",            "numpyro",            "NumPyro"),
                    ("bnlearn",            "bnlearn",            "bnlearn"),
                ]
                import importlib
                for module_key, dir_name, display_name in _FRAMEWORK_ANALYZERS:
                    try:
                        mod = importlib.import_module(f".{module_key}.analyzer", package="analysis")
                        fw_output_dir = output_dir / dir_name
                        fw_output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Generating {display_name} visualizations...")
                        fw_viz = mod.generate_analysis_from_logs(execution_dir, fw_output_dir, verbose)
                        if fw_viz:
                            results["visualization_files"].extend(fw_viz)
                            logger.info(f"Generated {len(fw_viz)} {display_name} visualization files")
                    except ImportError as e:
                        logger.debug(f"{module_key} analyzer not available: {e}")
                    except Exception as e:  # Optional analysis — broad catch intentional for third-party failures
                        logger.warning(f"{display_name} analysis failed: {e}")
            else:
                logger.warning(f"Execution directory not found at {execution_dir}. Skipping post-simulation analysis.")

            # Perform cross-model comparisons if multiple files
            if len(gnn_files) > 1:
                comparisons = perform_model_comparisons(results["statistical_analysis"], verbose)
                results["model_comparisons"].append(comparisons)

        # Perform cross-framework analysis if execution results exist
        if execution_dir.exists():
            logger.info("Performing cross-framework analysis...")
            try:
                from .analyzer import (
                    analyze_framework_outputs,
                    generate_framework_comparison_report,
                    visualize_cross_framework_metrics,
                )

                framework_comparison = analyze_framework_outputs(execution_dir, logger)
                results["framework_comparison"] = framework_comparison

                # Generate comparison report
                report_file = generate_framework_comparison_report(framework_comparison, results_dir, logger)
                results["framework_comparison_report"] = report_file

                # Generate comparison visualizations
                comparison_viz = visualize_cross_framework_metrics(framework_comparison, results_dir, logger)
                results["visualization_files"].extend(comparison_viz)

                logger.info(f"Generated {len(comparison_viz)} cross-framework comparison visualizations")
            except Exception as e:
                logger.warning(f"Cross-framework analysis failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            # 3. Comprehensive visualization of all execution outputs
            logger.info("Generating comprehensive visualizations for all execution outputs...")
            try:
                from .post_simulation import (
                    generate_unified_framework_dashboard,
                    visualize_all_framework_outputs,
                )

                # Use cross_framework folder for cross-implementation analysis
                viz_output_dir = results_dir / "cross_framework"
                comprehensive_viz = visualize_all_framework_outputs(execution_dir, viz_output_dir, logger)
                results["comprehensive_visualizations"] = comprehensive_viz
                results["visualization_files"].extend(comprehensive_viz)

                logger.info(f"Generated {len(comprehensive_viz)} comprehensive visualization files")

                # Generate unified framework dashboard for direct comparison
                logger.info("Generating unified framework dashboard...")
                try:
                    # Build framework_data structure for unified dashboard
                    framework_data_for_dashboard = {}
                    for sim_file in execution_dir.rglob("*simulation_results.json"):
                        try:
                            with open(sim_file, 'r') as f:
                                sim_data = json.load(f)

                            # Determine framework from path
                            path_parts = sim_file.parts
                            framework = "unknown"
                            for part in path_parts:
                                if part in ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy", "pytorch", "numpyro", "bnlearn"]:
                                    framework = part
                                    break

                            if framework != "unknown":
                                key = framework
                                if key not in framework_data_for_dashboard:
                                    framework_data_for_dashboard[key] = {
                                        "framework": framework,
                                        "simulation_data": sim_data
                                    }
                        except Exception as e:
                            logger.debug(f"Skipping {sim_file.name}: {e}")

                    if len(framework_data_for_dashboard) >= 2:
                        model_name = gnn_files[0].stem if gnn_files else "Active Inference Model"
                        dashboard_viz = generate_unified_framework_dashboard(
                            framework_data_for_dashboard,
                            viz_output_dir / "unified_dashboard",
                            model_name=model_name
                        )
                        results["visualization_files"].extend(dashboard_viz)
                        logger.info(f"Generated {len(dashboard_viz)} unified dashboard visualizations")
                    else:
                        logger.info("Less than 2 frameworks with data - skipping unified dashboard")
                except Exception as e:
                    logger.warning(f"Unified dashboard generation failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            except Exception as e:
                logger.warning(f"Comprehensive visualization generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            # 4. Generate unified cross-model comparison report
            try:
                from .generate_cross_model_report import generate_cross_model_report
                report_path = results_dir / "cross_model_comparison_report.md"
                report_file = generate_cross_model_report(
                    execution_dir, results_dir, report_path
                )
                if report_file:
                    results["cross_model_report"] = report_file
                    logger.info(f"Generated cross-model comparison report: {report_file}")
            except Exception as e:
                logger.warning(f"Cross-model report generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Save detailed results
        results_file = results_dir / "analysis_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy_types)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write analysis results: {e}")

        # Generate summary report
        summary = generate_analysis_summary(results)
        summary_file = results_dir / "analysis_summary.md"
        try:
            with open(summary_file, 'w') as f:
                f.write(summary)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write analysis summary: {e}")

        if results["success"]:
            log_step_success(logger, "Analysis processing completed successfully")
        else:
            log_step_error(logger, "Analysis processing failed")

        return results["success"]

    except Exception as e:
        # Use supported signature for log_step_error
        log_step_error(logger, f"Analysis processing failed: {e}")
        return False

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.
    
    Handles:
    - numpy integers, floats, arrays
    - sets and frozensets
    - objects with __dict__ (to prevent circular reference issues)
    - Path objects
    """
    if obj is None:
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__') and not isinstance(obj, dict):
        # Convert objects to dict representation to avoid circular references
        return f"<{type(obj).__name__}>"
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    # Use string representation for any other unknown types to ensure serializability
    # and prevent "Circular reference detected" or "not JSON serializable" errors
    return str(obj)

