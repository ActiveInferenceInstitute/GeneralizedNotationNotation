#!/usr/bin/env python3
"""
Analysis processor module for GNN analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import numpy as np
import re
from datetime import datetime

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from .analyzer import (
    perform_statistical_analysis,
    calculate_complexity_metrics,
    run_performance_benchmarks,
    perform_model_comparisons,
    generate_analysis_summary,
)

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
        
        # Create results directory
        results_dir = output_dir / "analysis_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "statistical_analysis": [],
            "complexity_metrics": [],
            "performance_benchmarks": [],
            "model_comparisons": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
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
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
            
            # Perform cross-model comparisons if multiple files
            if len(gnn_files) > 1:
                comparisons = perform_model_comparisons(results["statistical_analysis"], verbose)
                results["model_comparisons"].append(comparisons)
        
        # Save detailed results
        results_file = results_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy_types)
        
        # Generate summary report
        summary = generate_analysis_summary(results)
        summary_file = results_dir / "analysis_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "Analysis processing completed successfully")
        else:
            log_step_error(logger, "Analysis processing failed")
        
        return results["success"]
        
    except Exception as e:
        # Use supported signature for log_step_error
        log_step_error(logger, f"Analysis processing failed: {e}")
        return False

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
