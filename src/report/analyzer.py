#!/usr/bin/env python3
"""
Report Analyzer Module

This module provides data collection and analysis functionality for pipeline reporting.
It analyzes pipeline outputs and extracts relevant metrics and statistics.
"""

import json
import datetime
import logging
from pathlib import Path
from typing import Dict, Any

def collect_pipeline_data(pipeline_output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Collect data from all pipeline outputs for analysis.
    
    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        logger: Logger for this operation
        
    Returns:
        Dictionary containing collected pipeline data
    """
    pipeline_data = {
        "report_generation_time": datetime.datetime.now().isoformat(),
        "pipeline_output_directory": str(pipeline_output_dir),
        "steps": {},
        "summary": {
            "total_files_processed": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "processing_time": 0.0
        }
    }
    
    # Check for pipeline execution summary
    pipeline_summary_file = pipeline_output_dir / "pipeline_execution_summary.json"
    if pipeline_summary_file.exists():
        try:
            with open(pipeline_summary_file, 'r', encoding='utf-8') as f:
                pipeline_data["pipeline_summary"] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read pipeline summary: {e}")
    
    # Collect data from each step directory
    step_directories = [
        "setup_artifacts",
        "gnn_processing_step", 
        "test_reports",
        "type_check",
        "gnn_exports",
        "visualization",
        "mcp_processing_step",
        "ontology_processing",
        "gnn_rendered_simulators",
        "execution_results",
        "llm_processing_step",
        "audio_processing_step",
        "website",
        "report_processing_step"
    ]
    
    for step_dir in step_directories:
        step_path = pipeline_output_dir / step_dir
        if step_path.exists():
            step_data = analyze_step_directory(step_path, step_dir, logger)
            pipeline_data["steps"][step_dir] = step_data
    
    return pipeline_data

def analyze_step_directory(step_path: Path, step_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze a specific step directory and extract relevant data.
    
    Args:
        step_path: Path to the step directory
        step_name: Name of the step
        logger: Logger for this operation
        
    Returns:
        Dictionary containing step analysis data
    """
    step_data = {
        "directory": str(step_path),
        "exists": True,
        "file_count": 0,
        "total_size_mb": 0.0,
        "file_types": {},
        "last_modified": None
    }
    
    try:
        # Count files and calculate sizes
        for file_path in step_path.rglob("*"):
            if file_path.is_file():
                step_data["file_count"] += 1
                step_data["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                
                # Track file types
                file_ext = file_path.suffix.lower()
                step_data["file_types"][file_ext] = step_data["file_types"].get(file_ext, 0) + 1
                
                # Track last modification
                mtime = file_path.stat().st_mtime
                if step_data["last_modified"] is None or mtime > step_data["last_modified"]:
                    step_data["last_modified"] = mtime
        
        # Convert timestamp to ISO format
        if step_data["last_modified"]:
            step_data["last_modified"] = datetime.datetime.fromtimestamp(
                step_data["last_modified"]
            ).isoformat()
        
        # Round size to 2 decimal places
        step_data["total_size_mb"] = round(step_data["total_size_mb"], 2)
        
    except Exception as e:
        logger.warning(f"Failed to analyze step directory {step_name}: {e}")
        step_data["error"] = str(e)
    
    return step_data 