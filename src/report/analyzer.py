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
from typing import Dict, Any, List, Optional, Tuple

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
            "processing_time": 0.0,
            "total_size_mb": 0.0,
            "success_rate": 0.0
        },
        "performance_metrics": {},
        "error_analysis": {},
        "file_type_analysis": {},
        "step_dependencies": {}
    }
    
    # Check for pipeline execution summary
    pipeline_summary_file = pipeline_output_dir / "pipeline_execution_summary.json"
    if pipeline_summary_file.exists():
        try:
            with open(pipeline_summary_file, 'r', encoding='utf-8') as f:
                pipeline_data["pipeline_summary"] = json.load(f)
                
            # Extract performance metrics from summary
            summary = pipeline_data["pipeline_summary"]
            if "performance_metrics" in summary:
                pipeline_data["performance_metrics"] = summary["performance_metrics"]
            
            # Extract error analysis from summary
            if "errors" in summary:
                pipeline_data["error_analysis"] = analyze_errors(summary["errors"], logger)
                
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
            
            # Update summary statistics
            pipeline_data["summary"]["total_files_processed"] += step_data.get("file_count", 0)
            pipeline_data["summary"]["total_size_mb"] += step_data.get("total_size_mb", 0)
        else:
            pipeline_data["steps"][step_dir] = {
                "directory": str(step_path),
                "exists": False,
                "file_count": 0,
                "total_size_mb": 0.0,
                "file_types": {},
                "last_modified": None,
                "status": "missing"
            }
    
    # Calculate success rate
    total_steps = len(step_directories)
    successful_steps = len([step for step in pipeline_data["steps"].values() if step.get("exists", False)])
    pipeline_data["summary"]["success_rate"] = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
    
    # Analyze file types across all steps
    pipeline_data["file_type_analysis"] = analyze_file_types_across_steps(pipeline_data["steps"], logger)
    
    # Analyze step dependencies
    pipeline_data["step_dependencies"] = analyze_step_dependencies(pipeline_data["steps"], logger)
    
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
        "last_modified": None,
        "status": "success",
        "performance_metrics": {},
        "error_logs": [],
        "key_files": []
    }
    
    try:
        # Count files and calculate sizes
        for file_path in step_path.rglob("*"):
            if file_path.is_file():
                step_data["file_count"] += 1
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                step_data["total_size_mb"] += file_size_mb
                
                # Track file types
                file_ext = file_path.suffix.lower()
                if file_ext not in step_data["file_types"]:
                    step_data["file_types"][file_ext] = {"count": 0, "total_size_mb": 0.0}
                step_data["file_types"][file_ext]["count"] += 1
                step_data["file_types"][file_ext]["total_size_mb"] += file_size_mb
                
                # Track last modification
                mtime = file_path.stat().st_mtime
                if step_data["last_modified"] is None or mtime > step_data["last_modified"]:
                    step_data["last_modified"] = mtime
                
                # Identify key files
                if is_key_file(file_path, step_name):
                    step_data["key_files"].append({
                        "name": file_path.name,
                        "size_mb": round(file_size_mb, 2),
                        "type": file_ext
                    })
        
        # Convert timestamp to ISO format
        if step_data["last_modified"]:
            step_data["last_modified"] = datetime.datetime.fromtimestamp(
                step_data["last_modified"]
            ).isoformat()
        
        # Round size to 2 decimal places
        step_data["total_size_mb"] = round(step_data["total_size_mb"], 2)
        
        # Analyze step-specific performance and errors
        step_data.update(analyze_step_specific_data(step_path, step_name, logger))
        
    except Exception as e:
        logger.warning(f"Failed to analyze step directory {step_name}: {e}")
        step_data["error"] = str(e)
        step_data["status"] = "error"
    
    return step_data

def analyze_step_specific_data(step_path: Path, step_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze step-specific data like performance metrics and error logs.
    
    Args:
        step_path: Path to the step directory
        step_name: Name of the step
        logger: Logger for this operation
        
    Returns:
        Dictionary with step-specific analysis data
    """
    step_data = {}
    
    try:
        # Look for performance metrics files
        perf_files = list(step_path.glob("*performance*.json")) + list(step_path.glob("*metrics*.json"))
        for perf_file in perf_files:
            try:
                with open(perf_file, 'r', encoding='utf-8') as f:
                    step_data["performance_metrics"] = json.load(f)
                break
            except Exception as e:
                logger.debug(f"Failed to read performance file {perf_file}: {e}")
        
        # Look for error logs
        log_files = list(step_path.glob("*.log")) + list(step_path.glob("*error*.json"))
        error_logs = []
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "error" in content.lower() or "warning" in content.lower():
                        error_logs.append({
                            "file": log_file.name,
                            "size_mb": round(log_file.stat().st_size / (1024 * 1024), 2)
                        })
            except Exception as e:
                logger.debug(f"Failed to read log file {log_file}: {e}")
        
        if error_logs:
            step_data["error_logs"] = error_logs
            
    except Exception as e:
        logger.debug(f"Failed to analyze step-specific data for {step_name}: {e}")
    
    return step_data

def analyze_file_types_across_steps(steps: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze file types across all pipeline steps.
    
    Args:
        steps: Dictionary of step data
        logger: Logger for this operation
        
    Returns:
        Dictionary with file type analysis
    """
    file_type_analysis = {
        "total_by_type": {},
        "largest_files_by_type": {},
        "step_distribution": {}
    }
    
    try:
        for step_name, step_data in steps.items():
            if not step_data.get("exists", False):
                continue
                
            for file_ext, file_info in step_data.get("file_types", {}).items():
                # Track total counts and sizes
                if file_ext not in file_type_analysis["total_by_type"]:
                    file_type_analysis["total_by_type"][file_ext] = {
                        "count": 0,
                        "total_size_mb": 0.0
                    }
                
                file_type_analysis["total_by_type"][file_ext]["count"] += file_info["count"]
                file_type_analysis["total_by_type"][file_ext]["total_size_mb"] += file_info["total_size_mb"]
                
                # Track step distribution
                if file_ext not in file_type_analysis["step_distribution"]:
                    file_type_analysis["step_distribution"][file_ext] = {}
                
                file_type_analysis["step_distribution"][file_ext][step_name] = {
                    "count": file_info["count"],
                    "size_mb": file_info["total_size_mb"]
                }
        
        # Round sizes
        for file_ext in file_type_analysis["total_by_type"]:
            file_type_analysis["total_by_type"][file_ext]["total_size_mb"] = round(
                file_type_analysis["total_by_type"][file_ext]["total_size_mb"], 2
            )
            
    except Exception as e:
        logger.warning(f"Failed to analyze file types: {e}")
    
    return file_type_analysis

def analyze_step_dependencies(steps: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze dependencies between pipeline steps.
    
    Args:
        steps: Dictionary of step data
        logger: Logger for this operation
        
    Returns:
        Dictionary with dependency analysis
    """
    dependencies = {
        "step_order": [
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
        ],
        "dependency_chain": {},
        "missing_prerequisites": []
    }
    
    try:
        # Define step dependencies
        step_deps = {
            "gnn_processing_step": ["setup_artifacts"],
            "test_reports": ["gnn_processing_step"],
            "type_check": ["gnn_processing_step"],
            "gnn_exports": ["type_check"],
            "visualization": ["gnn_exports"],
            "mcp_processing_step": ["gnn_exports"],
            "ontology_processing": ["gnn_exports"],
            "gnn_rendered_simulators": ["gnn_exports"],
            "execution_results": ["gnn_rendered_simulators"],
            "llm_processing_step": ["gnn_exports"],
            "audio_processing_step": ["gnn_processing_step"],
            "website": ["visualization", "ontology_processing"],
            "report_processing_step": ["visualization", "ontology_processing", "audio_processing_step"]
        }
        
        # Check for missing prerequisites
        for step_name, prereqs in step_deps.items():
            if step_name in steps and steps[step_name].get("exists", False):
                missing = []
                for prereq in prereqs:
                    if prereq not in steps or not steps[prereq].get("exists", False):
                        missing.append(prereq)
                
                if missing:
                    dependencies["missing_prerequisites"].append({
                        "step": step_name,
                        "missing": missing
                    })
                
                dependencies["dependency_chain"][step_name] = {
                    "prerequisites": prereqs,
                    "missing_prerequisites": missing,
                    "status": "complete" if not missing else "incomplete"
                }
                
    except Exception as e:
        logger.warning(f"Failed to analyze step dependencies: {e}")
    
    return dependencies

def analyze_errors(errors: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze error patterns from pipeline execution.
    
    Args:
        errors: List of error dictionaries
        logger: Logger for this operation
        
    Returns:
        Dictionary with error analysis
    """
    error_analysis = {
        "total_errors": len(errors),
        "error_types": {},
        "step_error_distribution": {},
        "critical_errors": [],
        "warnings": []
    }
    
    try:
        for error in errors:
            error_type = error.get("type", "unknown")
            step_name = error.get("step", "unknown")
            
            # Count error types
            if error_type not in error_analysis["error_types"]:
                error_analysis["error_types"][error_type] = 0
            error_analysis["error_types"][error_type] += 1
            
            # Count errors by step
            if step_name not in error_analysis["step_error_distribution"]:
                error_analysis["step_error_distribution"][step_name] = 0
            error_analysis["step_error_distribution"][step_name] += 1
            
            # Categorize by severity
            if error.get("severity", "error") == "critical":
                error_analysis["critical_errors"].append(error)
            elif error.get("severity", "error") == "warning":
                error_analysis["warnings"].append(error)
                
    except Exception as e:
        logger.warning(f"Failed to analyze errors: {e}")
    
    return error_analysis

def is_key_file(file_path: Path, step_name: str) -> bool:
    """
    Determine if a file is a key file for a specific step.
    
    Args:
        file_path: Path to the file
        step_name: Name of the step
        
    Returns:
        True if the file is a key file for the step
    """
    key_patterns = {
        "setup_artifacts": ["installed_packages.json", "directory_structure.json"],
        "gnn_processing_step": ["gnn_discovery_report.json", "*.md"],
        "test_reports": ["pytest_report.xml", "test_summary.json"],
        "type_check": ["type_check_report.md", "type_check_summary.json"],
        "gnn_exports": ["*.json", "*.xml", "*.graphml"],
        "visualization": ["*.png", "*.svg", "*.html"],
        "mcp_processing_step": ["mcp_processing_report.json"],
        "ontology_processing": ["ontology_analysis.json", "ontology_summary.md"],
        "gnn_rendered_simulators": ["*.py", "*.jl", "*.toml"],
        "execution_results": ["execution_results.json", "*.md"],
        "llm_processing_step": ["llm_analysis.json", "*.md"],
        "audio_processing_step": ["*.wav", "*.mp3", "audio_analysis.json"],
        "website": ["*.html", "*.css", "*.js"],
        "report_processing_step": ["comprehensive_analysis_report.html", "report_summary.json"]
    }
    
    if step_name not in key_patterns:
        return False
    
    file_name = file_path.name
    for pattern in key_patterns[step_name]:
        if pattern.startswith("*."):
            if file_name.endswith(pattern[1:]):
                return True
        elif file_name == pattern:
            return True
    
    return False

def get_pipeline_health_score(pipeline_data: Dict[str, Any]) -> float:
    """
    Calculate a health score for the pipeline based on various metrics.
    
    Args:
        pipeline_data: Pipeline analysis data
        
    Returns:
        Health score between 0 and 100
    """
    try:
        score = 0.0
        total_weight = 0.0
        
        # Step completion rate (40% weight)
        steps = pipeline_data.get("steps", {})
        successful_steps = len([step for step in steps.values() if step.get("exists", False)])
        total_steps = len(steps)
        step_completion_rate = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
        score += step_completion_rate * 0.4
        total_weight += 0.4
        
        # File processing success (30% weight)
        summary = pipeline_data.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        score += success_rate * 0.3
        total_weight += 0.3
        
        # Error rate (20% weight)
        error_analysis = pipeline_data.get("error_analysis", {})
        total_errors = error_analysis.get("total_errors", 0)
        critical_errors = len(error_analysis.get("critical_errors", []))
        
        # Penalize for errors
        error_penalty = min(total_errors * 5 + critical_errors * 10, 100)
        score += (100 - error_penalty) * 0.2
        total_weight += 0.2
        
        # Performance (10% weight)
        performance_metrics = pipeline_data.get("performance_metrics", {})
        if "execution_time" in performance_metrics:
            # Assume good performance if under 10 minutes
            exec_time = performance_metrics["execution_time"]
            if exec_time < 600:  # 10 minutes
                score += 100 * 0.1
            else:
                score += max(0, 100 - (exec_time - 600) / 60) * 0.1
        else:
            score += 50 * 0.1  # Default score if no performance data
        total_weight += 0.1
        
        return round(score / total_weight, 1) if total_weight > 0 else 0.0
        
    except Exception:
        return 0.0 