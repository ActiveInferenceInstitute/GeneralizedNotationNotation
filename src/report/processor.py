#!/usr/bin/env python3
"""
Report Processor module for GNN Processing Pipeline.

This module provides report processing capabilities.
"""

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

logger = logging.getLogger(__name__)

def process_report(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process comprehensive report for GNN pipeline execution.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("report")
    
    try:
        log_step_start(logger, "Processing comprehensive report")
        
        # Create results directory
        results_dir = output_dir / "comprehensive_report"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find pipeline output directory (parent of current output_dir)
        # The structure is output/23_report_output/23_report_output, so we need to go up two levels
        pipeline_output_dir = output_dir.parent.parent
        
        # Load pipeline execution summary
        pipeline_summary_file = pipeline_output_dir / "pipeline_execution_summary.json"
        logger.info(f"Looking for pipeline summary at: {pipeline_summary_file}")
        logger.info(f"Pipeline output dir: {pipeline_output_dir}")
        logger.info(f"Current output dir: {output_dir}")
        pipeline_data = {}
        if pipeline_summary_file.exists():
            try:
                with open(pipeline_summary_file, 'r', encoding='utf-8') as f:
                    pipeline_data = json.load(f)
                logger.info(f"Loaded pipeline execution summary from {pipeline_summary_file}")
            except Exception as e:
                logger.warning(f"Failed to load pipeline summary: {e}")
        
        # Generate comprehensive analysis
        analysis_results = generate_comprehensive_pipeline_analysis(
            pipeline_data, pipeline_output_dir, logger
        )
        
        # Generate detailed reports
        report_files = []
        
        # Generate JSON summary report
        json_report_file = results_dir / "pipeline_analysis_summary.json"
        with open(json_report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        report_files.append("pipeline_analysis_summary.json")
        
        # Generate HTML report
        html_content = generate_comprehensive_html_report(analysis_results, logger)
        html_report_file = results_dir / "pipeline_analysis_report.html"
        with open(html_report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        report_files.append("pipeline_analysis_report.html")
        
        # Generate Markdown report
        markdown_content = generate_comprehensive_markdown_report(analysis_results, logger)
        markdown_report_file = results_dir / "pipeline_analysis_report.md"
        with open(markdown_report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        report_files.append("pipeline_analysis_report.md")
        
        # Generate executive summary
        executive_summary = generate_executive_summary(analysis_results, logger)
        summary_file = results_dir / "executive_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        report_files.append("executive_summary.md")
        
        # Log results
        logger.info(f"Generated comprehensive report with {len(report_files)} files:")
        for file_name in report_files:
            logger.info(f"  - {file_name}")
        
        logger.info(f"Pipeline health score: {analysis_results.get('health_score', 0)}/100")
        logger.info(f"Total steps analyzed: {len(analysis_results.get('step_analysis', {}))}")
        logger.info(f"Successful steps: {analysis_results.get('summary', {}).get('successful_steps', 0)}")
        logger.info(f"Failed steps: {analysis_results.get('summary', {}).get('failed_steps', 0)}")
        logger.info(f"Warnings: {analysis_results.get('summary', {}).get('total_warnings', 0)}")
        
        log_step_success(logger, "comprehensive report processing completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"report processing failed: {e}")
        return False

def generate_comprehensive_report(
    target_dir: Path,
    output_dir: Path,
    format: str = "json",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a comprehensive report for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Directory to save the report
        format: Output format (json, html, markdown)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with report results
    """
    logger = logging.getLogger("report")
    
    try:
        log_step_start(logger, "Generating comprehensive report")
        
        # Create report directory
        report_dir = output_dir / "comprehensive_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze GNN files
        gnn_files = list(target_dir.glob("*.md"))
        
        report_data = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "total_files": len(gnn_files),
            "files_analyzed": [],
            "summary": {
                "success": True,
                "errors": []
            }
        }
        
        # Process each file
        for gnn_file in gnn_files:
            try:
                file_info = analyze_gnn_file(gnn_file)
                report_data["files_analyzed"].append({
                    "file": str(gnn_file),
                    "info": file_info
                })
            except Exception as e:
                error_info = {
                    "file": str(gnn_file),
                    "error": str(e)
                }
                report_data["summary"]["errors"].append(error_info)
        
        # Generate report in specified format
        if format == "json":
            report_file = report_dir / "comprehensive_report.json"
            import json
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        elif format == "html":
            report_file = report_dir / "comprehensive_report.html"
            html_content = generate_html_report(report_data)
            with open(report_file, 'w') as f:
                f.write(html_content)
        elif format == "markdown":
            report_file = report_dir / "comprehensive_report.md"
            markdown_content = generate_markdown_report(report_data)
            with open(report_file, 'w') as f:
                f.write(markdown_content)
        
        log_step_success(logger, f"Comprehensive report generated in {format} format")
        
        return {
            "success": True,
            "report_file": str(report_file),
            "format": format,
            "files_analyzed": len(report_data["files_analyzed"])
        }
        
    except Exception as e:
        log_step_error(logger, f"Failed to generate comprehensive report: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def analyze_gnn_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a GNN file for report generation.
    
    Args:
        file_path: Path to GNN file
        
    Returns:
        Dictionary with file analysis
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic analysis
        analysis = {
            "file_size": len(content),
            "lines": len(content.split('\n')),
            "sections": [],
            "has_model_name": "ModelName:" in content,
            "has_state_space": "StateSpaceBlock:" in content,
            "has_gnn_version": "GNNVersionAndFlags:" in content
        }
        
        # Extract sections
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                current_section = line[1:].strip()
                analysis["sections"].append(current_section)
        
        return analysis
        
    except Exception as e:
        return {
            "error": str(e)
        }

def generate_html_report(report_data: Dict[str, Any]) -> str:
    """
    Generate HTML report.
    
    Args:
        report_data: Report data dictionary
        
    Returns:
        HTML content string
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GNN Comprehensive Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 10px; }}
            .summary {{ margin: 20px 0; }}
            .file-list {{ margin: 20px 0; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>GNN Comprehensive Report</h1>
            <p>Generated on: {report_data.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total files analyzed: {report_data.get('total_files', 0)}</p>
            <p>Files successfully analyzed: {len(report_data.get('files_analyzed', []))}</p>
            <p>Errors: {len(report_data.get('summary', {}).get('errors', []))}</p>
        </div>
        
        <div class="file-list">
            <h2>Files Analyzed</h2>
            <ul>
    """
    
    for file_info in report_data.get('files_analyzed', []):
        html += f"<li>{file_info['file']}</li>"
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """
    Generate Markdown report.
    
    Args:
        report_data: Report data dictionary
        
    Returns:
        Markdown content string
    """
    markdown = f"""# GNN Comprehensive Report

Generated on: {report_data.get('timestamp', 'Unknown')}

## Summary

- **Total files analyzed**: {report_data.get('total_files', 0)}
- **Files successfully analyzed**: {len(report_data.get('files_analyzed', []))}
- **Errors**: {len(report_data.get('summary', {}).get('errors', []))}

## Files Analyzed

"""
    
    for file_info in report_data.get('files_analyzed', []):
        markdown += f"- {file_info['file']}\n"
    
    return markdown


def generate_comprehensive_pipeline_analysis(
    pipeline_data: Dict[str, Any], 
    pipeline_output_dir: Path, 
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis of pipeline execution.
    
    Args:
        pipeline_data: Pipeline execution summary data
        pipeline_output_dir: Directory containing all pipeline outputs
        logger: Logger for this operation
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    analysis = {
        "report_generation_time": datetime.now().isoformat(),
        "pipeline_summary": pipeline_data,
        "step_analysis": {},
        "file_analysis": {},
        "error_analysis": {},
        "performance_analysis": {},
        "summary": {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "warnings": 0,
            "total_files_generated": 0,
            "total_size_mb": 0.0,
            "health_score": 0.0
        },
        "recommendations": []
    }
    
    try:
        # Analyze pipeline steps
        if "steps" in pipeline_data:
            analysis["step_analysis"] = analyze_pipeline_steps(pipeline_data["steps"], logger)
            analysis["summary"]["total_steps"] = len(pipeline_data["steps"])
            analysis["summary"]["successful_steps"] = len([
                step for step in pipeline_data["steps"] 
                if step.get("status") in ["SUCCESS", "SUCCESS_WITH_WARNINGS"]
            ])
            analysis["summary"]["failed_steps"] = len([
                step for step in pipeline_data["steps"] 
                if step.get("status") in ["FAILED", "TIMEOUT", "ERROR"]
            ])
            analysis["summary"]["warnings"] = len([
                step for step in pipeline_data["steps"] 
                if step.get("status") == "SUCCESS_WITH_WARNINGS"
            ])
        
        # Analyze output files
        analysis["file_analysis"] = analyze_output_files(pipeline_output_dir, logger)
        analysis["summary"]["total_files_generated"] = analysis["file_analysis"].get("total_files", 0)
        analysis["summary"]["total_size_mb"] = analysis["file_analysis"].get("total_size_mb", 0.0)
        
        # Analyze errors and warnings
        analysis["error_analysis"] = analyze_errors_and_warnings(pipeline_data, logger)
        
        # Analyze performance
        analysis["performance_analysis"] = analyze_performance_metrics(pipeline_data, logger)
        
        # Calculate health score
        analysis["summary"]["health_score"] = calculate_pipeline_health_score(analysis)
        
        # Generate recommendations
        analysis["recommendations"] = generate_recommendations(analysis, logger)
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive analysis: {e}")
        analysis["error"] = str(e)
    
    return analysis


def analyze_pipeline_steps(steps: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze individual pipeline steps."""
    step_analysis = {}
    
    for step in steps:
        step_name = step.get("script_name", "unknown")
        step_analysis[step_name] = {
            "status": step.get("status", "UNKNOWN"),
            "duration_seconds": step.get("duration_seconds", 0.0),
            "memory_usage_mb": step.get("memory_usage_mb", 0.0),
            "exit_code": step.get("exit_code", -1),
            "has_stdout": bool(step.get("stdout", "")),
            "has_stderr": bool(step.get("stderr", "")),
            "warnings": extract_warnings_from_stderr(step.get("stderr", "")),
            "errors": extract_errors_from_stderr(step.get("stderr", "")),
            "description": step.get("description", ""),
            "start_time": step.get("start_time", ""),
            "end_time": step.get("end_time", "")
        }
    
    return step_analysis


def analyze_output_files(pipeline_output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze files generated by the pipeline."""
    file_analysis = {
        "total_files": 0,
        "total_size_mb": 0.0,
        "files_by_step": {},
        "file_types": {},
        "largest_files": [],
        "missing_expected_files": []
    }
    
    try:
        # Expected output directories for each step
        expected_dirs = {
            "0_template.py": "0_template_output",
            "1_setup.py": "1_setup_output", 
            "2_tests.py": "2_tests_output",
            "3_gnn.py": "3_gnn_output",
            "4_model_registry.py": "4_model_registry_output",
            "5_type_checker.py": "5_type_checker_output",
            "6_validation.py": "6_validation_output",
            "7_export.py": "7_export_output",
            "8_visualization.py": "8_visualization_output",
            "9_advanced_viz.py": "9_advanced_viz_output",
            "10_ontology.py": "10_ontology_output",
            "11_render.py": "11_render_output",
            "12_execute.py": "12_execute_output",
            "13_llm.py": "13_llm_output",
            "14_ml_integration.py": "14_ml_integration_output",
            "15_audio.py": "15_audio_output",
            "16_analysis.py": "16_analysis_output",
            "17_integration.py": "17_integration_output",
            "18_security.py": "18_security_output",
            "19_research.py": "19_research_output",
            "20_website.py": "20_website_output",
            "21_mcp.py": "21_mcp_output",
            "22_gui.py": "22_gui_output",
            "23_report.py": "23_report_output"
        }
        
        for script_name, expected_dir in expected_dirs.items():
            step_dir = pipeline_output_dir / expected_dir
            step_files = []
            step_size = 0.0
            
            if step_dir.exists():
                for file_path in step_dir.rglob("*"):
                    if file_path.is_file():
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        step_size += file_size_mb
                        file_analysis["total_size_mb"] += file_size_mb
                        file_analysis["total_files"] += 1
                        
                        step_files.append({
                            "name": file_path.name,
                            "path": str(file_path.relative_to(pipeline_output_dir)),
                            "size_mb": round(file_size_mb, 2),
                            "extension": file_path.suffix
                        })
                        
                        # Track file types
                        ext = file_path.suffix.lower()
                        if ext not in file_analysis["file_types"]:
                            file_analysis["file_types"][ext] = {"count": 0, "total_size_mb": 0.0}
                        file_analysis["file_types"][ext]["count"] += 1
                        file_analysis["file_types"][ext]["total_size_mb"] += file_size_mb
            else:
                file_analysis["missing_expected_files"].append(expected_dir)
            
            file_analysis["files_by_step"][script_name] = {
                "directory": expected_dir,
                "exists": step_dir.exists(),
                "file_count": len(step_files),
                "total_size_mb": round(step_size, 2),
                "files": step_files
            }
        
        # Find largest files
        all_files = []
        for step_data in file_analysis["files_by_step"].values():
            all_files.extend(step_data["files"])
        
        file_analysis["largest_files"] = sorted(
            all_files, 
            key=lambda x: x["size_mb"], 
            reverse=True
        )[:10]  # Top 10 largest files
        
        # Round total size
        file_analysis["total_size_mb"] = round(file_analysis["total_size_mb"], 2)
        
    except Exception as e:
        logger.error(f"Failed to analyze output files: {e}")
        file_analysis["error"] = str(e)
    
    return file_analysis


def analyze_errors_and_warnings(pipeline_data: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze errors and warnings from pipeline execution."""
    error_analysis = {
        "total_errors": 0,
        "total_warnings": 0,
        "error_by_step": {},
        "warning_by_step": {},
        "critical_issues": [],
        "common_issues": {}
    }
    
    try:
        if "steps" in pipeline_data:
            for step in pipeline_data["steps"]:
                step_name = step.get("script_name", "unknown")
                stderr = step.get("stderr", "")
                
                # Extract errors and warnings from stderr
                errors = extract_errors_from_stderr(stderr)
                warnings = extract_warnings_from_stderr(stderr)
                
                error_analysis["error_by_step"][step_name] = errors
                error_analysis["warning_by_step"][step_name] = warnings
                error_analysis["total_errors"] += len(errors)
                error_analysis["total_warnings"] += len(warnings)
                
                # Identify critical issues
                if step.get("status") in ["FAILED", "TIMEOUT", "ERROR"]:
                    error_analysis["critical_issues"].append({
                        "step": step_name,
                        "status": step.get("status"),
                        "errors": errors,
                        "description": step.get("description", "")
                    })
        
        # Analyze common issues
        all_errors = []
        for errors in error_analysis["error_by_step"].values():
            all_errors.extend(errors)
        
        # Count common error patterns
        error_patterns = {}
        for error in all_errors:
            # Simple pattern matching for common issues
            if "not available" in error.lower():
                pattern = "Dependency not available"
            elif "timeout" in error.lower():
                pattern = "Timeout"
            elif "permission" in error.lower():
                pattern = "Permission error"
            elif "not found" in error.lower():
                pattern = "File not found"
            else:
                pattern = "Other error"
            
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
        
        error_analysis["common_issues"] = error_patterns
        
    except Exception as e:
        logger.error(f"Failed to analyze errors and warnings: {e}")
        error_analysis["error"] = str(e)
    
    return error_analysis


def analyze_performance_metrics(pipeline_data: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Analyze performance metrics from pipeline execution."""
    performance = {
        "total_duration_seconds": 0.0,
        "average_step_duration": 0.0,
        "slowest_steps": [],
        "memory_usage": {},
        "efficiency_score": 0.0
    }
    
    try:
        if "steps" in pipeline_data:
            step_durations = []
            memory_usage = []
            
            for step in pipeline_data["steps"]:
                duration = step.get("duration_seconds", 0.0)
                memory = step.get("memory_usage_mb", 0.0)
                
                if duration > 0:
                    step_durations.append(duration)
                    performance["total_duration_seconds"] += duration
                
                if memory > 0:
                    memory_usage.append(memory)
                
                # Track slowest steps
                if duration > 1.0:  # Steps taking more than 1 second
                    performance["slowest_steps"].append({
                        "step": step.get("script_name", "unknown"),
                        "duration_seconds": duration,
                        "description": step.get("description", "")
                    })
            
            # Calculate averages
            if step_durations:
                performance["average_step_duration"] = sum(step_durations) / len(step_durations)
            
            if memory_usage:
                performance["memory_usage"] = {
                    "average_mb": sum(memory_usage) / len(memory_usage),
                    "max_mb": max(memory_usage),
                    "min_mb": min(memory_usage)
                }
            
            # Calculate efficiency score (0-100)
            # Based on execution time and success rate
            total_steps = len(pipeline_data["steps"])
            successful_steps = len([s for s in pipeline_data["steps"] if s.get("status") in ["SUCCESS", "SUCCESS_WITH_WARNINGS"]])
            success_rate = (successful_steps / total_steps) * 100 if total_steps > 0 else 0
            
            # Time efficiency (penalize for very long execution)
            time_efficiency = max(0, 100 - (performance["total_duration_seconds"] / 60))  # Penalty for each minute
            
            performance["efficiency_score"] = (success_rate + time_efficiency) / 2
            
            # Sort slowest steps
            performance["slowest_steps"].sort(key=lambda x: x["duration_seconds"], reverse=True)
            performance["slowest_steps"] = performance["slowest_steps"][:5]  # Top 5 slowest
        
    except Exception as e:
        logger.error(f"Failed to analyze performance metrics: {e}")
        performance["error"] = str(e)
    
    return performance


def extract_errors_from_stderr(stderr: str) -> List[str]:
    """Extract error messages from stderr output."""
    errors = []
    if not stderr:
        return errors
    
    lines = stderr.split('\n')
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['error:', 'failed:', 'exception:', 'traceback']):
            errors.append(line)
    
    return errors


def extract_warnings_from_stderr(stderr: str) -> List[str]:
    """Extract warning messages from stderr output."""
    warnings = []
    if not stderr:
        return warnings
    
    lines = stderr.split('\n')
    for line in lines:
        line = line.strip()
        if 'warning:' in line.lower() or 'warn:' in line.lower():
            warnings.append(line)
    
    return warnings


def calculate_pipeline_health_score(analysis: Dict[str, Any]) -> float:
    """Calculate overall pipeline health score (0-100)."""
    try:
        score = 0.0
        total_weight = 0.0
        
        # Step success rate (40% weight)
        summary = analysis.get("summary", {})
        total_steps = summary.get("total_steps", 0)
        successful_steps = summary.get("successful_steps", 0)
        if total_steps > 0:
            step_success_rate = (successful_steps / total_steps) * 100
            score += step_success_rate * 0.4
            total_weight += 0.4
        
        # File generation success (30% weight)
        file_analysis = analysis.get("file_analysis", {})
        total_files = file_analysis.get("total_files", 0)
        if total_files > 0:
            # Assume good if we have reasonable number of files
            file_score = min(100, (total_files / 50) * 100)  # Scale based on expected files
            score += file_score * 0.3
            total_weight += 0.3
        
        # Error rate (20% weight)
        error_analysis = analysis.get("error_analysis", {})
        total_errors = error_analysis.get("total_errors", 0)
        total_warnings = error_analysis.get("total_warnings", 0)
        
        # Penalize for errors and warnings
        error_penalty = min(100, (total_errors * 10) + (total_warnings * 2))
        error_score = max(0, 100 - error_penalty)
        score += error_score * 0.2
        total_weight += 0.2
        
        # Performance (10% weight)
        performance = analysis.get("performance_analysis", {})
        efficiency_score = performance.get("efficiency_score", 50)
        score += efficiency_score * 0.1
        total_weight += 0.1
        
        return round(score / total_weight, 1) if total_weight > 0 else 0.0
        
    except Exception:
        return 0.0


def generate_recommendations(analysis: Dict[str, Any], logger: logging.Logger) -> List[str]:
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    try:
        # Check for failed steps
        failed_steps = analysis.get("summary", {}).get("failed_steps", 0)
        if failed_steps > 0:
            recommendations.append(f"Address {failed_steps} failed pipeline steps to improve reliability")
        
        # Check for warnings
        warnings = analysis.get("summary", {}).get("warnings", 0)
        if warnings > 0:
            recommendations.append(f"Review {warnings} steps with warnings to prevent future issues")
        
        # Check for missing files
        missing_files = analysis.get("file_analysis", {}).get("missing_expected_files", [])
        if missing_files:
            recommendations.append(f"Investigate missing output directories: {', '.join(missing_files)}")
        
        # Check for critical issues
        critical_issues = analysis.get("error_analysis", {}).get("critical_issues", [])
        if critical_issues:
            recommendations.append("Address critical issues that caused step failures")
        
        # Check performance
        slowest_steps = analysis.get("performance_analysis", {}).get("slowest_steps", [])
        if slowest_steps:
            slowest = slowest_steps[0]
            recommendations.append(f"Optimize {slowest['step']} which took {slowest['duration_seconds']:.1f}s")
        
        # Check health score
        health_score = analysis.get("summary", {}).get("health_score", 0)
        if health_score < 70:
            recommendations.append("Overall pipeline health is low - review all issues systematically")
        elif health_score < 90:
            recommendations.append("Pipeline health is good but could be improved")
        else:
            recommendations.append("Pipeline is performing well - consider optimization opportunities")
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        recommendations.append("Error generating recommendations - check logs")
    
    return recommendations


def generate_comprehensive_html_report(analysis: Dict[str, Any], logger: logging.Logger) -> str:
    """Generate comprehensive HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #333; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
        .step {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745; }}
        .step.failed {{ border-left-color: #dc3545; background: #f8d7da; }}
        .step.warning {{ border-left-color: #ffc107; background: #fff3cd; }}
        .step.timeout {{ border-left-color: #6c757d; background: #e2e3e5; }}
        .file-list {{ background: #f8f9fa; padding: 15px; border-radius: 8px; }}
        .file-item {{ padding: 5px 0; border-bottom: 1px solid #dee2e6; }}
        .file-item:last-child {{ border-bottom: none; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .success {{ color: #28a745; }}
        .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .recommendations ul {{ margin: 10px 0; }}
        .recommendations li {{ margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .health-score {{ font-size: 3em; font-weight: bold; text-align: center; margin: 20px 0; }}
        .health-score.good {{ color: #28a745; }}
        .health-score.warning {{ color: #ffc107; }}
        .health-score.poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GNN Pipeline Analysis Report</h1>
        <p>Generated on: {analysis.get('report_generation_time', 'Unknown')}</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Health Score</h3>
            <div class="health-score {'good' if analysis.get('summary', {}).get('health_score', 0) >= 80 else 'warning' if analysis.get('summary', {}).get('health_score', 0) >= 60 else 'poor'}">
                {analysis.get('summary', {}).get('health_score', 0)}/100
            </div>
        </div>
        <div class="summary-card">
            <h3>Total Steps</h3>
            <div class="value">{analysis.get('summary', {}).get('total_steps', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Successful</h3>
            <div class="value success">{analysis.get('summary', {}).get('successful_steps', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Failed</h3>
            <div class="value error">{analysis.get('summary', {}).get('failed_steps', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Warnings</h3>
            <div class="value warning">{analysis.get('summary', {}).get('warnings', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Files Generated</h3>
            <div class="value">{analysis.get('summary', {}).get('total_files_generated', 0)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Pipeline Steps Analysis</h2>
        {generate_steps_html(analysis.get('step_analysis', {}))}
    </div>
    
    <div class="section">
        <h2>File Generation Analysis</h2>
        {generate_files_html(analysis.get('file_analysis', {}))}
    </div>
    
    <div class="section">
        <h2>Error Analysis</h2>
        {generate_errors_html(analysis.get('error_analysis', {}))}
    </div>
    
    <div class="section">
        <h2>Performance Analysis</h2>
        {generate_performance_html(analysis.get('performance_analysis', {}))}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in analysis.get('recommendations', []))}
            </ul>
        </div>
    </div>
</body>
</html>"""
    
    return html


def generate_steps_html(step_analysis: Dict[str, Any]) -> str:
    """Generate HTML for steps analysis."""
    if not step_analysis:
        return "<p>No step analysis data available.</p>"
    
    html = ""
    for step_name, step_data in step_analysis.items():
        status = step_data.get("status", "UNKNOWN")
        css_class = "step"
        if status in ["FAILED", "ERROR"]:
            css_class += " failed"
        elif status == "SUCCESS_WITH_WARNINGS":
            css_class += " warning"
        elif status == "TIMEOUT":
            css_class += " timeout"
        
        html += f"""
        <div class="{css_class}">
            <h3>{step_name}</h3>
            <p><strong>Status:</strong> <span class="{'error' if status in ['FAILED', 'ERROR'] else 'warning' if status == 'SUCCESS_WITH_WARNINGS' else 'success'}">{status}</span></p>
            <p><strong>Duration:</strong> {step_data.get('duration_seconds', 0):.2f}s</p>
            <p><strong>Memory:</strong> {step_data.get('memory_usage_mb', 0):.1f} MB</p>
            <p><strong>Description:</strong> {step_data.get('description', 'N/A')}</p>
        """
        
        if step_data.get("errors"):
            html += "<p><strong>Errors:</strong></p><ul>"
            for error in step_data["errors"]:
                html += f"<li class='error'>{error}</li>"
            html += "</ul>"
        
        if step_data.get("warnings"):
            html += "<p><strong>Warnings:</strong></p><ul>"
            for warning in step_data["warnings"]:
                html += f"<li class='warning'>{warning}</li>"
            html += "</ul>"
        
        html += "</div>"
    
    return html


def generate_files_html(file_analysis: Dict[str, Any]) -> str:
    """Generate HTML for file analysis."""
    if not file_analysis:
        return "<p>No file analysis data available.</p>"
    
    html = f"""
    <p><strong>Total Files Generated:</strong> {file_analysis.get('total_files', 0)}</p>
    <p><strong>Total Size:</strong> {file_analysis.get('total_size_mb', 0):.2f} MB</p>
    """
    
    if file_analysis.get("file_types"):
        html += "<h3>File Types</h3><table><tr><th>Type</th><th>Count</th><th>Total Size (MB)</th></tr>"
        for ext, data in file_analysis["file_types"].items():
            html += f"<tr><td>{ext or 'no extension'}</td><td>{data['count']}</td><td>{data['total_size_mb']:.2f}</td></tr>"
        html += "</table>"
    
    if file_analysis.get("largest_files"):
        html += "<h3>Largest Files</h3><div class='file-list'>"
        for file_info in file_analysis["largest_files"][:10]:
            html += f"<div class='file-item'>{file_info['name']} ({file_info['size_mb']:.2f} MB)</div>"
        html += "</div>"
    
    return html


def generate_errors_html(error_analysis: Dict[str, Any]) -> str:
    """Generate HTML for error analysis."""
    if not error_analysis:
        return "<p>No error analysis data available.</p>"
    
    html = f"""
    <p><strong>Total Errors:</strong> <span class='error'>{error_analysis.get('total_errors', 0)}</span></p>
    <p><strong>Total Warnings:</strong> <span class='warning'>{error_analysis.get('total_warnings', 0)}</span></p>
    """
    
    if error_analysis.get("critical_issues"):
        html += "<h3>Critical Issues</h3><ul>"
        for issue in error_analysis["critical_issues"]:
            html += f"<li class='error'><strong>{issue['step']}:</strong> {issue['status']} - {issue['description']}</li>"
        html += "</ul>"
    
    if error_analysis.get("common_issues"):
        html += "<h3>Common Issues</h3><table><tr><th>Issue Type</th><th>Count</th></tr>"
        for issue_type, count in error_analysis["common_issues"].items():
            html += f"<tr><td>{issue_type}</td><td>{count}</td></tr>"
        html += "</table>"
    
    return html


def generate_performance_html(performance: Dict[str, Any]) -> str:
    """Generate HTML for performance analysis."""
    if not performance:
        return "<p>No performance data available.</p>"
    
    html = f"""
    <p><strong>Total Duration:</strong> {performance.get('total_duration_seconds', 0):.2f} seconds</p>
    <p><strong>Average Step Duration:</strong> {performance.get('average_step_duration', 0):.2f} seconds</p>
    <p><strong>Efficiency Score:</strong> {performance.get('efficiency_score', 0):.1f}/100</p>
    """
    
    if performance.get("memory_usage"):
        mem = performance["memory_usage"]
        html += f"""
        <h3>Memory Usage</h3>
        <p><strong>Average:</strong> {mem.get('average_mb', 0):.1f} MB</p>
        <p><strong>Maximum:</strong> {mem.get('max_mb', 0):.1f} MB</p>
        <p><strong>Minimum:</strong> {mem.get('min_mb', 0):.1f} MB</p>
        """
    
    if performance.get("slowest_steps"):
        html += "<h3>Slowest Steps</h3><table><tr><th>Step</th><th>Duration (s)</th><th>Description</th></tr>"
        for step in performance["slowest_steps"]:
            html += f"<tr><td>{step['step']}</td><td>{step['duration_seconds']:.2f}</td><td>{step['description']}</td></tr>"
        html += "</table>"
    
    return html


def generate_comprehensive_markdown_report(analysis: Dict[str, Any], logger: logging.Logger) -> str:
    """Generate comprehensive Markdown report."""
    markdown = f"""# GNN Pipeline Analysis Report

**Generated:** {analysis.get('report_generation_time', 'Unknown')}

## Executive Summary

- **Health Score:** {analysis.get('summary', {}).get('health_score', 0)}/100
- **Total Steps:** {analysis.get('summary', {}).get('total_steps', 0)}
- **Successful Steps:** {analysis.get('summary', {}).get('successful_steps', 0)}
- **Failed Steps:** {analysis.get('summary', {}).get('failed_steps', 0)}
- **Warnings:** {analysis.get('summary', {}).get('warnings', 0)}
- **Files Generated:** {analysis.get('summary', {}).get('total_files_generated', 0)}
- **Total Size:** {analysis.get('summary', {}).get('total_size_mb', 0):.2f} MB

## Pipeline Steps Analysis

"""
    
    # Add steps analysis
    step_analysis = analysis.get('step_analysis', {})
    for step_name, step_data in step_analysis.items():
        status = step_data.get("status", "UNKNOWN")
        status_emoji = "✅" if status == "SUCCESS" else "⚠️" if status == "SUCCESS_WITH_WARNINGS" else "❌"
        
        markdown += f"""
### {status_emoji} {step_name}
- **Status:** {status}
- **Duration:** {step_data.get('duration_seconds', 0):.2f}s
- **Memory:** {step_data.get('memory_usage_mb', 0):.1f} MB
- **Description:** {step_data.get('description', 'N/A')}
"""
        
        if step_data.get("errors"):
            markdown += "- **Errors:**\n"
            for error in step_data["errors"]:
                markdown += f"  - ❌ {error}\n"
        
        if step_data.get("warnings"):
            markdown += "- **Warnings:**\n"
            for warning in step_data["warnings"]:
                markdown += f"  - ⚠️ {warning}\n"
        
        markdown += "\n"
    
    # Add file analysis
    file_analysis = analysis.get('file_analysis', {})
    if file_analysis:
        markdown += """
## File Generation Analysis

"""
        markdown += f"- **Total Files:** {file_analysis.get('total_files', 0)}\n"
        markdown += f"- **Total Size:** {file_analysis.get('total_size_mb', 0):.2f} MB\n\n"
        
        if file_analysis.get("file_types"):
            markdown += "### File Types\n\n"
            markdown += "| Type | Count | Total Size (MB) |\n"
            markdown += "|------|-------|----------------|\n"
            for ext, data in file_analysis["file_types"].items():
                markdown += f"| {ext or 'no extension'} | {data['count']} | {data['total_size_mb']:.2f} |\n"
            markdown += "\n"
        
        if file_analysis.get("largest_files"):
            markdown += "### Largest Files\n\n"
            for file_info in file_analysis["largest_files"][:10]:
                markdown += f"- {file_info['name']} ({file_info['size_mb']:.2f} MB)\n"
            markdown += "\n"
    
    # Add error analysis
    error_analysis = analysis.get('error_analysis', {})
    if error_analysis:
        markdown += """
## Error Analysis

"""
        markdown += f"- **Total Errors:** {error_analysis.get('total_errors', 0)}\n"
        markdown += f"- **Total Warnings:** {error_analysis.get('total_warnings', 0)}\n\n"
        
        if error_analysis.get("critical_issues"):
            markdown += "### Critical Issues\n\n"
            for issue in error_analysis["critical_issues"]:
                markdown += f"- ❌ **{issue['step']}:** {issue['status']} - {issue['description']}\n"
            markdown += "\n"
        
        if error_analysis.get("common_issues"):
            markdown += "### Common Issues\n\n"
            markdown += "| Issue Type | Count |\n"
            markdown += "|------------|-------|\n"
            for issue_type, count in error_analysis["common_issues"].items():
                markdown += f"| {issue_type} | {count} |\n"
            markdown += "\n"
    
    # Add performance analysis
    performance = analysis.get('performance_analysis', {})
    if performance:
        markdown += """
## Performance Analysis

"""
        markdown += f"- **Total Duration:** {performance.get('total_duration_seconds', 0):.2f} seconds\n"
        markdown += f"- **Average Step Duration:** {performance.get('average_step_duration', 0):.2f} seconds\n"
        markdown += f"- **Efficiency Score:** {performance.get('efficiency_score', 0):.1f}/100\n\n"
        
        if performance.get("memory_usage"):
            mem = performance["memory_usage"]
            markdown += f"- **Average Memory:** {mem.get('average_mb', 0):.1f} MB\n"
            markdown += f"- **Maximum Memory:** {mem.get('max_mb', 0):.1f} MB\n"
            markdown += f"- **Minimum Memory:** {mem.get('min_mb', 0):.1f} MB\n\n"
        
        if performance.get("slowest_steps"):
            markdown += "### Slowest Steps\n\n"
            markdown += "| Step | Duration (s) | Description |\n"
            markdown += "|------|--------------|-------------|\n"
            for step in performance["slowest_steps"]:
                markdown += f"| {step['step']} | {step['duration_seconds']:.2f} | {step['description']} |\n"
            markdown += "\n"
    
    # Add recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        markdown += """
## Recommendations

"""
        for i, rec in enumerate(recommendations, 1):
            markdown += f"{i}. {rec}\n"
        markdown += "\n"
    
    return markdown


def generate_executive_summary(analysis: Dict[str, Any], logger: logging.Logger) -> str:
    """Generate executive summary of the pipeline analysis."""
    summary = analysis.get('summary', {})
    health_score = summary.get('health_score', 0)
    
    # Determine overall status
    if health_score >= 90:
        status = "🟢 EXCELLENT"
        status_color = "green"
    elif health_score >= 70:
        status = "🟡 GOOD"
        status_color = "yellow"
    elif health_score >= 50:
        status = "🟠 FAIR"
        status_color = "orange"
    else:
        status = "🔴 POOR"
        status_color = "red"
    
    executive_summary = f"""# GNN Pipeline Executive Summary

## Overall Status: {status}

**Health Score:** {health_score}/100

## Key Metrics

- **Pipeline Steps:** {summary.get('total_steps', 0)} total
  - ✅ Successful: {summary.get('successful_steps', 0)}
  - ❌ Failed: {summary.get('failed_steps', 0)}
  - ⚠️ Warnings: {summary.get('warnings', 0)}

- **Output Generation:**
  - Files Created: {summary.get('total_files_generated', 0)}
  - Total Size: {summary.get('total_size_mb', 0):.2f} MB

## Critical Issues

"""
    
    # Add critical issues
    error_analysis = analysis.get('error_analysis', {})
    critical_issues = error_analysis.get('critical_issues', [])
    
    if critical_issues:
        for issue in critical_issues:
            executive_summary += f"- ❌ **{issue['step']}:** {issue['status']} - {issue['description']}\n"
    else:
        executive_summary += "- ✅ No critical issues identified\n"
    
    executive_summary += "\n## Top Recommendations\n\n"
    
    # Add top recommendations
    recommendations = analysis.get('recommendations', [])
    for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
        executive_summary += f"{i}. {rec}\n"
    
    executive_summary += f"""
## Next Steps

1. **Immediate Actions:** Address any critical issues identified above
2. **Short-term:** Implement the top recommendations to improve pipeline health
3. **Long-term:** Monitor pipeline performance and establish regular health checks

---
*Report generated on {analysis.get('report_generation_time', 'Unknown')}*
"""
    
    return executive_summary
