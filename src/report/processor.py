#!/usr/bin/env python3
"""
Report Processor module for GNN Processing Pipeline.

This module provides report processing capabilities.
"""

from pathlib import Path
from typing import Dict, Any, List
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
    Process report for GNN files.
    
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
        log_step_start(logger, "Processing report")
        
        # Create results directory
        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic report processing
        results = {
            "processed_files": 0,
            "success": True,
            "errors": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if gnn_files:
            results["processed_files"] = len(gnn_files)
        
        # Save results
        import json
        results_file = results_dir / "report_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results["success"]:
            log_step_success(logger, "report processing completed successfully")
        else:
            log_step_error(logger, "report processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, "report processing failed", {"error": str(e)})
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
        report_dir = output_dir
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
