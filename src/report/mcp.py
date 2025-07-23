#!/usr/bin/env python3
"""
Report Generation MCP Integration

This module provides Model Context Protocol (MCP) integration for the report generation
module, exposing comprehensive reporting capabilities as callable tools.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .generator import generate_comprehensive_report
from .analyzer import collect_pipeline_data, analyze_step_directory
from .formatters import generate_html_report, generate_markdown_report

logger = logging.getLogger(__name__)

def generate_pipeline_report(
    pipeline_output_dir: str,
    report_output_dir: Optional[str] = None,
    report_format: str = "html"
) -> Dict[str, Any]:
    """
    Generate a comprehensive pipeline report.
    
    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        report_output_dir: Directory to save generated reports (optional)
        report_format: Report format (html, markdown, json, all)
        
    Returns:
        Dictionary with report generation results
    """
    try:
        pipeline_path = Path(pipeline_output_dir)
        if not pipeline_path.exists():
            return {
                "success": False,
                "error": f"Pipeline output directory does not exist: {pipeline_output_dir}"
            }
        
        # Use default report directory if not specified
        if report_output_dir is None:
            report_path = pipeline_path / "report_processing_step"
        else:
            report_path = Path(report_output_dir)
        
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive report
        success = generate_comprehensive_report(pipeline_path, report_path, logger)
        
        if not success:
            return {
                "success": False,
                "error": "Report generation failed"
            }
        
        # Collect generated files
        generated_files = []
        for file_path in report_path.glob("*"):
            if file_path.is_file():
                generated_files.append({
                    "name": file_path.name,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "type": file_path.suffix
                })
        
        return {
            "success": True,
            "pipeline_directory": str(pipeline_path),
            "report_directory": str(report_path),
            "generated_files": generated_files,
            "format": report_format,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def analyze_pipeline_data(pipeline_output_dir: str) -> Dict[str, Any]:
    """
    Analyze pipeline data without generating reports.
    
    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        
    Returns:
        Dictionary with pipeline analysis data
    """
    try:
        pipeline_path = Path(pipeline_output_dir)
        if not pipeline_path.exists():
            return {
                "success": False,
                "error": f"Pipeline output directory does not exist: {pipeline_output_dir}"
            }
        
        # Collect pipeline data
        pipeline_data = collect_pipeline_data(pipeline_path, logger)
        
        # Calculate summary statistics
        total_files = sum(step.get('file_count', 0) for step in pipeline_data.get('steps', {}).values())
        total_size_mb = sum(step.get('total_size_mb', 0) for step in pipeline_data.get('steps', {}).values())
        
        analysis_summary = {
            "total_steps_analyzed": len(pipeline_data.get('steps', {})),
            "total_files_processed": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "steps_with_data": len([step for step in pipeline_data.get('steps', {}).values() if step.get('exists', False)]),
            "steps_missing": len([step for step in pipeline_data.get('steps', {}).values() if not step.get('exists', False)])
        }
        
        return {
            "success": True,
            "pipeline_directory": str(pipeline_path),
            "analysis_summary": analysis_summary,
            "step_details": pipeline_data.get('steps', {}),
            "pipeline_summary": pipeline_data.get('pipeline_summary'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pipeline analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_custom_report(
    pipeline_output_dir: str,
    report_output_dir: str,
    report_format: str = "html",
    include_steps: Optional[List[str]] = None,
    exclude_steps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate a custom report with specific step filtering.
    
    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        report_output_dir: Directory to save generated reports
        report_format: Report format (html, markdown, json)
        include_steps: List of steps to include (optional)
        exclude_steps: List of steps to exclude (optional)
        
    Returns:
        Dictionary with custom report generation results
    """
    try:
        pipeline_path = Path(pipeline_output_dir)
        report_path = Path(report_output_dir)
        
        if not pipeline_path.exists():
            return {
                "success": False,
                "error": f"Pipeline output directory does not exist: {pipeline_output_dir}"
            }
        
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Collect pipeline data
        pipeline_data = collect_pipeline_data(pipeline_path, logger)
        
        # Filter steps if specified
        if include_steps or exclude_steps:
            filtered_steps = {}
            for step_name, step_data in pipeline_data.get('steps', {}).items():
                if include_steps and step_name not in include_steps:
                    continue
                if exclude_steps and step_name in exclude_steps:
                    continue
                filtered_steps[step_name] = step_data
            
            pipeline_data['steps'] = filtered_steps
        
        # Generate report based on format
        if report_format == "html":
            content = generate_html_report(pipeline_data, logger)
            report_file = report_path / "custom_analysis_report.html"
        elif report_format == "markdown":
            content = generate_markdown_report(pipeline_data, logger)
            report_file = report_path / "custom_analysis_report.md"
        elif report_format == "json":
            content = json.dumps(pipeline_data, indent=2, default=str)
            report_file = report_path / "custom_analysis_report.json"
        else:
            return {
                "success": False,
                "error": f"Unsupported report format: {report_format}"
            }
        
        # Write report file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "pipeline_directory": str(pipeline_path),
            "report_directory": str(report_path),
            "report_file": str(report_file),
            "format": report_format,
            "steps_included": len(pipeline_data.get('steps', {})),
            "file_size_mb": round(report_file.stat().st_size / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Custom report generation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_report_module_info() -> Dict[str, Any]:
    """
    Get information about the report generation module.
    
    Returns:
        Dictionary with module information
    """
    return {
        "module_name": "report",
        "description": "Comprehensive analysis and reporting capabilities for the GNN pipeline",
        "version": "1.0.0",
        "available_functions": [
            "generate_pipeline_report",
            "analyze_pipeline_data", 
            "generate_custom_report",
            "get_report_module_info"
        ],
        "supported_formats": ["html", "markdown", "json"],
        "supported_steps": [
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
        "dependencies": [
            "pathlib",
            "json", 
            "datetime",
            "logging"
        ]
    }

def register_tools(mcp) -> None:
    """
    Register report generation tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    try:
        # Register report generation tools
        mcp.register_tool(
            "generate_pipeline_report",
            generate_pipeline_report,
            "Generate a comprehensive pipeline report from all pipeline outputs"
        )
        
        mcp.register_tool(
            "analyze_pipeline_data", 
            analyze_pipeline_data,
            "Analyze pipeline data without generating reports"
        )
        
        mcp.register_tool(
            "generate_custom_report",
            generate_custom_report,
            "Generate a custom report with specific step filtering"
        )
        
        mcp.register_tool(
            "get_report_module_info",
            get_report_module_info,
            "Get information about the report generation module"
        )
        
        logger.info("Report generation tools registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to register report generation tools: {e}")
        raise

