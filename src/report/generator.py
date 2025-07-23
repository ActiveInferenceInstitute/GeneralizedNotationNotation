#!/usr/bin/env python3
"""
Report Generator Module

This module provides the main report generation functionality for the GNN pipeline.
It orchestrates the collection, analysis, and formatting of pipeline data into comprehensive reports.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .analyzer import collect_pipeline_data
from .formatters import generate_html_report, generate_markdown_report

def generate_comprehensive_report(
    pipeline_output_dir: Path, 
    report_output_dir: Path, 
    logger: logging.Logger
) -> bool:
    """
    Generate a comprehensive analysis report from all pipeline outputs.
    
    Args:
        pipeline_output_dir: Directory containing all pipeline outputs
        report_output_dir: Directory to save the generated report
        logger: Logger for this operation
        
    Returns:
        True if report generation succeeded, False otherwise
    """
    try:
        logger.info("Analyzing pipeline outputs for comprehensive report generation")
        
        # Collect data from all pipeline steps
        pipeline_data = collect_pipeline_data(pipeline_output_dir, logger)
        
        # Generate HTML report
        html_content = generate_html_report(pipeline_data, logger)
        
        # Write HTML report
        report_file = report_output_dir / "comprehensive_analysis_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate JSON summary
        summary_file = report_output_dir / "report_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        # Generate markdown report
        markdown_content = generate_markdown_report(pipeline_data, logger)
        markdown_file = report_output_dir / "comprehensive_analysis_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Generated comprehensive report with {len(pipeline_data)} data points")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        return False 