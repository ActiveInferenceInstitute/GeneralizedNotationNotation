#!/usr/bin/env python3
"""
Report Formatters Module

This module provides formatting functionality for generating HTML and Markdown reports
from pipeline analysis data.
"""

import logging
from typing import Dict, Any

def generate_html_report(pipeline_data: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Generate an HTML report from pipeline data.
    
    Args:
        pipeline_data: Collected pipeline data
        logger: Logger for this operation
        
    Returns:
        HTML content as string
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Comprehensive Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .step-card {{ background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin: 10px 0; }}
        .step-title {{ font-weight: bold; color: #495057; margin-bottom: 10px; }}
        .step-details {{ font-size: 14px; color: #6c757d; }}
        .summary {{ background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 15px; margin: 20px 0; }}
        .error {{ background-color: #ffebee; border: 1px solid #f44336; }}
        .warning {{ background-color: #fff3e0; border: 1px solid #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .timestamp {{ color: #6c757d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 GNN Pipeline Comprehensive Analysis Report</h1>
        <p class="timestamp">Generated: {pipeline_data.get('report_generation_time', 'Unknown')}</p>
        
        <div class="summary">
            <h2>📊 Pipeline Overview</h2>
            <p><strong>Output Directory:</strong> {pipeline_data.get('pipeline_output_directory', 'Unknown')}</p>
            <p><strong>Total Steps Analyzed:</strong> {len(pipeline_data.get('steps', {}))}</p>
        </div>
        
        <h2>🔍 Step-by-Step Analysis</h2>
"""
    
    # Add step details
    for step_name, step_data in pipeline_data.get('steps', {}).items():
        if step_data.get('exists', False):
            html_content += f"""
        <div class="step-card">
            <div class="step-title">📁 {step_name.replace('_', ' ').title()}</div>
            <div class="step-details">
                <p><strong>Files:</strong> {step_data.get('file_count', 0)}</p>
                <p><strong>Size:</strong> {step_data.get('total_size_mb', 0)} MB</p>
                <p><strong>Last Modified:</strong> {step_data.get('last_modified', 'Unknown')}</p>
                <p><strong>File Types:</strong> {', '.join([f'{ext}: {count}' for ext, count in step_data.get('file_types', {}).items()])}</p>
            </div>
        </div>
"""
        else:
            html_content += f"""
        <div class="step-card error">
            <div class="step-title">❌ {step_name.replace('_', ' ').title()}</div>
            <div class="step-details">
                <p>Step directory not found or empty</p>
            </div>
        </div>
"""
    
    # Add pipeline summary if available
    if 'pipeline_summary' in pipeline_data:
        summary = pipeline_data['pipeline_summary']
        html_content += f"""
        <h2>📈 Pipeline Execution Summary</h2>
        <div class="summary">
            <p><strong>Start Time:</strong> {summary.get('start_time', 'Unknown')}</p>
            <p><strong>End Time:</strong> {summary.get('end_time', 'Unknown')}</p>
            <p><strong>Overall Status:</strong> {summary.get('overall_status', 'Unknown')}</p>
            <p><strong>Total Steps:</strong> {len(summary.get('steps', []))}</p>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return html_content

def generate_markdown_report(pipeline_data: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Generate a markdown report from pipeline data.
    
    Args:
        pipeline_data: Collected pipeline data
        logger: Logger for this operation
        
    Returns:
        Markdown content as string
    """
    markdown_content = f"""# 🎯 GNN Pipeline Comprehensive Analysis Report

**Generated:** {pipeline_data.get('report_generation_time', 'Unknown')}  
**Pipeline Output Directory:** {pipeline_data.get('pipeline_output_directory', 'Unknown')}

## 📊 Pipeline Overview

- **Total Steps Analyzed:** {len(pipeline_data.get('steps', {}))}
- **Report Generation Time:** {pipeline_data.get('report_generation_time', 'Unknown')}

## 🔍 Step-by-Step Analysis

"""
    
    # Add step details
    for step_name, step_data in pipeline_data.get('steps', {}).items():
        if step_data.get('exists', False):
            markdown_content += f"""### 📁 {step_name.replace('_', ' ').title()}

- **Files:** {step_data.get('file_count', 0)}
- **Size:** {step_data.get('total_size_mb', 0)} MB
- **Last Modified:** {step_data.get('last_modified', 'Unknown')}
- **File Types:** {', '.join([f'{ext}: {count}' for ext, count in step_data.get('file_types', {}).items()])}

"""
        else:
            markdown_content += f"""### ❌ {step_name.replace('_', ' ').title()}

*Step directory not found or empty*

"""
    
    # Add pipeline summary if available
    if 'pipeline_summary' in pipeline_data:
        summary = pipeline_data['pipeline_summary']
        markdown_content += f"""## 📈 Pipeline Execution Summary

- **Start Time:** {summary.get('start_time', 'Unknown')}
- **End Time:** {summary.get('end_time', 'Unknown')}
- **Overall Status:** {summary.get('overall_status', 'Unknown')}
- **Total Steps:** {len(summary.get('steps', []))}

"""
    
    return markdown_content 