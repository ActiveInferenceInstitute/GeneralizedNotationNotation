#!/usr/bin/env python3
"""
Report Formatters Module

This module provides formatting functionality for generating HTML and Markdown reports
from pipeline analysis data.
"""

import logging
from typing import Dict, Any
from .analyzer import get_pipeline_health_score

def generate_html_report(pipeline_data: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Generate an HTML report from pipeline data.
    
    Args:
        pipeline_data: Collected pipeline data
        logger: Logger for this operation
        
    Returns:
        HTML content as string
    """
    # Calculate health score
    health_score = get_pipeline_health_score(pipeline_data)
    health_color = get_health_color(health_score)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Comprehensive Analysis Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa; 
            color: #333;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 40px; 
            padding-bottom: 20px; 
            border-bottom: 3px solid #007bff; 
        }}
        h1 {{ 
            color: #2c3e50; 
            margin: 0; 
            font-size: 2.5em; 
            font-weight: 300; 
        }}
        h2 {{ 
            color: #34495e; 
            margin-top: 40px; 
            margin-bottom: 20px; 
            font-size: 1.8em; 
            border-left: 4px solid #007bff; 
            padding-left: 15px; 
        }}
        h3 {{ 
            color: #495057; 
            margin-top: 25px; 
            margin-bottom: 15px; 
            font-size: 1.4em; 
        }}
        .health-score {{ 
            display: inline-block; 
            background-color: {health_color}; 
            color: white; 
            padding: 10px 20px; 
            border-radius: 25px; 
            font-size: 1.2em; 
            font-weight: bold; 
            margin: 10px 0; 
        }}
        .summary-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin: 30px 0; 
        }}
        .summary-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 25px; 
            border-radius: 10px; 
            text-align: center; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }}
        .summary-card h3 {{ 
            margin: 0 0 15px 0; 
            font-size: 1.1em; 
            opacity: 0.9; 
        }}
        .summary-card .value {{ 
            font-size: 2em; 
            font-weight: bold; 
            margin: 10px 0; 
        }}
        .step-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin: 30px 0; 
        }}
        .step-card {{ 
            background-color: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 10px; 
            padding: 20px; 
            transition: transform 0.2s, box-shadow 0.2s; 
        }}
        .step-card:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 6px 12px rgba(0,0,0,0.15); 
        }}
        .step-card.success {{ 
            border-left: 4px solid #28a745; 
        }}
        .step-card.error {{ 
            border-left: 4px solid #dc3545; 
        }}
        .step-card.missing {{ 
            border-left: 4px solid #ffc107; 
            opacity: 0.7; 
        }}
        .step-title {{ 
            font-weight: bold; 
            color: #495057; 
            margin-bottom: 15px; 
            font-size: 1.1em; 
        }}
        .step-details {{ 
            font-size: 14px; 
            color: #6c757d; 
            line-height: 1.6; 
        }}
        .step-details p {{ 
            margin: 8px 0; 
        }}
        .file-types {{ 
            display: flex; 
            flex-wrap: wrap; 
            gap: 5px; 
            margin-top: 10px; 
        }}
        .file-type-tag {{ 
            background-color: #e9ecef; 
            color: #495057; 
            padding: 2px 8px; 
            border-radius: 12px; 
            font-size: 12px; 
        }}
        .performance-section {{ 
            background-color: #e8f5e8; 
            border: 1px solid #4caf50; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 30px 0; 
        }}
        .error-section {{ 
            background-color: #ffebee; 
            border: 1px solid #f44336; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 30px 0; 
        }}
        .warning-section {{ 
            background-color: #fff3e0; 
            border: 1px solid #ff9800; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 30px 0; 
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
            background-color: white; 
            border-radius: 8px; 
            overflow: hidden; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }}
        th, td {{ 
            border: 1px solid #dee2e6; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #f8f9fa; 
            font-weight: 600; 
            color: #495057; 
        }}
        .timestamp {{ 
            color: #6c757d; 
            font-size: 14px; 
            text-align: center; 
            margin-top: 20px; 
        }}
        .key-files {{ 
            margin-top: 15px; 
        }}
        .key-file {{ 
            background-color: #e3f2fd; 
            padding: 8px 12px; 
            border-radius: 6px; 
            margin: 5px 0; 
            font-size: 13px; 
        }}
        .dependency-chain {{ 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
        }}
        .missing-deps {{ 
            color: #dc3545; 
            font-weight: bold; 
        }}
        .complete-deps {{ 
            color: #28a745; 
            font-weight: bold; 
        }}
        .chart-container {{ 
            margin: 30px 0; 
            text-align: center; 
        }}
        .progress-bar {{ 
            width: 100%; 
            height: 20px; 
            background-color: #e9ecef; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 10px 0; 
        }}
        .progress-fill {{ 
            height: 100%; 
            background: linear-gradient(90deg, #28a745, #20c997); 
            transition: width 0.3s ease; 
        }}
        @media (max-width: 768px) {{
            .container {{ 
                padding: 15px; 
            }}
            .summary-grid {{ 
                grid-template-columns: 1fr; 
            }}
            .step-grid {{ 
                grid-template-columns: 1fr; 
            }}
            h1 {{ 
                font-size: 2em; 
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 GNN Pipeline Comprehensive Analysis Report</h1>
            <div class="health-score">Health Score: {health_score}/100</div>
            <p class="timestamp">Generated: {pipeline_data.get('report_generation_time', 'Unknown')}</p>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Pipeline Output Directory</h3>
                <div class="value">{pipeline_data.get('pipeline_output_directory', 'Unknown')}</div>
            </div>
            <div class="summary-card">
                <h3>Total Steps Analyzed</h3>
                <div class="value">{len(pipeline_data.get('steps', {}))}</div>
            </div>
            <div class="summary-card">
                <h3>Total Files Processed</h3>
                <div class="value">{pipeline_data.get('summary', {}).get('total_files_processed', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Total Size</h3>
                <div class="value">{pipeline_data.get('summary', {}).get('total_size_mb', 0)} MB</div>
            </div>
        </div>
        
        <h2>📊 Pipeline Overview</h2>
        <div class="performance-section">
            <h3>Success Rate</h3>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {pipeline_data.get('summary', {}).get('success_rate', 0)}%"></div>
            </div>
            <p><strong>{pipeline_data.get('summary', {}).get('success_rate', 0):.1f}%</strong> of pipeline steps completed successfully</p>
        </div>
"""
    
    # Add performance metrics if available
    performance_metrics = pipeline_data.get('performance_metrics', {})
    if performance_metrics:
        html_content += f"""
        <h2>⚡ Performance Metrics</h2>
        <div class="performance-section">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
"""
        for metric, value in performance_metrics.items():
            html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        html_content += """
            </table>
        </div>
"""
    
    # Add error analysis if available
    error_analysis = pipeline_data.get('error_analysis', {})
    if error_analysis and error_analysis.get('total_errors', 0) > 0:
        html_content += f"""
        <h2>⚠️ Error Analysis</h2>
        <div class="error-section">
            <h3>Total Errors: {error_analysis.get('total_errors', 0)}</h3>
            <h4>Error Types:</h4>
            <ul>
"""
        for error_type, count in error_analysis.get('error_types', {}).items():
            html_content += f"<li><strong>{error_type}:</strong> {count}</li>"
        html_content += """
            </ul>
        </div>
"""
    
    # Add step-by-step analysis
    html_content += """
        <h2>🔍 Step-by-Step Analysis</h2>
        <div class="step-grid">
"""
    
    for step_name, step_data in pipeline_data.get('steps', {}).items():
        status_class = "success" if step_data.get('exists', False) else "missing"
        if step_data.get('status') == "error":
            status_class = "error"
        
        html_content += f"""
        <div class="step-card {status_class}">
            <div class="step-title">📁 {step_name.replace('_', ' ').title()}</div>
            <div class="step-details">
"""
        
        if step_data.get('exists', False):
            html_content += f"""
                <p><strong>Files:</strong> {step_data.get('file_count', 0)}</p>
                <p><strong>Size:</strong> {step_data.get('total_size_mb', 0)} MB</p>
                <p><strong>Last Modified:</strong> {step_data.get('last_modified', 'Unknown')}</p>
                <p><strong>Status:</strong> {step_data.get('status', 'success')}</p>
"""
            
            # Add file types
            file_types = step_data.get('file_types', {})
            if file_types:
                html_content += '<div class="file-types">'
                for ext, info in file_types.items():
                    count = info.get('count', 0) if isinstance(info, dict) else info
                    html_content += f'<span class="file-type-tag">{ext}: {count}</span>'
                html_content += '</div>'
            
            # Add key files
            key_files = step_data.get('key_files', [])
            if key_files:
                html_content += '<div class="key-files"><strong>Key Files:</strong>'
                for key_file in key_files[:3]:  # Show first 3 key files
                    html_content += f'<div class="key-file">{key_file["name"]} ({key_file["size_mb"]} MB)</div>'
                html_content += '</div>'
            
            # Add dependency information
            dependencies = pipeline_data.get('step_dependencies', {}).get('dependency_chain', {}).get(step_name, {})
            if dependencies:
                missing = dependencies.get('missing_prerequisites', [])
                if missing:
                    html_content += f'<div class="dependency-chain"><strong>Missing Dependencies:</strong> <span class="missing-deps">{", ".join(missing)}</span></div>'
                else:
                    html_content += '<div class="dependency-chain"><strong>Dependencies:</strong> <span class="complete-deps">Complete</span></div>'
        else:
            html_content += '<p><em>Step directory not found or empty</em></p>'
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
        </div>
"""
    
    # Add file type analysis
    file_type_analysis = pipeline_data.get('file_type_analysis', {})
    if file_type_analysis.get('total_by_type'):
        html_content += """
        <h2>📁 File Type Analysis</h2>
        <table>
            <tr><th>File Type</th><th>Count</th><th>Total Size (MB)</th></tr>
"""
        for file_ext, info in file_type_analysis['total_by_type'].items():
            html_content += f"<tr><td>{file_ext}</td><td>{info['count']}</td><td>{info['total_size_mb']}</td></tr>"
        html_content += """
        </table>
"""
    
    # Add pipeline summary if available
    if 'pipeline_summary' in pipeline_data:
        summary = pipeline_data['pipeline_summary']
        html_content += f"""
        <h2>📈 Pipeline Execution Summary</h2>
        <div class="performance-section">
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
    # Calculate health score
    health_score = get_pipeline_health_score(pipeline_data)
    
    markdown_content = f"""# 🎯 GNN Pipeline Comprehensive Analysis Report

**Generated:** {pipeline_data.get('report_generation_time', 'Unknown')}  
**Pipeline Output Directory:** {pipeline_data.get('pipeline_output_directory', 'Unknown')}  
**Health Score:** {health_score}/100

## 📊 Pipeline Overview

- **Total Steps Analyzed:** {len(pipeline_data.get('steps', {}))}
- **Total Files Processed:** {pipeline_data.get('summary', {}).get('total_files_processed', 0)}
- **Total Size:** {pipeline_data.get('summary', {}).get('total_size_mb', 0)} MB
- **Success Rate:** {pipeline_data.get('summary', {}).get('success_rate', 0):.1f}%

## ⚡ Performance Metrics

"""
    
    # Add performance metrics if available
    performance_metrics = pipeline_data.get('performance_metrics', {})
    if performance_metrics:
        for metric, value in performance_metrics.items():
            markdown_content += f"- **{metric.replace('_', ' ').title()}:** {value}\n"
    else:
        markdown_content += "*No performance metrics available*\n"
    
    # Add error analysis if available
    error_analysis = pipeline_data.get('error_analysis', {})
    if error_analysis and error_analysis.get('total_errors', 0) > 0:
        markdown_content += f"""
## ⚠️ Error Analysis

**Total Errors:** {error_analysis.get('total_errors', 0)}

### Error Types:
"""
        for error_type, count in error_analysis.get('error_types', {}).items():
            markdown_content += f"- **{error_type}:** {count}\n"
        
        if error_analysis.get('critical_errors'):
            markdown_content += f"\n**Critical Errors:** {len(error_analysis['critical_errors'])}\n"
        
        if error_analysis.get('warnings'):
            markdown_content += f"\n**Warnings:** {len(error_analysis['warnings'])}\n"
    
    # Add step-by-step analysis
    markdown_content += """
## 🔍 Step-by-Step Analysis

"""
    
    for step_name, step_data in pipeline_data.get('steps', {}).items():
        if step_data.get('exists', False):
            status_icon = "✅" if step_data.get('status') == "success" else "⚠️" if step_data.get('status') == "error" else "❌"
            markdown_content += f"""### {status_icon} {step_name.replace('_', ' ').title()}

- **Files:** {step_data.get('file_count', 0)}
- **Size:** {step_data.get('total_size_mb', 0)} MB
- **Last Modified:** {step_data.get('last_modified', 'Unknown')}
- **Status:** {step_data.get('status', 'success')}

"""
            
            # Add file types
            file_types = step_data.get('file_types', {})
            if file_types:
                file_type_list = []
                for ext, info in file_types.items():
                    count = info.get('count', 0) if isinstance(info, dict) else info
                    file_type_list.append(f"{ext}: {count}")
                markdown_content += f"- **File Types:** {', '.join(file_type_list)}\n"
            
            # Add key files
            key_files = step_data.get('key_files', [])
            if key_files:
                markdown_content += "- **Key Files:**\n"
                for key_file in key_files[:3]:  # Show first 3 key files
                    markdown_content += f"  - {key_file['name']} ({key_file['size_mb']} MB)\n"
            
            # Add dependency information
            dependencies = pipeline_data.get('step_dependencies', {}).get('dependency_chain', {}).get(step_name, {})
            if dependencies:
                missing = dependencies.get('missing_prerequisites', [])
                if missing:
                    markdown_content += f"- **Missing Dependencies:** {', '.join(missing)}\n"
                else:
                    markdown_content += "- **Dependencies:** Complete\n"
            
            markdown_content += "\n"
        else:
            markdown_content += f"""### ❌ {step_name.replace('_', ' ').title()}

*Step directory not found or empty*

"""
    
    # Add file type analysis
    file_type_analysis = pipeline_data.get('file_type_analysis', {})
    if file_type_analysis.get('total_by_type'):
        markdown_content += """
## 📁 File Type Analysis

| File Type | Count | Total Size (MB) |
|-----------|-------|-----------------|
"""
        for file_ext, info in file_type_analysis['total_by_type'].items():
            markdown_content += f"| {file_ext} | {info['count']} | {info['total_size_mb']} |\n"
    
    # Add dependency analysis
    dependencies = pipeline_data.get('step_dependencies', {})
    if dependencies.get('missing_prerequisites'):
        markdown_content += """
## 🔗 Dependency Analysis

### Missing Prerequisites:
"""
        for missing in dependencies['missing_prerequisites']:
            markdown_content += f"- **{missing['step']}** missing: {', '.join(missing['missing'])}\n"
    
    # Add pipeline summary if available
    if 'pipeline_summary' in pipeline_data:
        summary = pipeline_data['pipeline_summary']
        markdown_content += f"""
## 📈 Pipeline Execution Summary

- **Start Time:** {summary.get('start_time', 'Unknown')}
- **End Time:** {summary.get('end_time', 'Unknown')}
- **Overall Status:** {summary.get('overall_status', 'Unknown')}
- **Total Steps:** {len(summary.get('steps', []))}

"""
    
    return markdown_content

def get_health_color(health_score: float) -> str:
    """
    Get color for health score display.
    
    Args:
        health_score: Health score between 0 and 100
        
    Returns:
        CSS color string
    """
    if health_score >= 80:
        return "#28a745"  # Green
    elif health_score >= 60:
        return "#ffc107"  # Yellow
    elif health_score >= 40:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red 