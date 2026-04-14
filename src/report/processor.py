#!/usr/bin/env python3
"""
Report Processor module for GNN Processing Pipeline.

This module provides report processing capabilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from utils.pipeline_template import log_step_error, log_step_start, log_step_success

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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GNN Comprehensive Analysis Report</title>
        <style>
            :root {{
                --bg-color: #fcfcfc;
                --text-main: #333333;
                --text-muted: #666666;
                --accent: #2c3e50;
                --border: #e2e8f0;
            }}
            body {{
                font-family: 'Merriweather', 'Georgia', serif;
                line-height: 1.8;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px 20px;
                background-color: var(--bg-color);
                color: var(--text-main);
            }}
            .manuscript-header {{
                text-align: center;
                border-bottom: 2px solid var(--accent);
                padding-bottom: 20px;
                margin-bottom: 40px;
            }}
            h1 {{ font-size: 2.2em; color: var(--accent); margin-bottom: 10px; font-family: 'Inter', sans-serif; }}
            h2 {{ font-size: 1.5em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 5px; margin-top: 40px; font-family: 'Inter', sans-serif; }}
            .metadata-block {{
                background: #ffffff;
                border: 1px solid var(--border);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            }}
            .metadata-block p {{ margin: 5px 0; font-family: 'Inter', sans-serif; font-size: 0.9em; }}
            .file-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .file-card {{
                background: #ffffff;
                border: 1px solid var(--border);
                padding: 15px;
                border-radius: 6px;
                font-family: 'Inter', sans-serif;
                font-size: 0.9em;
            }}
            .file-card code {{ color: #e53e3e; background: #fff5f5; padding: 2px 4px; border-radius: 4px; }}
            .mermaid-container {{ margin: 40px 0; text-align: center; }}
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
        </script>
    </head>
    <body>
        <div class="manuscript-header">
            <h1>GNN Pipeline Complete Output Report</h1>
            <p style="font-style: italic; color: var(--text-muted);">Generated on: {report_data.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="mermaid-container">
            <div class="mermaid">
            graph LR
                A[GNN Input Files] --> B{{GNN Processor}}
                B --> C[Serialization]
                B --> D[Semantic Ontology]
                B --> E[LLM Inference]
                C --> F[Simulation Generation]
                D --> F
                E --> G[Final Analysis]
                F --> G
            </div>
            <p style="font-size: 0.8em; color: var(--text-muted);">Figure 1. GNN pipeline flow diagram demonstrating data integration topologies.</p>
        </div>

        <h2>I. Executive Summary</h2>
        <div class="metadata-block">
            <p><strong>Total Scanned Entities:</strong> {report_data.get('total_files', 0)}</p>
            <p><strong>Entities Successfully Evaluated:</strong> {len(report_data.get('files_analyzed', []))}</p>
            <p><strong>Evaluation Errors:</strong> {len(report_data.get('summary', {}).get('errors', []))}</p>
        </div>
        
        <h2>II. Processed Models Validation</h2>
        <div class="file-grid">
    """

    for file_info in report_data.get('files_analyzed', []):
        info = file_info.get('info', {})
        size = info.get('file_size', 0)
        lines = info.get('lines', 0)
        html += f"""
            <div class="file-card">
                <strong>{Path(file_info['file']).name}</strong>
                <p>Size: {size} bytes | Lines: {lines}</p>
                <p>State Space Matrix: <code>{"Yes" if info.get('has_state_space') else "No"}</code></p>
            </div>
        """

    html += """
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
    markdown = f"""# GNN Comprehensive Analysis Report

> **Generated on:** {report_data.get('timestamp', 'Unknown')}
> **Purpose:** Top-level structural audit of GNN notation topologies.

## System Topology Flow

```mermaid
graph LR
    A[GNN Input Files] --> B{{GNN Processor}}
    B --> C[Serialization]
    B --> D[Semantic Ontology]
    B --> E[LLM Inference]
    C --> F[Simulation Generation]
    D --> F
    E --> G[Final Analysis]
    F --> G
```

## I. Executive Summary

- **Total Scanned Entities**: {report_data.get('total_files', 0)}
- **Entities Successfully Evaluated**: {len(report_data.get('files_analyzed', []))}
- **Evaluation Errors**: {len(report_data.get('summary', {}).get('errors', []))}

## II. Processed Models Validation

| Model Filename | Size (Bytes) | Lines | State Space Discovered |
|---|---|---|---|
"""

    for file_info in report_data.get('files_analyzed', []):
        info = file_info.get('info', {})
        name = Path(file_info['file']).name
        size = info.get('file_size', 0)
        lines = info.get('lines', 0)
        has_state = "✅ Yes" if info.get('has_state_space') else "❌ No"
        markdown += f"| `{name}` | {size} | {lines} | {has_state} |\n"

    return markdown

# Explicit alias so __init__.py can import by this name without shadowing generator's generate_comprehensive_report
generate_report = generate_comprehensive_report
