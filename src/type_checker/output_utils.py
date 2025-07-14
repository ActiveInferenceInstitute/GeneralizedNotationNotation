"""
Output utilities for GNN Type Checker

Provides modular functions for:
- Per-file detailed Markdown and JSON reports
- Cross-file summary and statistics (Markdown, JSON, CSV)
- Static artifacts (variable tables, section matrices, etc.)
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def write_markdown(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)

def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def write_csv(path: Path, rows: List[List[Any]], header: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

def per_file_markdown_report(filename: str, result: Dict[str, Any]) -> str:
    """Generate a detailed Markdown report for a single file's type check result."""
    lines = [f"# Type Check Report: {filename}"]
    lines.append(f"Status: {'✅ VALID' if result['is_valid'] else '❌ INVALID'}")
    lines.append(f"File: {result.get('file_path', 'Unknown')}\n")
    
    if result.get('errors'):
        lines.append("## Errors:")
        for e in result['errors']:
            lines.append(f"- {e}")
        lines.append("")
    
    if result.get('warnings'):
        lines.append("## Warnings:")
        for w in result['warnings']:
            lines.append(f"- {w}")
        lines.append("")
    
    # Model Overview
    lines.append("## Model Overview")
    lines.append(f"- **Model Type**: {result.get('model_type', 'Unknown')}")
    lines.append(f"- **Variables**: {result.get('variable_count', 0)}")
    lines.append(f"- **Connections**: {result.get('connection_count', 0)}")
    lines.append(f"- **Overall Complexity**: {result.get('model_complexity', {}).get('overall_complexity', 0):.2f}")
    lines.append("")
    
    # Section presence
    if 'sections' in result:
        lines.append("## Section Presence:")
        required_sections = ['GNNSection', 'GNNVersionAndFlags', 'ModelName', 'StateSpaceBlock', 'Connections', 'Footer', 'Signature']
        optional_sections = ['ModelAnnotation', 'InitialParameterization', 'Equations', 'Time', 'ActInfOntologyAnnotation', 'ModelParameters']
        
        lines.append("### Required Sections:")
        for sec in required_sections:
            present = result['sections'].get(sec, False)
            lines.append(f"- {sec}: {'✅' if present else '❌'}")
        
        lines.append("\n### Optional Sections:")
        for sec in optional_sections:
            present = result['sections'].get(sec, False)
            lines.append(f"- {sec}: {'✅' if present else '❌'}")
        lines.append("")
    
    # Variable analysis
    if 'variables' in result:
        lines.append("## Variable Analysis")
        lines.append(f"- **Total Variables**: {result.get('variable_count', 0)}")
        
        type_dist = result.get('type_distribution', {})
        if type_dist:
            lines.append("- **Type Distribution**:")
            for var_type, count in type_dist.items():
                lines.append(f"  - {var_type}: {count}")
        
        dim_analysis = result.get('dimension_analysis', {})
        if dim_analysis:
            lines.append("- **Dimension Analysis**:")
            lines.append(f"  - Scalars: {dim_analysis.get('scalar_vars', 0)}")
            lines.append(f"  - Vectors: {dim_analysis.get('vector_vars', 0)}")
            lines.append(f"  - Matrices: {dim_analysis.get('matrix_vars', 0)}")
            lines.append(f"  - Tensors: {dim_analysis.get('tensor_vars', 0)}")
            lines.append(f"  - Max Dimensions: {dim_analysis.get('max_dimensions', 0)}")
        
        lines.append("\n### Variables Table:")
        lines.append("| Name | Type | Dimensions | Elements |")
        lines.append("|---|---|---|---|")
        for v in result['variables']:
            lines.append(f"| {v['name']} | {v['type']} | {v['dimensions']} | {v.get('total_elements', 'N/A')} |")
        lines.append("")
    
    # Connection analysis
    if 'connections' in result:
        lines.append("## Connection Analysis")
        lines.append(f"- **Total Connections**: {result.get('connection_count', 0)}")
        
        conn_types = result.get('connection_types', {})
        if conn_types:
            lines.append("- **Connection Types**:")
            lines.append(f"  - Directed: {conn_types.get('directed', 0)}")
            lines.append(f"  - Undirected: {conn_types.get('undirected', 0)}")
            lines.append(f"  - Temporal: {conn_types.get('temporal', 0)}")
        
        lines.append("\n### Connections:")
        lines.append("| Source | Target | Type | Temporal |")
        lines.append("|---|---|---|---|")
        for conn in result['connections']:
            lines.append(f"| {conn['source']} | {conn['target']} | {conn['type']} | {'Yes' if conn['is_temporal'] else 'No'} |")
        lines.append("")
    
    # Complexity analysis
    if 'model_complexity' in result:
        lines.append("## Complexity Analysis")
        complexity = result['model_complexity']
        lines.append(f"- **Variable Complexity**: {complexity.get('variable_complexity', 0)}")
        lines.append(f"- **Connection Complexity**: {complexity.get('connection_complexity', 0)}")
        lines.append(f"- **Equation Complexity**: {complexity.get('equation_complexity', 0)}")
        lines.append(f"- **Overall Complexity Score**: {complexity.get('overall_complexity', 0):.2f}")
        lines.append("")
    
    return '\n'.join(lines)

def per_file_json_report(filename: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable dict for a single file's type check result."""
    return result

def summary_markdown_report(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Generate a Markdown summary for all files."""
    lines = ["# GNN Type Check Summary\n"]
    
    # Overall statistics
    total_files = len(all_results)
    valid_files = sum(1 for r in all_results.values() if r.get('is_valid', False))
    invalid_files = total_files - valid_files
    
    lines.append("## Overall Statistics")
    lines.append(f"- **Total Files**: {total_files}")
    lines.append(f"- **Valid Files**: {valid_files}")
    lines.append(f"- **Invalid Files**: {invalid_files}")
    lines.append(f"- **Success Rate**: {(valid_files/total_files*100):.1f}%\n")
    
    # Complexity statistics
    complexities = [r.get('model_complexity', {}).get('overall_complexity', 0) 
                   for r in all_results.values()]
    if complexities:
        avg_complexity = sum(complexities) / len(complexities)
        max_complexity = max(complexities)
        min_complexity = min(complexities)
        lines.append("## Complexity Statistics")
        lines.append(f"- **Average Complexity**: {avg_complexity:.2f}")
        lines.append(f"- **Max Complexity**: {max_complexity:.2f}")
        lines.append(f"- **Min Complexity**: {min_complexity:.2f}\n")
    
    # File details table
    lines.append("## File Details")
    lines.append("| File | Status | Errors | Warnings | Variables | Connections | Complexity | Model Type |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for fname, res in all_results.items():
        status = '✅' if res.get('is_valid', False) else '❌'
        n_err = len(res.get('errors', []))
        n_warn = len(res.get('warnings', []))
        n_vars = res.get('variable_count', 0)
        n_conn = res.get('connection_count', 0)
        complexity = res.get('model_complexity', {}).get('overall_complexity', 0)
        model_type = res.get('model_type', 'Unknown')
        lines.append(f"| {os.path.basename(fname)} | {status} | {n_err} | {n_warn} | {n_vars} | {n_conn} | {complexity:.2f} | {model_type} |")
    
    return '\n'.join(lines)

def summary_json_report(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a JSON-serializable summary for all files."""
    summary = {
        'files_checked': len(all_results),
        'valid': sum(1 for r in all_results.values() if r['is_valid']),
        'invalid': sum(1 for r in all_results.values() if not r['is_valid']),
        'files': all_results
    }
    return summary

def variables_table_csv(all_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    """Return rows for a CSV table of all variables across all files."""
    rows = []
    for fname, res in all_results.items():
        for v in res.get('variables', []):
            rows.append([os.path.basename(fname), v['name'], v['type'], v['dimensions']])
    return rows

def section_presence_matrix_csv(all_results: Dict[str, Dict[str, Any]], section_list: List[str]) -> List[List[Any]]:
    """Return rows for a CSV matrix of section presence (files × sections)."""
    header = ['File'] + section_list
    rows = [header]
    for fname, res in all_results.items():
        row = [os.path.basename(fname)]
        for sec in section_list:
            row.append(1 if res.get('sections', {}).get(sec, False) else 0)
        rows.append(row)
    return rows 

def connections_table_csv(all_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    """Return rows for a CSV table of all connections across all files."""
    rows = []
    for fname, res in all_results.items():
        for conn in res.get('connections', []):
            rows.append([
                os.path.basename(fname),
                conn['source'],
                conn['target'],
                conn['type'],
                'Yes' if conn['is_temporal'] else 'No'
            ])
    return rows

def complexity_analysis_csv(all_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    """Return rows for a CSV table of complexity analysis across all files."""
    rows = []
    for fname, res in all_results.items():
        complexity = res.get('model_complexity', {}).get('model_complexity', {})
        rows.append([
            os.path.basename(fname),
            res.get('variable_count', 0),
            res.get('connection_count', 0),
            complexity.get('equation_complexity', 0),
            complexity.get('overall_complexity', 0),
            res.get('model_complexity', {}).get('model_type', 'Unknown'),
            res.get('model_complexity', {}).get('time_dynamics', {}).get('is_dynamic', False)
        ])
    return rows

def type_distribution_csv(all_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    """Return rows for a CSV table of type distribution across all files."""
    rows = []
    for fname, res in all_results.items():
        type_dist = res.get('type_distribution', {})
        for var_type, count in type_dist.items():
            rows.append([os.path.basename(fname), var_type, count])
    return rows 