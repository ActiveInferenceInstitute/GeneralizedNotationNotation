#!/usr/bin/env python3
"""
File Processing Utilities

This module provides utility functions for file processing and analysis
that can be used across different pipeline steps.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import datetime


def detect_content_type(content: str) -> str:
    """Detect the type of content in the file."""
    if content.strip().startswith('{') or content.strip().startswith('['):
        return 'json'
    elif '---' in content[:100] and '\n' in content[:100]:
        return 'yaml'
    elif content.strip().startswith('#'):
        return 'markdown'
    elif any(keyword in content.lower() for keyword in ['gnn', 'statespaceblock', 'connections']):
        return 'gnn_specification'
    else:
        return 'text'


def extract_markdown_metadata(content: str) -> dict:
    """Extract metadata from markdown content."""
    metadata = {}
    lines = content.splitlines()
    
    # Extract headers
    headers = [line.strip('# ') for line in lines if line.strip().startswith('#')]
    metadata['headers'] = headers[:10]  # First 10 headers
    
    # Extract code blocks
    code_blocks = []
    in_code_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                code_blocks.append('\n'.join(current_block))
                current_block = []
            in_code_block = not in_code_block
        elif in_code_block:
            current_block.append(line)
    
    metadata['code_blocks_count'] = len(code_blocks)
    metadata['total_lines'] = len(lines)
    
    return metadata


def extract_structured_metadata(content: str, file_type: str) -> dict:
    """Extract metadata from structured files (JSON, YAML)."""
    try:
        if file_type.lower() == '.json':
            data = json.loads(content)
        else:  # YAML
            data = yaml.safe_load(content)
        
        metadata = {
            'structure_type': 'structured',
            'top_level_keys': list(data.keys()) if isinstance(data, dict) else ['array'],
            'data_type': type(data).__name__,
            'is_valid': True
        }
        
        if isinstance(data, dict):
            metadata['key_count'] = len(data)
            metadata['nested_structure'] = analyze_nested_structure(data)
        
        return metadata
    except Exception as e:
        return {
            'structure_type': 'structured',
            'is_valid': False,
            'error': str(e)
        }


def extract_generic_metadata(content: str) -> dict:
    """Extract metadata from generic text content."""
    lines = content.splitlines()
    words = content.split()
    
    return {
        'structure_type': 'text',
        'line_count': len(lines),
        'word_count': len(words),
        'character_count': len(content),
        'non_empty_lines': len([line for line in lines if line.strip()]),
        'average_line_length': sum(len(line) for line in lines) / max(len(lines), 1)
    }


def analyze_nested_structure(data: dict, max_depth: int = 3) -> dict:
    """Analyze the nested structure of a dictionary."""
    def _analyze_level(obj, depth=0):
        if depth >= max_depth:
            return {'type': 'max_depth_reached'}
        
        if isinstance(obj, dict):
            return {
                'type': 'dict',
                'keys': list(obj.keys()),
                'key_count': len(obj),
                'sample_values': {k: type(v).__name__ for k, v in list(obj.items())[:5]}
            }
        elif isinstance(obj, list):
            return {
                'type': 'list',
                'length': len(obj),
                'sample_types': [type(item).__name__ for item in obj[:5]]
            }
        else:
            return {'type': type(obj).__name__}
    
    return _analyze_level(data)


def generate_processed_content(content: str, analysis: dict) -> str:
    """Generate processed content with annotations."""
    processed_lines = []
    
    # Add processing header
    processed_lines.append(f"# Processed File: {analysis['file_name']}")
    processed_lines.append(f"# Processing Timestamp: {analysis['processing_timestamp']}")
    processed_lines.append(f"# Content Type: {analysis['content_type']}")
    processed_lines.append(f"# File Size: {analysis['file_size_bytes']} bytes, {analysis['file_size_lines']} lines")
    processed_lines.append("")
    
    # Add original content with line numbers
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        processed_lines.append(f"{i:4d}: {line}")
    
    return '\n'.join(processed_lines)


def generate_summary_report(analysis: dict) -> str:
    """Generate a comprehensive summary report in Markdown format."""
    report_lines = []
    
    report_lines.append(f"# File Processing Summary")
    report_lines.append("")
    report_lines.append(f"**File:** {analysis['file_name']}")
    report_lines.append(f"**Processed:** {analysis['processing_timestamp']}")
    report_lines.append(f"**Content Type:** {analysis['content_type']}")
    report_lines.append("")
    
    report_lines.append("## File Statistics")
    report_lines.append(f"- **Size:** {analysis['file_size_bytes']} bytes")
    report_lines.append(f"- **Lines:** {analysis['file_size_lines']}")
    report_lines.append(f"- **Extension:** {analysis['file_extension']}")
    report_lines.append("")
    
    if 'metadata' in analysis:
        metadata = analysis['metadata']
        report_lines.append("## Content Analysis")
        
        if metadata.get('structure_type') == 'structured':
            report_lines.append(f"- **Structure:** {metadata.get('structure_type', 'Unknown')}")
            report_lines.append(f"- **Valid:** {metadata.get('is_valid', 'Unknown')}")
            if 'top_level_keys' in metadata:
                report_lines.append(f"- **Top-level Keys:** {', '.join(metadata['top_level_keys'][:10])}")
        elif metadata.get('structure_type') == 'text':
            report_lines.append(f"- **Words:** {metadata.get('word_count', 0)}")
            report_lines.append(f"- **Characters:** {metadata.get('character_count', 0)}")
            report_lines.append(f"- **Non-empty Lines:** {metadata.get('non_empty_lines', 0)}")
            report_lines.append(f"- **Average Line Length:** {metadata.get('average_line_length', 0):.1f}")
        elif 'headers' in metadata:
            report_lines.append(f"- **Headers Found:** {len(metadata.get('headers', []))}")
            report_lines.append(f"- **Code Blocks:** {metadata.get('code_blocks_count', 0)}")
    
    report_lines.append("")
    report_lines.append("## Processing Options")
    for key, value in analysis.get('processing_options', {}).items():
        report_lines.append(f"- **{key}:** {value}")
    
    return '\n'.join(report_lines)


def analyze_file_content(file_path: Path, options: dict = None) -> dict:
    """
    Perform comprehensive file analysis.
    
    Args:
        file_path: Path to the file to analyze
        options: Analysis options
        
    Returns:
        Dictionary containing analysis results
    """
    if options is None:
        options = {}
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Perform comprehensive file analysis
        analysis_results = {
            'file_name': file_path.name,
            'file_size_bytes': len(content),
            'file_size_lines': len(content.splitlines()),
            'file_extension': file_path.suffix,
            'content_type': detect_content_type(content),
            'processing_timestamp': datetime.datetime.now().isoformat(),
            'processing_options': options
        }
        
        # Extract metadata based on file type
        if file_path.suffix.lower() == '.md':
            analysis_results['metadata'] = extract_markdown_metadata(content)
        elif file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
            analysis_results['metadata'] = extract_structured_metadata(content, file_path.suffix)
        else:
            analysis_results['metadata'] = extract_generic_metadata(content)
        
        return analysis_results
        
    except Exception as e:
        return {
            'file_name': file_path.name,
            'error': str(e),
            'processing_timestamp': datetime.datetime.now().isoformat()
        }
