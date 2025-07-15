"""
Shared Functions for Pipeline Modules

This module contains common functions used across multiple pipeline modules
to reduce code duplication and ensure consistency.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from . import log_step_start, log_step_success, log_step_warning, log_step_error

def find_gnn_files(target_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Find GNN files in the target directory.
    
    Args:
        target_dir: Directory to search for GNN files
        recursive: Whether to search recursively
        
    Returns:
        List of paths to GNN files
    """
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    # Filter for actual GNN files (containing GNN-specific content)
    filtered_files = []
    for file_path in gnn_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for GNN-specific sections
                if any(section in content for section in ['ModelName', 'StateSpaceBlock', 'Connections']):
                    filtered_files.append(file_path)
        except Exception:
            continue
    
    return filtered_files

def parse_gnn_sections(content: str) -> Dict[str, Any]:
    """
    Parse common GNN sections from file content.
    
    Args:
        content: Raw file content
        
    Returns:
        Dictionary with parsed sections
    """
    sections = {}
    
    # Common GNN section headers
    section_headers = [
        'ModelName', 'ModelAnnotation', 'StateSpaceBlock', 'Connections',
        'InitialParameterization', 'Equations', 'Time', 'ActInfOntologyAnnotation'
    ]
    
    for header in section_headers:
        pattern = rf"^##\s*{re.escape(header)}\s*$"
        match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
        
        if match:
            # Extract content between this header and the next header
            start_pos = match.end()
            next_header_match = re.search(r"^##\s+\w+", content[start_pos:], re.MULTILINE)
            
            if next_header_match:
                section_content = content[start_pos:start_pos + next_header_match.start()]
            else:
                section_content = content[start_pos:]
            
            sections[header] = section_content.strip()
    
    return sections

def extract_model_parameters(content: str) -> Dict[str, Any]:
    """
    Extract model parameters from GNN content.
    
    Args:
        content: GNN file content
        
    Returns:
        Dictionary of parameter names and values
    """
    parameters = {}
    
    # Look for ModelParameters section
    pattern = r"^##\s*ModelParameters\s*$"
    match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
    
    if match:
        start_pos = match.end()
        next_header_match = re.search(r"^##\s+\w+", content[start_pos:], re.MULTILINE)
        
        if next_header_match:
            param_section = content[start_pos:start_pos + next_header_match.start()]
        else:
            param_section = content[start_pos:]
        
        # Parse parameter assignments
        param_pattern = r"^\s*(\w+)\s*=\s*(.+?)(?:\s*###.*)?$"
        for line in param_section.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                param_match = re.match(param_pattern, line)
                if param_match:
                    param_name = param_match.group(1)
                    param_value_str = param_match.group(2).strip()
                    
                    # Try to parse the value
                    try:
                        import ast
                        param_value = ast.literal_eval(param_value_str)
                    except (ValueError, SyntaxError):
                        param_value = param_value_str
                    
                    parameters[param_name] = param_value
    
    return parameters

def create_processing_report(
    step_name: str,
    target_dir: Path,
    output_dir: Path,
    processed_files: List[Path],
    errors: List[str],
    warnings: List[str],
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized processing report.
    
    Args:
        step_name: Name of the processing step
        target_dir: Target directory that was processed
        output_dir: Output directory where results were saved
        processed_files: List of files that were successfully processed
        errors: List of error messages
        warnings: List of warning messages
        additional_info: Additional information to include in the report
        
    Returns:
        Dictionary containing the processing report
    """
    report = {
        "step_name": step_name,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "target_directory": str(target_dir),
        "output_directory": str(output_dir),
        "files_processed": len(processed_files),
        "files_with_errors": len(errors),
        "files_with_warnings": len(warnings),
        "success_rate": len(processed_files) / (len(processed_files) + len(errors)) * 100 if (len(processed_files) + len(errors)) > 0 else 0,
        "processed_files": [str(f) for f in processed_files],
        "errors": errors,
        "warnings": warnings
    }
    
    if additional_info:
        report.update(additional_info)
    
    return report

def save_processing_report(
    report: Dict[str, Any],
    output_dir: Path,
    filename: str = "processing_report.json"
) -> Path:
    """
    Save a processing report to a JSON file.
    
    Args:
        report: Processing report dictionary
        output_dir: Directory to save the report
        filename: Name of the report file
        
    Returns:
        Path to the saved report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_path

def validate_file_paths(*paths: Path) -> Tuple[bool, List[str]]:
    """
    Validate that file paths exist and are accessible.
    
    Args:
        *paths: Paths to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    for path in paths:
        if not path.exists():
            errors.append(f"Path does not exist: {path}")
        elif not os.access(path, os.R_OK):
            errors.append(f"Path is not readable: {path}")
    
    return len(errors) == 0, errors

def ensure_output_directory(output_dir: Path, logger: logging.Logger) -> bool:
    """
    Ensure output directory exists and is writable.
    
    Args:
        output_dir: Output directory to create/validate
        logger: Logger instance
        
    Returns:
        True if directory is ready, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = output_dir / ".test_write_access"
        test_file.write_text("test")
        test_file.unlink()
        
        return True
    except Exception as e:
        log_step_error(logger, f"Failed to create/validate output directory {output_dir}: {e}")
        return False

def log_processing_summary(
    logger: logging.Logger,
    step_name: str,
    total_files: int,
    successful_files: int,
    failed_files: int,
    warnings: int = 0
):
    """
    Log a standardized processing summary.
    
    Args:
        logger: Logger instance
        step_name: Name of the processing step
        total_files: Total number of files processed
        successful_files: Number of successfully processed files
        failed_files: Number of failed files
        warnings: Number of warnings
    """
    if total_files == 0:
        log_step_warning(logger, f"No files were processed in {step_name}")
    elif failed_files == 0:
        success_rate = (successful_files / total_files) * 100
        log_step_success(logger, f"{step_name} completed successfully: {successful_files}/{total_files} files ({success_rate:.1f}%)")
    elif successful_files > 0:
        success_rate = (successful_files / total_files) * 100
        log_step_warning(logger, f"{step_name} completed with issues: {successful_files}/{total_files} files successful ({success_rate:.1f}%)")
    else:
        log_step_error(logger, f"{step_name} failed: {failed_files}/{total_files} files failed")
    
    if warnings > 0:
        logger.warning(f"{warnings} warnings were generated during processing") 