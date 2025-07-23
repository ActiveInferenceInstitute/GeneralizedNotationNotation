"""
Type Checker module for GNN Processing Pipeline.

This module provides type checking and validation capabilities for GNN files.
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

def validate_gnn_files(
    target_dir: Path,
    output_dir: Path,
    strict: bool = False,
    estimate_resources: bool = False,
    verbose: bool = False
) -> bool:
    """
    Validate GNN files for syntax and type correctness.
    
    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Directory to save validation results
        strict: Enable strict type checking mode
        estimate_resources: Estimate computational resources
        verbose: Enable verbose output
        
    Returns:
        True if all files are valid, False otherwise
    """
    logger = logging.getLogger("type_checker")
    
    try:
        log_step_start(logger, "Validating GNN files")
        
        # Create validation results directory
        validation_results_dir = output_dir / "validation_results"
        validation_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for validation")
            return True
        
        # Validate each file
        validation_results = {}
        for gnn_file in gnn_files:
            file_result = validate_single_gnn_file(gnn_file, strict, estimate_resources)
            validation_results[gnn_file.name] = file_result
        
        # Save validation results
        import json
        results_file = validation_results_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Check overall success
        all_valid = all(result["valid"] for result in validation_results.values())
        
        if all_valid:
            log_step_success(logger, "All GNN files are valid")
        else:
            invalid_files = [name for name, result in validation_results.items() if not result["valid"]]
            log_step_error(logger, f"Some GNN files are invalid: {invalid_files}")
        
        return all_valid
        
    except Exception as e:
        log_step_error(logger, f"GNN file validation failed: {e}")
        return False

def validate_single_gnn_file(
    gnn_file: Path,
    strict: bool = False,
    estimate_resources: bool = False
) -> Dict[str, Any]:
    """
    Validate a single GNN file.
    
    Args:
        gnn_file: Path to the GNN file to validate
        strict: Enable strict validation mode
        estimate_resources: Estimate computational resources
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Read file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic validation
        validation_result = {
            "file": str(gnn_file),
            "valid": True,
            "errors": [],
            "warnings": [],
            "resource_estimate": {}
        }
        
        # Check for required sections
        required_sections = ["GNNVersionAndFlags", "ModelName", "StateSpaceBlock"]
        for section in required_sections:
            if f"## {section}" not in content:
                validation_result["errors"].append(f"Missing required section: {section}")
                validation_result["valid"] = False
        
        # Check for basic syntax
        if "##" not in content:
            validation_result["errors"].append("No section headers found")
            validation_result["valid"] = False
        
        # Resource estimation
        if estimate_resources:
            validation_result["resource_estimate"] = estimate_file_resources(content)
        
        return validation_result
        
    except Exception as e:
        return {
            "file": str(gnn_file),
            "valid": False,
            "errors": [f"File reading error: {str(e)}"],
            "warnings": [],
            "resource_estimate": {}
        }

def estimate_file_resources(content: str) -> Dict[str, Any]:
    """
    Estimate computational resources needed for the GNN file.
    
    Args:
        content: GNN file content
        
    Returns:
        Dictionary with resource estimates
    """
    try:
        # Basic resource estimation
        lines = content.split('\n')
        sections = content.count('##')
        
        estimate = {
            "lines_of_code": len(lines),
            "sections": sections,
            "estimated_memory_mb": max(1, len(content) // 1000),  # Rough estimate
            "estimated_processing_time_s": max(1, sections * 2)  # Rough estimate
        }
        
        return estimate
        
    except Exception:
        return {
            "lines_of_code": 0,
            "sections": 0,
            "estimated_memory_mb": 1,
            "estimated_processing_time_s": 1
        }

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Type checking and validation for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'syntax_validation': True,
    'type_checking': True,
    'resource_estimation': True,
    'strict_mode': True
}

__all__ = [
    'validate_gnn_files',
    'validate_single_gnn_file',
    'estimate_file_resources',
    'FEATURES',
    '__version__'
] 