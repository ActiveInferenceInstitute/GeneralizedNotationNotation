#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 6: Enhanced Validation

This step performs enhanced validation and quality assurance on GNN models,
including semantic validation, performance profiling, and consistency checking.

Usage:
    python 6_validation.py [options]
    (Typically called by main.py)
"""

import sys
import logging
from pathlib import Path
import json
import datetime
from typing import Dict, Any, List, Optional, Union

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    get_output_dir_for_script,
    get_pipeline_config
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("6_validation", verbose=False)

# Import step-specific modules
try:
    from validation.semantic_validator import SemanticValidator
    from validation.performance_profiler import PerformanceProfiler
    from validation.consistency_checker import ConsistencyChecker
    
    DEPENDENCIES_AVAILABLE = True
    logger.debug("Successfully imported validation dependencies")
    
except ImportError as e:
    log_step_warning(logger, f"Failed to import validation dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

def process_validation_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized validation processing function.
    
    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("validation_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_config = config.get_step_config("6_validation.py")
            
            # Set up output directory
            step_output_dir = get_output_dir_for_script("6_validation.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate step requirements
            if not DEPENDENCIES_AVAILABLE:
                log_step_warning(logger, "Validation dependencies are not available, functionality will be limited")
            
            # Log processing parameters
            logger.info(f"Processing GNN files from: {target_dir}")
            logger.info(f"Output directory: {step_output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Extract additional parameters from kwargs
            validation_level = kwargs.get('validation_level', 'standard')
            profile_performance = kwargs.get('profile_performance', True)
            check_consistency = kwargs.get('check_consistency', True)
            
            logger.info(f"Validation level: {validation_level}")
            logger.info(f"Performance profiling: {profile_performance}")
            logger.info(f"Consistency checking: {check_consistency}")
            
            # Validate input directory
            if not target_dir.exists():
                log_step_error(logger, f"Input directory does not exist: {target_dir}")
                return False
            
            # Find GNN files
            pattern = "**/*.md" if recursive else "*.md"
            gnn_files = list(target_dir.glob(pattern))
            
            if not gnn_files:
                log_step_warning(logger, f"No GNN files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            logger.info(f"Found {len(gnn_files)} GNN files to validate")
            
            # Process files
            successful_files = 0
            failed_files = 0
            warnings_count = 0
            
            # Initialize validators
            semantic_validator = SemanticValidator(validation_level) if DEPENDENCIES_AVAILABLE else None
            performance_profiler = PerformanceProfiler() if DEPENDENCIES_AVAILABLE and profile_performance else None
            consistency_checker = ConsistencyChecker() if DEPENDENCIES_AVAILABLE and check_consistency else None
            
            # Process each file
            validation_results = []
            for gnn_file in gnn_files:
                try:
                    # Read file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_result = {
                        "file_path": str(gnn_file),
                        "file_name": gnn_file.name,
                        "validation_level": validation_level,
                        "semantic_validation": {},
                        "performance_profile": {},
                        "consistency_check": {},
                        "warnings": [],
                        "errors": [],
                        "status": "unknown"
                    }
                    
                    # Semantic validation
                    if semantic_validator:
                        semantic_result = semantic_validator.validate(content)
                        file_result["semantic_validation"] = semantic_result
                        
                        if not semantic_result.get("is_valid", False):
                            file_result["errors"].extend(semantic_result.get("errors", []))
                    else:
                        # Fallback validation
                        semantic_result = validate_semantic_fallback(content)
                        file_result["semantic_validation"] = semantic_result
                        
                        if not semantic_result.get("is_valid", False):
                            file_result["errors"].extend(semantic_result.get("errors", []))
                    
                    # Performance profiling
                    if performance_profiler:
                        profile_result = performance_profiler.profile(content)
                        file_result["performance_profile"] = profile_result
                        
                        if profile_result.get("warnings", []):
                            file_result["warnings"].extend(profile_result.get("warnings", []))
                    else:
                        # Fallback profiling
                        profile_result = profile_performance_fallback(content)
                        file_result["performance_profile"] = profile_result
                    
                    # Consistency checking
                    if consistency_checker:
                        consistency_result = consistency_checker.check(content)
                        file_result["consistency_check"] = consistency_result
                        
                        if not consistency_result.get("is_consistent", False):
                            file_result["warnings"].extend(consistency_result.get("warnings", []))
                    else:
                        # Fallback consistency check
                        consistency_result = check_consistency_fallback(content)
                        file_result["consistency_check"] = consistency_result
                    
                    # Determine overall status
                    if file_result["errors"]:
                        file_result["status"] = "failed"
                        failed_files += 1
                    elif file_result["warnings"]:
                        file_result["status"] = "warnings"
                        successful_files += 1
                        warnings_count += 1
                    else:
                        file_result["status"] = "passed"
                        successful_files += 1
                    
                    # Save individual file result
                    file_output_dir = step_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_result_path = file_output_dir / f"{gnn_file.stem}_validation.json"
                    with open(file_result_path, 'w') as f:
                        json.dump(file_result, f, indent=2)
                    
                    validation_results.append(file_result)
                    
                except Exception as e:
                    log_step_error(logger, f"Unexpected error validating {gnn_file}: {e}")
                    failed_files += 1
            
            # Generate summary report
            summary_file = step_output_dir / "validation_summary.json"
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step_name": "validation",
                "input_directory": str(target_dir),
                "output_directory": str(step_output_dir),
                "total_files": len(gnn_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "files_with_warnings": warnings_count,
                "validation_level": validation_level,
                "profile_performance": profile_performance,
                "check_consistency": check_consistency,
                "performance_metrics": performance_tracker.get_summary()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Generate detailed report
            detailed_file = step_output_dir / "validation_detailed.json"
            with open(detailed_file, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            logger.info(f"Summary report saved: {summary_file}")
            logger.info(f"Detailed report saved: {detailed_file}")
            
            # Determine success
            if failed_files == 0:
                if warnings_count > 0:
                    log_step_warning(logger, f"Validation completed with warnings in {warnings_count} files")
                else:
                    log_step_success(logger, f"Successfully validated {successful_files} GNN models")
                return True
            elif successful_files > 0:
                log_step_warning(logger, f"Partially successful: {failed_files} files failed validation")
                return True  # Still consider successful for pipeline continuation
            else:
                log_step_error(logger, "All files failed validation")
                return False
            
    except Exception as e:
        log_step_error(logger, f"Validation processing failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def validate_semantic_fallback(content: str) -> Dict[str, Any]:
    """
    Fallback implementation for semantic validation when validator module is not available.
    
    Args:
        content: GNN file content
        
    Returns:
        Validation result with status and errors
    """
    import re
    
    # Simple validation rules
    errors = []
    warnings = []
    
    # Check for required elements
    if not re.search(r'StateSpaceBlock', content):
        errors.append("Missing StateSpaceBlock definition")
    
    if not re.search(r'ModelName', content):
        warnings.append("Missing ModelName definition")
    
    # Check for potential inconsistencies
    state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
    connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
    
    # Check for empty state blocks
    for i, block in enumerate(state_blocks):
        if not block.strip():
            errors.append(f"Empty StateSpaceBlock at index {i}")
    
    # Check for connections referencing non-existent blocks
    block_names = []
    for block in state_blocks:
        name_match = re.search(r'Name:\s*([^\n]+)', block)
        if name_match:
            block_names.append(name_match.group(1).strip())
    
    for i, conn in enumerate(connections):
        from_match = re.search(r'From:\s*([^\n]+)', conn)
        to_match = re.search(r'To:\s*([^\n]+)', conn)
        
        if from_match and from_match.group(1).strip() not in block_names:
            errors.append(f"Connection {i} references non-existent 'From' block: {from_match.group(1).strip()}")
        
        if to_match and to_match.group(1).strip() not in block_names:
            errors.append(f"Connection {i} references non-existent 'To' block: {to_match.group(1).strip()}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def profile_performance_fallback(content: str) -> Dict[str, Any]:
    """
    Fallback implementation for performance profiling when profiler module is not available.
    
    Args:
        content: GNN file content
        
    Returns:
        Performance profile with metrics and warnings
    """
    import re
    
    # Simple performance metrics
    metrics = {}
    warnings = []
    
    # Count state blocks
    state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
    metrics["state_block_count"] = len(state_blocks)
    
    # Count connections
    connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
    metrics["connection_count"] = len(connections)
    
    # Count dimensions
    dimensions = []
    for block in state_blocks:
        dim_match = re.search(r'Dimensions:\s*([^\n]+)', block)
        if dim_match:
            try:
                dims = [int(d.strip()) for d in dim_match.group(1).strip().split(',')]
                dimensions.extend(dims)
            except ValueError:
                pass
    
    metrics["total_dimensions"] = sum(dimensions) if dimensions else 0
    metrics["max_dimension"] = max(dimensions) if dimensions else 0
    
    # Simple performance warnings
    if metrics["state_block_count"] > 10:
        warnings.append(f"Large number of state blocks ({metrics['state_block_count']})")
    
    if metrics["connection_count"] > 20:
        warnings.append(f"Large number of connections ({metrics['connection_count']})")
    
    if metrics["max_dimension"] > 100:
        warnings.append(f"High-dimensional state space ({metrics['max_dimension']})")
    
    # Estimate memory usage
    estimated_memory_mb = (metrics["total_dimensions"] * 8 * 2) / (1024 * 1024)
    metrics["estimated_memory_mb"] = estimated_memory_mb
    
    if estimated_memory_mb > 100:
        warnings.append(f"High estimated memory usage ({estimated_memory_mb:.2f} MB)")
    
    return {
        "metrics": metrics,
        "warnings": warnings
    }

def check_consistency_fallback(content: str) -> Dict[str, Any]:
    """
    Fallback implementation for consistency checking when checker module is not available.
    
    Args:
        content: GNN file content
        
    Returns:
        Consistency check result with status and warnings
    """
    import re
    
    # Simple consistency checks
    warnings = []
    
    # Check for consistent naming conventions
    state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
    block_names = []
    
    for block in state_blocks:
        name_match = re.search(r'Name:\s*([^\n]+)', block)
        if name_match:
            block_names.append(name_match.group(1).strip())
    
    # Check for naming consistency
    camel_case = sum(1 for name in block_names if name and name[0].isupper() and '_' not in name)
    snake_case = sum(1 for name in block_names if '_' in name)
    
    if camel_case > 0 and snake_case > 0:
        warnings.append("Inconsistent naming conventions (mix of camelCase and snake_case)")
    
    # Check for duplicate names
    duplicate_names = set([name for name in block_names if block_names.count(name) > 1])
    if duplicate_names:
        warnings.append(f"Duplicate block names found: {', '.join(duplicate_names)}")
    
    # Check for consistent indentation
    lines = content.split('\n')
    indentation_patterns = set()
    
    for line in lines:
        if line.strip() and line.startswith(' '):
            indent = len(line) - len(line.lstrip(' '))
            indentation_patterns.add(indent)
    
    if len(indentation_patterns) > 2:
        warnings.append("Inconsistent indentation patterns")
    
    return {
        "is_consistent": len(warnings) == 0,
        "warnings": warnings
    }

# Create standardized pipeline script
run_script = create_standardized_pipeline_script(
    "6_validation.py",
    process_validation_standardized,
    "Enhanced validation and quality assurance",
    additional_arguments={
        "validation_level": {
            "type": str,
            "choices": ["basic", "standard", "strict", "research"],
            "default": "standard",
            "help": "Validation level for semantic validation"
        },
        "profile_performance": {"type": bool, "default": True, "help": "Enable performance profiling"},
        "check_consistency": {"type": bool, "default": True, "help": "Enable consistency checking"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 