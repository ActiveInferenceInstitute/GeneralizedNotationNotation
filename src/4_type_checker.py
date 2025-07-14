#!/usr/bin/env python3

"""
GNN Processing Pipeline - Step 4: Type Checker

This script validates GNN files for syntax correctness and type consistency.

Usage:
    python 4_type_checker.py [options]
    (Typically called by main.py)
"""

import sys
from pathlib import Path
import argparse
import logging

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

# Initialize logger for this step
logger = setup_step_logging("4_type_checker", verbose=False)

# Import type checker functionality
try:
    from type_checker import checker
    from type_checker.resource_estimator import GNNResourceEstimator as ResourceEstimator
    logger.debug("Successfully imported type checker modules")
except ImportError as e:
    log_step_error(logger, f"Could not import type checker: {e}")
    checker = None
    ResourceEstimator = None

def run_type_checking(target_dir: Path, output_dir: Path, recursive: bool = False, 
                     strict: bool = False, estimate_resources: bool = False):
    """Run type checking and validation."""
    log_step_start(logger, f"Running type checking on: {target_dir}")
    
    if not checker:
        log_step_error(logger, "Type checker not available")
        return False
    
    # Use centralized output directory configuration
    type_check_output_dir = get_output_dir_for_script("4_type_checker.py", output_dir)
    
    try:
        # Create type checker instance
        type_checker = checker.GNNTypeChecker(strict_mode=strict)
        
        # Run type checking
        results = type_checker.check_directory(
            dir_path=str(target_dir),
            recursive=recursive
        )
        
        # Run resource estimation if requested
        if estimate_resources and ResourceEstimator:
            try:
                estimator = ResourceEstimator()
                resource_results = estimator.estimate_from_directory(
                    dir_path=str(target_dir),
                    recursive=recursive
                )
                logger.debug(f"Resource estimation completed for {len(resource_results)} files")
                log_step_success(logger, "Resource estimation completed")
            except Exception as e:
                log_step_warning(logger, f"Resource estimation failed: {e}")
        
        # Generate reports
        try:
            report_path = type_checker.generate_report(
                results=results,
                output_dir_base=type_check_output_dir,
                project_root_path=target_dir
            )
            logger.info(f"Type checking report generated: {report_path}")
        except Exception as e:
            log_step_warning(logger, f"Report generation failed: {e}")
        
        # Log results summary
        files_checked = len(results)
        valid_files = sum(1 for result in results.values() if result.get('is_valid', False))
        
        if valid_files == files_checked:
            log_step_success(logger, f"Type checking completed successfully. Files checked: {files_checked}, All valid: {valid_files}")
            return True
        else:
            log_step_warning(logger, f"Type checking completed with issues. Files checked: {files_checked}, Valid: {valid_files}")
            return files_checked > 0  # Consider success if we processed files, even with issues
        
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for type checking."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("4_type_checker.py", {})
    log_step_start(logger, f"{step_info.get('description', 'GNN syntax and type validation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run type checking
    success = run_type_checking(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        recursive=getattr(parsed_args, 'recursive', False),
        strict=getattr(parsed_args, 'strict', False),
        estimate_resources=getattr(parsed_args, 'estimate_resources', False)
    )
    
    if success:
        log_step_success(logger, "Type checking completed successfully")
        return 0
    else:
        log_step_error(logger, "Type checking failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("4_type_checker")
    else:
        # Fallback argument parsing
        parser = argparse.ArgumentParser(description="GNN syntax and type validation")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--strict", action="store_true",
                          help="Enable strict validation mode")
        parser.add_argument("--estimate-resources", action="store_true",
                          help="Estimate computational resources")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 