#!/usr/bin/env python3
"""
GNN Type Checker CLI

Command-line interface for the GNN type checker.
"""

import sys
import argparse
import json
from pathlib import Path
import logging

from .checker import GNNTypeChecker
from .resource_estimator import GNNResourceEstimator

logger = logging.getLogger(__name__)

def main(cmd_args=None):
    """
    Main function to run the type checker from command line.
    
    Args:
        cmd_args: Command line arguments (if None, sys.argv[1:] is used)
        
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parser = argparse.ArgumentParser(description="GNN Type Checker")
    parser.add_argument("input_path", help="Path to GNN file or directory")
    parser.add_argument("-r", "--report-file", default="type_check_report.md",
                        help="Filename for the main type checking report (markdown). Default: type_check_report.md")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--strict", action="store_true", help="Enable strict type checking mode")
    parser.add_argument("--estimate-resources", action="store_true", help="Estimate computational resources for the GNN models")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Base directory to save all output files (type checker reports, resource reports, etc.).")
    parser.add_argument("--project-root", help="Absolute path to the project root, for relative path generation in reports")
    
    parsed_args = parser.parse_args(cmd_args)
    
    # The caller (4_gnn_type_checker.py or user via CLI) is responsible for configuring logging levels.
    # This script just uses the logger.

    actual_output_dir = Path(parsed_args.output_dir).resolve()
    try:
        actual_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {actual_output_dir}: {e}")
        return 1 # Critical failure

    markdown_report_name = Path(parsed_args.report_file).name # Use only the filename part

    logger.info(f"GNN Type Checker CLI starting...")
    logger.info(f"  Input path: {parsed_args.input_path}")
    logger.info(f"  Output directory: {actual_output_dir}")
    logger.info(f"  Main report filename: {markdown_report_name}")
    logger.debug(f"  Recursive: {parsed_args.recursive}, Strict: {parsed_args.strict}, Estimate Resources: {parsed_args.estimate_resources}")
    if parsed_args.project_root:
        logger.debug(f"  Project root for relative paths: {parsed_args.project_root}")

    checker = GNNTypeChecker(strict_mode=parsed_args.strict)
    input_path_obj = Path(parsed_args.input_path)
    results = {}
    
    try:
        if input_path_obj.is_file():
            is_valid, errors, warnings = checker.check_file(str(input_path_obj))
            results = {str(input_path_obj): {"is_valid": is_valid, "errors": errors, "warnings": warnings}}
        elif input_path_obj.is_dir():
            results = checker.check_directory(str(input_path_obj), recursive=parsed_args.recursive)
        else:
            logger.error(f"Input path {parsed_args.input_path} is not a valid file or directory.")
            return 1
    except Exception as e_check:
        logger.error(f"An error occurred during GNN checking phase: {e_check}", exc_info=True)
        return 1

    try:
        report_summary_text = checker.generate_report(results, actual_output_dir, report_md_filename=markdown_report_name, project_root_path=parsed_args.project_root)
        logger.info("\n--- Type Check Report Summary ---")
        for line in report_summary_text.splitlines(): # Log line by line to respect logger formatting
            logger.info(line)
        logger.info("--- End of Type Check Report Summary ---")
        logger.info(f"Main type check report saved in: {actual_output_dir / markdown_report_name}")
        
        # Generate and save the detailed JSON data
        json_output_dir = actual_output_dir / "resources"
        json_output_dir.mkdir(parents=True, exist_ok=True)
        type_check_data_json_path = json_output_dir / "type_check_data.json"
        checker.generate_json_data(results, type_check_data_json_path) # Call the new public method
        logger.info(f"Detailed JSON data saved in: {type_check_data_json_path}")

    except Exception as e_report:
        logger.error(f"An error occurred during report generation for type checking: {e_report}", exc_info=True)
        # Decide if this is fatal; type checking might have finished.
        # For now, let's say if report fails, the step has an issue.
        # The `results` are still available if an error happened here.

    if parsed_args.estimate_resources:
        logger.info("\nEstimating computational resources...")
        resource_estimator_output_base_dir = actual_output_dir / "resource_estimates"
        try:
            resource_estimator_output_base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e_res_dir:
            logger.error(f"Failed to create resource estimates output directory {resource_estimator_output_base_dir}: {e_res_dir}")
            # Don't fail the whole script if only this sub-dir creation fails, but log it.

        type_check_data_json_path = actual_output_dir / "resources" / "type_check_data.json"
        estimator = GNNResourceEstimator(type_check_data=str(type_check_data_json_path) if type_check_data_json_path.exists() else None)
        
        try:
            if input_path_obj.is_file():
                result = estimator.estimate_from_file(str(input_path_obj))
                estimator.results = {str(input_path_obj): result} # Store for report generation
            elif input_path_obj.is_dir():
                estimator.estimate_from_directory(str(input_path_obj), recursive=parsed_args.recursive)
            # else case already handled above for main checker

            resource_report_summary_text = estimator.generate_report(str(resource_estimator_output_base_dir), project_root_path=parsed_args.project_root)
            logger.info("\n--- Resource Estimation Report Summary ---")
            summary_lines = resource_report_summary_text.split('\n')[:5] # Show first 5 lines
            for r_line in summary_lines:
                 logger.info(r_line)
            logger.info(f"Resource estimation reports (markdown, JSON, HTML) saved in: {resource_estimator_output_base_dir}")
            logger.info("--- End of Resource Estimation Report Summary ---")
        except Exception as e_est:
            logger.error(f"An error occurred during resource estimation: {e_est}", exc_info=True)
            # Resource estimation failure doesn't necessarily mean the type check failed.
            # The overall exit code depends on type check errors.

    # Determine final exit code based on type checking results
    has_type_errors = any(not r.get("is_valid", True) for r in results.values())
    if has_type_errors:
        logger.error("Type checking found errors in one or more GNN files.")
        return 1
    else:
        logger.info("Type checking completed. No errors found.")
        return 0

if __name__ == "__main__":
    # When run directly as `python -m gnn_type_checker.cli ...` or `python src/gnn_type_checker/cli.py ...`,
    # this block executes. It should set up basic logging if no other configuration exists.
    if not logging.getLogger().hasHandlers():
        # BasicConfig for the root logger. Any loggers created by this module will inherit.
        # Determine verbosity from args (if any includes a verbose flag) or default.
        # For simplicity, as this cli.py doesn't have its own --verbose, default to INFO.
        # A more advanced direct run might parse a --verbose here too.
        logging.basicConfig(level=logging.INFO, 
                            format="%Y-%m-%d %H:%M:%S - %(name)s - %(levelname)s - %(message)s", 
                            stream=sys.stdout)
        logger.info("CLI executed directly: Basic logging configured to INFO level.")

    sys.exit(main()) # cmd_args will be None, so argparse uses sys.argv[1:] 