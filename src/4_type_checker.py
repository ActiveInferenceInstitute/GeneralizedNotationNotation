#!/usr/bin/env python3

"""
Pipeline Step 4: Type Checker

All outputs from this step must go under output/type_check/ and its subfolders.
The type checker CLI now enforces this policy and will refuse to run if --output-dir is not a subdirectory named 'type_check'.
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

# Import the CLI entry point from type_checker
try:
    from type_checker import cli as type_checker_cli
    logger.debug("Successfully imported type_checker.cli")
except ImportError as e:
    log_step_error(logger, f"Could not import type_checker.cli: {e}")
    sys.exit(1)

def main():
    """Main function for type checking using the CLI entry point."""
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("4_type_checker.py", {})
    log_step_start(logger, f"{step_info.get('description', 'GNN syntax and type validation')}")

    # Use centralized argument parsing
    # NOTE: All outputs must go under output/type_check/ (enforced by CLI)
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

    # Build CLI argument list for type_checker.cli.main()
    cli_args = []
    # Required positional argument for CLI: input_path
    cli_args.append(str(parsed_args.target_dir))
    # Output directory
    cli_args.extend(["-o", str(parsed_args.output_dir)])
    # Main report filename (optional, default is fine)
    # Recursive
    if getattr(parsed_args, 'recursive', False):
        cli_args.append("--recursive")
    # Strict mode
    if getattr(parsed_args, 'strict', False):
        cli_args.append("--strict")
    # Estimate resources
    if getattr(parsed_args, 'estimate_resources', False):
        cli_args.append("--estimate-resources")
    # Verbosity (handled by logger, but could be passed if CLI supports it)
    # Project root (optional, not always available)
    if hasattr(parsed_args, 'project_root') and parsed_args.project_root:
        cli_args.extend(["--project-root", str(parsed_args.project_root)])

    # Call the CLI main function
    logger.info(f"Invoking type_checker.cli.main with args: {cli_args}")
    exit_code = type_checker_cli.main(cli_args)
    if exit_code == 0:
        log_step_success(logger, "Type checking completed successfully (CLI)")
    else:
        log_step_error(logger, "Type checking failed (CLI)")
    sys.exit(exit_code)

if __name__ == '__main__':
    main() 