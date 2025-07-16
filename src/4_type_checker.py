#!/usr/bin/env python3

"""
Pipeline Step 4: Type Checker

All outputs from this step must go under output/type_check/ and its subfolders.
The type checker CLI now enforces this policy and will refuse to run if --output-dir is not a subdirectory named 'type_check'.
"""

import sys
import logging
from pathlib import Path

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

from type_checker.checker import run_type_checking
from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step
logger = setup_step_logging("4_type_checker", verbose=False)

def process_type_checking_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized type checking processing function.
    
    Args:
        target_dir: Directory containing GNN files to type check
        output_dir: Output directory for type checking results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options (strict, estimate_resources, etc.)
        
    Returns:
        True if type checking succeeded, False otherwise
    """
    try:
        # Build CLI argument list for type_checker.cli.main()
        cli_args = []
        # Required positional argument for CLI: input_path
        cli_args.append(str(target_dir))
        
        # Output directory - ensure it's the step-specific type_check directory
        if UTILS_AVAILABLE:
            from utils.argument_utils import get_step_output_dir
            step_output_dir = get_step_output_dir("4_type_checker", output_dir)
        else:
            # Fallback: append type_check to the output directory
            step_output_dir = output_dir / "type_check"
        
        cli_args.extend(["-o", str(step_output_dir)])
        
        # Main report filename (optional, default is fine)
        # Recursive
        if recursive:
            cli_args.append("--recursive")
        # Strict mode
        if kwargs.get('strict', False):
            cli_args.append("--strict")
        # Estimate resources
        if kwargs.get('estimate_resources', False):
            cli_args.append("--estimate-resources")
        # Verbosity (handled by logger, but could be passed if CLI supports it)
        # Project root (optional, not always available)
        if 'project_root' in kwargs and kwargs['project_root']:
            cli_args.extend(["--project-root", str(kwargs['project_root'])])
        
        # Call the type checking function
        logger.info(f"Invoking type_checker with args: {cli_args}")
        exit_code = run_type_checking(target_dir, output_dir, logger, recursive, kwargs.get('strict', False))
        
        return exit_code == 0
        
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "4_type_checker.py",
    process_type_checking_standardized,
    "GNN syntax and type validation",
    additional_arguments={
        "strict": {"type": bool, "default": False, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "default": True, "help": "Estimate computational resources"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script())