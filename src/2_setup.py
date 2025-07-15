#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 2: Setup

This script performs initial setup tasks:
- Verifies and creates necessary output directories.
- Sets up the Python virtual environment and installs dependencies.

Usage:
    python 2_setup.py [options]
    (Typically called by main.py)
    
Options:
    Same as main.py (passes arguments through)
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

# Initialize logger for this step  
logger = setup_step_logging("2_setup", verbose=False)

# Import setup module for environment setup
try:
    from setup.setup import perform_full_setup
    from setup.utils import (
        ensure_directory, 
        find_gnn_files, 
        get_output_paths,
        is_venv_active, 
        get_venv_info, 
        check_system_dependencies,
        log_environment_info, 
        verify_directories, 
        list_installed_packages
    )
    logger.debug("Successfully imported setup modules")
except ImportError as e:
    log_step_error(logger, f"Could not import setup modules from src/setup/: {e}")
    logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.")
    perform_full_setup = None
    sys.exit(1)

def perform_setup_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized setup processing function.
    
    Args:
        target_dir: Directory containing GNN files (for validation)
        output_dir: Output directory for setup artifacts
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options (recreate_venv, dev, etc.)
        
    Returns:
        True if setup succeeded, False otherwise
    """
    try:
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Log environment information
        log_environment_info(logger)
        
        # Verify directories
        if not verify_directories(target_dir, output_dir, logger, find_gnn_files, verbose):
            log_step_error(logger, "Directory verification failed")
            return False
        
        # Perform full environment setup
        logger.info("Performing full environment setup...")
        if perform_full_setup:
            try:
                exit_code = perform_full_setup(
                    verbose=verbose,
                    recreate_venv=kwargs.get('recreate_venv', False),
                    dev=kwargs.get('dev', False)
                )
                if exit_code != 0:
                    log_step_error(logger, "Environment setup failed")
                    return False
            except Exception as e:
                log_step_error(logger, f"Environment setup failed with exception: {e}")
                return False
        else:
            log_step_warning(logger, "Project environment setup not available, skipping")
        
        # List installed packages
        venv_info = get_venv_info()
        list_installed_packages(logger, venv_info, output_dir, verbose)
        
        log_step_success(logger, "Setup completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Setup processing failed: {e}")
        return False

def main(parsed_args):
    """Main function for setup operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("2_setup.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Environment setup and dependency installation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Perform setup
    success = perform_setup_standardized(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False),
                recreate_venv=getattr(parsed_args, 'recreate_venv', False),
                dev=getattr(parsed_args, 'dev', False)
            )
    
    if success:
        log_step_success(logger, "Setup completed successfully")
        return 0
    else:
        log_step_error(logger, "Setup failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("2_setup")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="Environment setup and dependency installation")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--recreate-venv", action="store_true",
                          help="Recreate virtual environment")
        parser.add_argument("--dev", action="store_true",
                          help="Install development dependencies")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 