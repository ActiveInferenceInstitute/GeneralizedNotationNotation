#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 2: Setup

This script performs initial setup tasks:
- Verifies and creates necessary output directories.
- Sets up the Python virtual environment and installs dependencies.

Usage:
    python 1_setup.py [options]
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
    get_output_dir_for_script
)

from utils.pipeline_template import create_standardized_pipeline_script

# Initialize logger for this step  
logger = setup_step_logging("1_setup", verbose=False)

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

def process_setup_standardized(
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
                    dev=kwargs.get('dev', False),
                    skip_jax_test=kwargs.get('skip_jax_test', True)
                )
                if exit_code != 0:
                    log_step_error(logger, "Setup failed")
                    return False
            except Exception as e:
                log_step_error(logger, f"Setup failed: {e}")
                return False
        else:
            log_step_warning(logger, "Project environment setup not available, skipping")
        
        # List installed packages
        venv_info = get_venv_info()
        list_installed_packages(logger, venv_info, output_dir, verbose)
        
        log_step_success(logger, "Setup completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Setup failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "1_setup.py",
    process_setup_standardized,
    "Environment setup and dependency installation",
    additional_arguments={
        "recreate_venv": {"type": bool, "default": False, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "default": False, "help": "Install development dependencies"},
        "skip_jax_test": {"type": bool, "default": True, "help": "Skip JAX installation testing (faster setup)"}
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 