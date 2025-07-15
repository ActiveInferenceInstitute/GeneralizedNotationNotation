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

import os
import sys
from pathlib import Path
import argparse
import subprocess
import shutil
import json
import time
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

# Import setup module for environment setup
try:
    from setup import setup as project_env_setup
    from setup.utils import ensure_directory, find_gnn_files, get_output_paths
    from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
    from setup.utils import log_environment_info, verify_directories, list_installed_packages
    logger.debug("Successfully imported setup modules")
except ImportError as e:
    log_step_error(logger, f"Could not import setup modules from src/setup/: {e}")
    logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.")
    project_env_setup = None
    sys.exit(1)

# Initialize logger for this step  
logger = setup_step_logging("2_setup", verbose=False)

def main(parsed_args: argparse.Namespace):
    """Main function for setup operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("2_setup.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Environment setup and dependency installation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Log environment information
    log_environment_info(logger)
    
    # Verify directories
    if not verify_directories(logger, parsed_args.target_dir, parsed_args.output_dir, parsed_args.verbose):
        log_step_error(logger, "Directory verification failed")
        return 1
    
    # Perform full environment setup
    logger.info("Performing full environment setup...")
    if project_env_setup:
        try:
            exit_code = project_env_setup.perform_full_setup(
                verbose=parsed_args.verbose,
                recreate_venv=getattr(parsed_args, 'recreate_venv', False),
                dev=getattr(parsed_args, 'dev', False)
            )
            if exit_code != 0:
                log_step_error(logger, "Environment setup failed")
                return 1
        except Exception as e:
            log_step_error(logger, f"Environment setup failed with exception: {e}")
            return 1
    else:
        log_step_warning(logger, "Project environment setup not available, skipping")
    
    # List installed packages
    list_installed_packages(logger, verbose=parsed_args.verbose, output_dir=parsed_args.output_dir)
    
    log_step_success(logger, "Setup completed successfully")
    return 0

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("2_setup")
    else:
        # Fallback argument parsing
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