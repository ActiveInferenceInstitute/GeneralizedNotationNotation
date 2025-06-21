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

# Initialize logger for this step  
logger = setup_step_logging("2_setup", verbose=False)

# Import setup module for environment setup
try:
    from setup import setup as project_env_setup
    from setup.utils import ensure_directory, find_gnn_files, get_output_paths
    from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
    logger.debug("Successfully imported setup modules")
except ImportError as e:
    log_step_error(logger, f"Could not import setup modules from src/setup/: {e}")
    logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.")
    project_env_setup = None
    sys.exit(1)

def log_environment_info():
    """Log detailed information about the current environment."""
    log_step_start(logger, "Logging environment information")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Operating system
    import platform
    logger.info(f"Operating system: {platform.platform()}")
    
    # Virtual environment status
    logger.debug("Checking virtual environment status...")
    venv_info = get_venv_info()
    if venv_info["is_active"]:
        logger.info(f"Virtual environment: Active at {venv_info['venv_path']}")
        logger.info(f"  - Python: {venv_info['python_executable']}")
        logger.info(f"  - Pip: {venv_info['pip_executable']}")
    else:
        logger.info("Virtual environment: Not active")
    
    # System dependencies
    logger.debug("Checking system dependencies...")
    deps = check_system_dependencies()
    logger.info("System dependencies:")
    for dep, available in deps.items():
        logger.info(f"  - {dep}: {'✓ Available' if available else '✗ Not found'}")
    
    log_step_success(logger, "Environment information logged successfully")

def verify_directories(target_dir, output_dir, verbose=False):
    """Verify that target directory exists and create output directories."""
    log_step_start(logger, f"Verifying directories - target: {target_dir}, output: {output_dir}")
    
    target_path = Path(target_dir)
    
    # Check if target directory exists
    if not target_path.is_dir():
        log_step_error(logger, f"Target directory '{target_dir}' does not exist or is not a directory")
        return False
    
    try:
        logger.debug(f"Searching for GNN files in {target_path}...")
        gnn_files = find_gnn_files(target_path, recursive=True)
        logger.debug(f"Found {len(gnn_files)} GNN .md files (recursively in target: {target_path.resolve()})")
        
        if not gnn_files and verbose:
            log_step_warning(logger, f"No GNN files found in {target_path}. This might be expected if you're planning to create them later.")
    except Exception as e:
        log_step_error(logger, f"Error scanning for GNN files in {target_path}: {e}")
    
    # Create output directory structure using centralized configuration
    try:
        logger.info("Creating output directory structure...")
        
        output_paths = get_output_paths(output_dir)
        logger.info(f"Output directory structure verified/created: {Path(output_dir).resolve()}")
        
        # Save directory structure info for reference
        structure_info = {name: str(path) for name, path in output_paths.items()}
        structure_file = Path(output_dir) / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        logger.debug(f"Directory structure info saved to: {structure_file}")
        log_step_success(logger, "Directory structure created successfully")
        return True
    except Exception as e:
        log_step_error(logger, f"Error creating output directories: {e}")
        return False

def list_installed_packages(verbose: bool = False, output_dir: str = None):
    """
    List all installed packages in the virtual environment.
    
    Args:
        verbose: If True, print detailed package information.
        output_dir: If provided, save the package list to this directory.
    """
    log_step_start(logger, "Generating installed packages report")
    
    # Check if we're in a virtual environment
    venv_info = get_venv_info()
    if not venv_info["is_active"]:
        log_step_warning(logger, "Not running in a virtual environment, skipping package listing")
        return
    
    # Get pip executable
    pip_executable = venv_info["pip_executable"]
    if not pip_executable or not Path(pip_executable).exists():
        log_step_warning(logger, f"pip not found at {pip_executable}, skipping package listing")
        return
    
    try:
        # Run pip list with JSON format
        logger.debug(f"Running {pip_executable} list --format=json")
        pip_list_cmd = [pip_executable, "list", "--format=json"]
        result = subprocess.run(
            pip_list_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse JSON output
        packages = json.loads(result.stdout)
        package_dict = {pkg["name"]: pkg["version"] for pkg in packages}
        
        # Print summary
        logger.info(f"Found {len(package_dict)} installed packages in the virtual environment")
        
        # Log packages based on verbosity
        if verbose:
            logger.info("Installed packages:")
            for name, version in sorted(package_dict.items()):
                logger.info(f"  - {name}: {version}")
        else:
            # Show key packages even in non-verbose mode
            key_packages = ["pip", "pytest", "numpy", "matplotlib", "scipy"]
            logger.info("Key installed packages:")
            for pkg in key_packages:
                if pkg in package_dict:
                    logger.info(f"  - {pkg}: {package_dict[pkg]}")
        
        # Save detailed package list to file if output_dir is provided
        if output_dir:
            setup_dir = get_output_dir_for_script("2_setup.py", Path(output_dir))
            setup_dir.mkdir(parents=True, exist_ok=True)
            packages_file = setup_dir / "installed_packages.json"
            
            with open(packages_file, 'w') as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "virtual_env": venv_info["venv_path"],
                    "python_executable": venv_info["python_executable"],
                    "packages": package_dict
                }, f, indent=2)
            
            logger.debug(f"Package list saved to: {packages_file}")
        
        log_step_success(logger, f"Successfully listed {len(package_dict)} packages")
        
    except subprocess.CalledProcessError as e:
        log_step_error(logger, f"pip list command failed: {e}")
    except json.JSONDecodeError as e:
        log_step_error(logger, f"Failed to parse pip list JSON output: {e}")
    except Exception as e:
        log_step_error(logger, f"Unexpected error listing packages: {e}")

def main(parsed_args: argparse.Namespace):
    """Main function for setup operations."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("2_setup.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Environment setup and dependency installation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Log environment information
    log_environment_info()
    
    # Verify directories
    if not verify_directories(parsed_args.target_dir, parsed_args.output_dir, parsed_args.verbose):
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
    list_installed_packages(verbose=parsed_args.verbose, output_dir=parsed_args.output_dir)
    
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