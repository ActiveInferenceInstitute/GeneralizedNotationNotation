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
import logging
import argparse
import subprocess
import shutil
import json
import time

# Attempt to import the logging utility
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None
        # Temporary basic config for this specific warning if util is missing
        _temp_logger_name = __name__ if __name__ != "__main__" else "src.2_setup_import_warning"
        _temp_logger = logging.getLogger(_temp_logger_name)
        if not _temp_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
            _temp_logger.warning(
                "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic."
            )

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Import setup module for environment setup
try:
    from setup import setup as project_env_setup
    from setup.utils import ensure_directory, find_gnn_files, get_output_paths
    from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
    logger.debug("Successfully imported setup modules")
except ImportError as e:
    # Fallback if the script is run in a context where src/setup is not directly importable
    sys.path.append(str(Path(__file__).parent.resolve())) # Ensure src/ is in path
    try:
        from setup import setup as project_env_setup
        from setup.utils import ensure_directory, find_gnn_files, get_output_paths
        from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
        logger.debug("Successfully imported setup modules via alternate path")
    except ImportError as e2:
        logger.error(f"Error: Could not import setup modules from src/setup/: {e2}")
        logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.")
        project_env_setup = None
        sys.exit(1)

def log_environment_info():
    """Log detailed information about the current environment."""
    logger.info("--- Environment Information ---")
    sys.stdout.flush()
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    sys.stdout.flush()
    
    # Operating system
    import platform
    logger.info(f"Operating system: {platform.platform()}")
    sys.stdout.flush()
    
    # Virtual environment status
    logger.debug("Checking virtual environment status...")
    venv_info = get_venv_info()
    if venv_info["is_active"]:
        logger.info(f"Virtual environment: Active at {venv_info['venv_path']}")
        logger.info(f"  - Python: {venv_info['python_executable']}")
        logger.info(f"  - Pip: {venv_info['pip_executable']}")
    else:
        logger.info("Virtual environment: Not active")
    sys.stdout.flush()
    
    # System dependencies
    logger.debug("Checking system dependencies...")
    deps = check_system_dependencies()
    logger.info("System dependencies:")
    for dep, available in deps.items():
        logger.info(f"  - {dep}: {'✓ Available' if available else '✗ Not found'}")
    sys.stdout.flush()
    
    logger.info("-----------------------------")
    sys.stdout.flush()

def verify_directories(target_dir, output_dir, verbose=False):
    """Verify that target directory exists and create output directories."""
    logger.info("🔍 Verifying directories...")
    sys.stdout.flush()
    
    target_path = Path(target_dir)
    
    # Check if target directory exists
    if not target_path.is_dir():
        logger.error(f"❌ Error: Target directory '{target_dir}' does not exist or is not a directory")
        return False
    
    try:
        logger.debug(f"Searching for GNN files in {target_path}...")
        gnn_files = find_gnn_files(target_path, recursive=True)
        logger.debug(f"  Found {len(gnn_files)} GNN .md files (recursively in target: {target_path.resolve()})")
        
        if not gnn_files and verbose:
            logger.warning(f"⚠️ Warning: No GNN files found in {target_path}. This might be expected if you're planning to create them later.")
    except Exception as e:
        logger.error(f"❌ Error scanning for GNN files in {target_path}: {e}")
    
    # Create output directory structure
    try:
        logger.info("📁 Creating output directory structure...")
        sys.stdout.flush()
        
        output_paths = get_output_paths(output_dir)
        logger.info(f"✅ Output directory structure verified/created: {Path(output_dir).resolve()}")
        sys.stdout.flush()
        
        # Save directory structure info for reference
        structure_info = {name: str(path) for name, path in output_paths.items()}
        structure_file = Path(output_dir) / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        logger.debug(f"  📄 Directory structure info saved to: {structure_file}")
        sys.stdout.flush()
        return True
    except Exception as e:
        logger.error(f"❌ Error creating output directories: {e}")
        return False

def main(parsed_args: argparse.Namespace):
    """Main function for the setup step (Step 2).

    Orchestrates directory verification and Python virtual environment setup.

    Args:
        parsed_args (argparse.Namespace): Pre-parsed command-line arguments.
            Expected attributes include: target_dir, output_dir, verbose, dev.
    """    
    start_time = time.time()
    
    # Setup logging level for this script's logger based on verbosity.
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logger.setLevel(log_level)
    logger.debug(f"Script logger '{logger.name}' level set to {logging.getLevelName(log_level)}.")
    
    # Ensure logs are flushed immediately for better visibility in main.py
    sys.stdout.flush()

    logger.info("▶️ Starting Step 2: Setup")
    logger.debug(f"  Parsed arguments for setup: {parsed_args}")
    sys.stdout.flush()
    
    # Log detailed environment information
    logger.info("📊 Gathering environment information...")
    sys.stdout.flush()
    log_environment_info()
    sys.stdout.flush()
    
    logger.debug("  Phase 1: Verifying project directories...")
    sys.stdout.flush()
    
    target_dir_abs = Path(parsed_args.target_dir).resolve()
    output_dir_abs = Path(parsed_args.output_dir).resolve()

    # Verify directories
    if not verify_directories(str(target_dir_abs), str(output_dir_abs), parsed_args.verbose):
        logger.error("❌ Directory verification failed. Halting setup step.")
        sys.exit(1)
    
    logger.info("  ✅ Project directories verified successfully.")
    sys.stdout.flush()
    
    # Verify virtual environment and install dependencies
    logger.info("  Phase 2: Setting up Python virtual environment and dependencies...")
    logger.info("  ⏳ This may take a few minutes, especially if installing dependencies...")
    sys.stdout.flush()

    if project_env_setup and hasattr(project_env_setup, 'perform_full_setup'):
        try:
            logger.debug("  🐍 Calling perform_full_setup from src/setup/setup.py")
            sys.stdout.flush()
            
            # Check if we should recreate the virtualenv based on argument
            recreate_venv = getattr(parsed_args, 'recreate_venv', False)
            # Check if we should install dev dependencies
            install_dev = getattr(parsed_args, 'dev', False)
            
            logger.info(f"  📦 Setup config: recreate_venv={recreate_venv}, install_dev={install_dev}")
            sys.stdout.flush()
            
            # Call the setup function
            env_setup_result = project_env_setup.perform_full_setup(
                verbose=parsed_args.verbose,
                recreate_venv=recreate_venv,
                dev=install_dev
            )
            
            if env_setup_result != 0:
                logger.error(f"❌ Python virtual environment and dependency setup failed (returned code: {env_setup_result}).")
                sys.stdout.flush()
                sys.exit(1)
                
            logger.info("  ✅ Python virtual environment and dependencies setup completed.")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"❌ Error during virtual environment setup or core dependency installation: {e}")
            sys.stdout.flush()
            sys.exit(1)
    else:
        logger.error("❌ Critical setup phase (virtual environment and dependencies) was skipped due to import issues.")
        sys.stdout.flush()
        sys.exit(1)
    
    total_duration = time.time() - start_time
    logger.info(f"✅ Step 2: Setup complete (took {total_duration:.1f} seconds)")
    sys.exit(0)

if __name__ == "__main__":
    script_file_path = Path(__file__).resolve()
    project_root_for_defaults = script_file_path.parent.parent
    default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples"
    default_output_dir = project_root_for_defaults / "output"

    parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 2: Setup (Standalone)." )
    parser.add_argument(
        "--target-dir", 
        type=Path, 
        default=default_target_dir,
        help=f"Target directory for GNN source files (used for verification, default: {default_target_dir.relative_to(project_root_for_defaults) if default_target_dir.is_relative_to(project_root_for_defaults) else default_target_dir})"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=default_output_dir,
        help=f"Main directory to save outputs (default: {default_output_dir.relative_to(project_root_for_defaults) if default_output_dir.is_relative_to(project_root_for_defaults) else default_output_dir})"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=False, # Default to False for standalone, main.py controls for pipeline
        help="Enable verbose (DEBUG level) logging."
    )
    parser.add_argument(
        "--recreate-venv", 
        action="store_true", 
        help="Recreate virtual environment even if it already exists."
    )
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Also install development dependencies from requirements-dev.txt."
    )
    cli_args = parser.parse_args()

    # Setup logging for standalone execution
    log_level_to_set = logging.DEBUG if cli_args.verbose else logging.INFO
    if setup_standalone_logging:
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__)
    else:
        # Fallback basic config if utility function couldn't be imported
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=log_level_to_set,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                stream=sys.stdout
            )
        logging.getLogger(__name__).setLevel(log_level_to_set)
        logging.getLogger(__name__).warning("Using fallback basic logging due to missing setup_standalone_logging utility.")

    main(cli_args) 