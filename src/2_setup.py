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

# Attempt to import the new logging utility
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None
        # Temporary basic config for this specific warning if util is missing
        # This logger will be __main__ if script is run directly.
        _temp_logger_name = __name__ if __name__ != "__main__" else "src.2_setup_import_warning"
        _temp_logger = logging.getLogger(_temp_logger_name)
        if not _temp_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
            _temp_logger.warning(
                "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic."
            )
        elif not _temp_logger.hasHandlers(): # If root has handlers, but this one doesn't
            _temp_logger.addHandler(logging.StreamHandler(sys.stderr)) # Ensure it can output
            _temp_logger.warning(
                "Could not import setup_standalone_logging from utils.logging_utils. Standalone logging might be basic (using existing root handlers)."
            )
        else: # Already has handlers
             _temp_logger.warning(
                "Could not import setup_standalone_logging from utils.logging_utils (already has handlers)."
            )

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Relative import to access setup.py in the parent's sibling directory 'setup'
# Assuming 2_setup.py is in src/ and setup.py is in src/setup/
try:
    logger.debug("Attempting to import setup modules...")
    from setup import setup as project_env_setup # Accessing src/setup/setup.py
    from setup.utils import ensure_directory, find_gnn_files, get_output_paths
    from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
    logger.debug("Successfully imported setup modules")
except ImportError as e:
    # This might happen if PYTHONATH is not set up correctly or if the structure is unexpected.
    # Fallback for simpler execution or testing if needed, though relative import is preferred for packages.
    logger.warning(f"Initial import failed: {e}. Trying alternate import path...")
    sys.path.append(str(Path(__file__).parent.resolve())) # Ensure src/ is in path
    try:
        from setup import setup as project_env_setup
        from setup.utils import ensure_directory, find_gnn_files, get_output_paths
        from setup.utils import is_venv_active, get_venv_info, check_system_dependencies
        logger.debug("Successfully imported setup modules via alternate path")
    except ImportError as e2:
        logger.error("Error: Could not import setup modules from src/setup/")
        logger.error(f"Import error: {e2}")
        logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.")
        project_env_setup = None # Ensure it exists for later checks
        sys.exit(1)

def log_environment_info():
    """Log detailed information about the current environment."""
    logger.info("--- Environment Information ---")
    
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
    
    logger.info("-----------------------------")

def verify_directories(target_dir, output_dir, verbose=False):
    """Verify that target directory exists and create output directories."""
    logger.info("🔍 Verifying directories...")
    target_path = Path(target_dir)
    
    # Use logger.debug for verbose messages, logger.info for standard messages
    logger.debug(f"Verifying target directory: {target_path.resolve()}")

    # Check if target directory exists
    if not target_path.is_dir(): # More specific check for directory
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
        output_paths = get_output_paths(output_dir)
        logger.info(f"✅ Output directory structure verified/created: {Path(output_dir).resolve()}")
        
        # Save directory structure info for reference
        logger.debug("Saving directory structure info...")
        structure_info = {name: str(path) for name, path in output_paths.items()}
        structure_file = Path(output_dir) / "directory_structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        logger.debug(f"  📄 Directory structure info saved to: {structure_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating output directories: {e}")
        return False

def verify_virtualenv_setup(verbose=False):
    """Verify that the virtual environment is properly set up."""
    logger.info("🔍 Verifying virtual environment setup...")
    # Check if we're currently running in a venv
    if is_venv_active():
        logger.info("✅ Running inside a virtual environment")
        return True
    
    # If we're not in a venv, check if one exists
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv"
    
    if venv_path.exists() and venv_path.is_dir():
        logger.warning("⚠️ Virtual environment exists but is not activated")
        
        # Determine activation command based on platform
        if sys.platform == "win32":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        logger.info(f"  To activate, run: {activate_cmd}")
        return True
    else:
        logger.warning("⚠️ No virtual environment found. It will be created in the next step.")
        return True  # We'll create it later, so return True

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
    sys.stdout.flush()  # Force flush
    
    # Log detailed environment information
    logger.info("📊 Gathering environment information...")
    sys.stdout.flush()  # Force flush
    log_environment_info()
    sys.stdout.flush()  # Force flush
    
    logger.debug("  Phase 1: Verifying project directories...")
    sys.stdout.flush()  # Force flush
    
    target_dir_abs = Path(parsed_args.target_dir).resolve()
    output_dir_abs = Path(parsed_args.output_dir).resolve()

    # Pass verbose to verify_directories, although it now uses logger levels directly.
    if not verify_directories(str(target_dir_abs), str(output_dir_abs), parsed_args.verbose):
        logger.error("❌ Directory verification failed. Halting setup step.")
        sys.exit(1)
    
    logger.info("  ✅ Project directories verified successfully.")
    sys.stdout.flush()  # Force flush
    
    # Verify virtualenv setup
    logger.debug("  Verifying virtual environment setup...")
    sys.stdout.flush()  # Force flush
    verify_virtualenv_setup(parsed_args.verbose)
    sys.stdout.flush()  # Force flush
    
    logger.info("  Phase 2: Setting up Python virtual environment and dependencies...")
    logger.info("  ⏳ This may take a few minutes, especially if installing dependencies...")
    sys.stdout.flush()  # Force flush

    if project_env_setup and hasattr(project_env_setup, 'perform_full_setup'):
        try:
            logger.debug("  🐍 Attempting to run perform_full_setup from src/setup/setup.py")
            sys.stdout.flush()  # Force flush
            
            # Check if we should recreate the virtualenv based on argument
            recreate_venv = getattr(parsed_args, 'recreate_venv', False)
            # Check if we should install dev dependencies
            install_dev = getattr(parsed_args, 'dev', False)
            
            logger.info(f"  📦 Setup config: recreate_venv={recreate_venv}, install_dev={install_dev}")
            sys.stdout.flush()  # Force flush
            
            # Progress indicator for long-running setup
            logger.info("  🔄 Starting virtual environment and dependency setup...")
            sys.stdout.flush()  # Force flush
            setup_start_time = time.time()
            
            # Set up a progress reporting thread that will log periodically during the setup process
            stop_progress_thread = False
            
            def progress_reporter():
                last_report_time = time.time()
                progress_stages = [
                    "Checking system requirements...",
                    "Creating virtual environment...",
                    "Installing dependencies...",
                    "Processing packages...",
                    "Building wheels...",
                    "Finalizing installation..."
                ]
                stage_idx = 0
                
                while not stop_progress_thread:
                    current_time = time.time()
                    # Report progress every 30 seconds if no other output
                    if current_time - last_report_time >= 30:
                        elapsed = current_time - setup_start_time
                        stage_message = progress_stages[stage_idx % len(progress_stages)]
                        logger.info(f"  ⏳ Setup in progress: {stage_message} (elapsed: {elapsed:.1f}s)")
                        sys.stdout.flush()  # Force flush
                        last_report_time = current_time
                        stage_idx += 1
                    time.sleep(5)  # Check every 5 seconds
            
            # Start progress reporter in a separate thread
            import threading
            progress_thread = threading.Thread(target=progress_reporter, daemon=True)
            progress_thread.start()
            
            try:
                # Pass verbose, recreate_venv, and dev to perform_full_setup
                env_setup_result = project_env_setup.perform_full_setup(
                    verbose=parsed_args.verbose,
                    recreate_venv=recreate_venv,
                    dev=install_dev
                )
            finally:
                # Stop the progress thread
                stop_progress_thread = True
                progress_thread.join(timeout=1.0)  # Give it 1 second to finish
            
            setup_duration = time.time() - setup_start_time
            logger.info(f"  ⏱️ Setup process took {setup_duration:.1f} seconds")
            sys.stdout.flush()  # Force flush
            
            if env_setup_result != 0:
                logger.error(f"❌ Python virtual environment and dependency setup failed (returned code: {env_setup_result}).")
                sys.stdout.flush()  # Force flush
                
                # Provide more helpful error message based on common issues
                try:
                    # Try to import pymdp to see if it's actually available despite version issues
                    import importlib
                    
                    try:
                        spec = importlib.util.find_spec('pymdp')
                        if spec:
                            logger.warning("⚠️ pymdp module appears to be available but may not meet version requirements.")
                            logger.warning("   You may continue with caution, but PyMDP-dependent steps might fail.")
                        sys.stdout.flush()  # Force flush
                    except ImportError:
                        logger.warning("⚠️ Could not find pymdp module at all.")
                        sys.stdout.flush()  # Force flush
                        
                    # Check if we can suggest a fix for the dependency issue
                    req_file = Path(__file__).parent / "requirements.txt"
                    if req_file.exists():
                        with open(req_file, 'r') as f:
                            content = f.read()
                        if "inferactively-pymdp>=0.2.0" in content:
                            logger.warning("⚠️ Your requirements.txt specifies inferactively-pymdp>=0.2.0 but the latest available version may be lower.")
                            logger.warning("   Consider manually editing requirements.txt to use inferactively-pymdp<0.1.0 instead.")
                            sys.stdout.flush()  # Force flush
                except Exception as e:
                    logger.debug(f"Error during additional diagnostics: {e}")
                
                sys.exit(1)
            logger.info("  ✅ Python virtual environment and dependencies setup completed.")
            sys.stdout.flush()  # Force flush

            # Phase 3: Verify key dependencies
            logger.info("  Phase 3: Verifying key dependencies...")
            sys.stdout.flush()  # Force flush
            
            try:
                # Run pip list to get installed packages
                logger.info("  📦 Checking installed packages...")
                sys.stdout.flush()  # Force flush
                python_exe = sys.executable
                result = subprocess.run(
                    [python_exe, "-m", "pip", "list"], 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True
                )
                
                # Check for key dependencies
                key_deps = ["numpy", "scipy", "jax", "discopy"]
                missing_deps = []
                
                for dep in key_deps:
                    if dep not in result.stdout:
                        missing_deps.append(dep)
                
                if missing_deps:
                    logger.warning(f"  ⚠️ Some key dependencies may be missing: {', '.join(missing_deps)}")
                    sys.stdout.flush()  # Force flush
                    
                    # Try to fix by installing them directly
                    if parsed_args.verbose:
                        logger.info("  Attempting to install missing key dependencies directly...")
                    
                    for dep in missing_deps:
                        try:
                            logger.info(f"  📦 Installing missing dependency: {dep}...")
                            sys.stdout.flush()  # Force flush
                            subprocess.run(
                                [python_exe, "-m", "pip", "install", dep],
                                check=True,
                                stdout=subprocess.PIPE if not parsed_args.verbose else None,
                                stderr=subprocess.PIPE if not parsed_args.verbose else None
                            )
                            logger.info(f"  ✅ Successfully installed {dep}")
                            sys.stdout.flush()  # Force flush
                        except subprocess.CalledProcessError:
                            logger.warning(f"  ⚠️ Failed to install {dep}")
                            sys.stdout.flush()  # Force flush
                else:
                    logger.info("  ✅ All key dependencies appear to be installed.")
                    sys.stdout.flush()  # Force flush
                
                # Save pip list output to a file for reference
                pip_list_file = Path(parsed_args.output_dir) / "logs" / "pip_list.txt"
                with open(pip_list_file, 'w') as f:
                    f.write(result.stdout)
                logger.debug(f"  📄 Pip list saved to: {pip_list_file}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Error checking installed dependencies: {e}")
                sys.stdout.flush()  # Force flush

        except Exception as e:
            logger.error(f"❌ Error during virtual environment setup or core dependency installation: {e}", exc_info=True)
            sys.stdout.flush()  # Force flush
            sys.exit(1)
    else:
        logger.warning("⚠️ Warning: 'project_env_setup.perform_full_setup' not available from src/setup/setup.py. Skipping virtual environment setup.")
        logger.error("❌ Critical setup phase (virtual environment and dependencies) was skipped due to import issues.")
        sys.stdout.flush()  # Force flush
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