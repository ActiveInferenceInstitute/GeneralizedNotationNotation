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
import logging # Add logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Relative import to access setup.py in the parent's sibling directory 'setup'
# Assuming 2_setup.py is in src/ and setup.py is in src/setup/
try:
    from setup import setup as project_env_setup # Accessing src/setup/setup.py
except ImportError as e:
    # This might happen if PYTHONATH is not set up correctly or if the structure is unexpected.
    # Fallback for simpler execution or testing if needed, though relative import is preferred for packages.
    sys.path.append(str(Path(__file__).parent.resolve())) # Ensure src/ is in path
    try:
        from setup import setup as project_env_setup
    except ImportError:
        logger.error("Error: Could not import 'perform_full_setup' from src/setup/setup.py") # Changed from print
        logger.error(f"Import error: {e}") # Changed from print
        logger.error("Ensure src/setup/setup.py exists and src/ is in your PYTHONPATH or accessible.") # Changed from print
        project_env_setup = None # Ensure it exists for later checks

def verify_directories(target_dir, output_dir, verbose=False):
    """Verify that target directory exists and create output directories."""
    target_path = Path(target_dir)
    
    # Use logger.debug for verbose messages, logger.info for standard messages
    logger.debug(f"Verifying target directory: {target_path.resolve()}")

    # Check if target directory exists
    if not target_path.is_dir(): # More specific check for directory
        logger.error(f"❌ Error: Target directory '{target_dir}' does not exist or is not a directory") # Changed from print
        return False
    
    logger.debug(f"  Found {sum(1 for f in target_path.rglob('*.md') if f.is_file())} .md files (recursively in target: {target_path.resolve()})")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    logger.debug(f"📂 Ensuring output directory: {output_path.resolve()}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization output directory
    viz_output = output_path / "gnn_examples_visualization"
    logger.debug(f"  📂 Ensuring visualization directory: {viz_output.resolve()}")
    viz_output.mkdir(exist_ok=True)
    
    # Create type checker output directory
    type_output = output_path / "gnn_type_check"
    logger.debug(f"  📂 Ensuring type check directory: {type_output.resolve()}")
    type_output.mkdir(exist_ok=True)
    
    logger.info(f"✅ Output directory structure verified/created: {output_path.resolve()}") # Was print if verbose
    
    return True

def main(args):
    """Main function for the setup step."""
    # Logger level for __name__ is set by main.py based on args.verbose.
    # This script's logger.info/debug messages will respect that.
    
    logger.info("▶️ Starting Step 2: Setup") # Was print if verbose
    logger.debug("  Phase 1: Verifying project directories...") # Was print if verbose
    
    # Verify project structure directories (e.g., output folders)
    if not verify_directories(args.target_dir, args.output_dir, args.verbose):
        logger.error("❌ Directory verification failed. Halting setup step.") # Changed from print
        return 1 # Indicate failure
    
    logger.info("  ✅ Project directories verified successfully.") # Was print if verbose
    logger.info("  Phase 2: Setting up Python virtual environment and dependencies...") # Was print and added newline

    if project_env_setup and hasattr(project_env_setup, 'perform_full_setup'):
        try:
            logger.debug("  🐍 Attempting to run perform_full_setup from src/setup/setup.py") # Was print if verbose
            # The perform_full_setup function in src/setup/setup.py now handles its own
            # pathing relative to the 'src/' directory.
            # NOTE: project_env_setup.perform_full_setup() likely prints directly.
            # For full logging control, it should be modified to accept a logger or verbosity level,
            # and use logging instead of print, or return its output for this script to log.
            env_setup_result = project_env_setup.perform_full_setup() # Pass verbose
            if env_setup_result != 0:
                logger.error("❌ Python virtual environment and dependency setup failed.") # Changed from print
                return 1 # Propagate failure
            logger.info("  ✅ Python virtual environment and dependencies setup completed.") # Was print if verbose
        except Exception as e:
            logger.error(f"❌ Error during virtual environment setup: {e}", exc_info=True) # Changed from print, added exc_info
            # import traceback # Not needed if using exc_info=True
            # traceback.print_exc()
            return 1 # Indicate failure
    else:
        logger.warning("⚠️ Warning: `project_env_setup` or `perform_full_setup` not available. Skipping virtual environment setup.") # Was print
        logger.error("❌ Critical setup phase (virtual environment and dependencies) was skipped due to import issues.") # Changed from print
        return 1 # Treat as failure as this is a critical step
    
    logger.info("✅ Step 2: Setup complete") # Was print if verbose
    return 0 # Indicate success

if __name__ == "__main__":
    # Basic configuration for running this script standalone
    # In a pipeline, main.py should configure logging.
    # Determine log level based on a simple environment variable or default to INFO
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Create a dummy args object if needed, or use a minimal one for standalone execution
    class DummyArgs:
        def __init__(self):
            self.verbose = (log_level == logging.DEBUG) # Example: link verbose to DEBUG
            # Add other attributes that main() might expect, with sensible defaults for standalone run
            self.output_dir = "../output" # Example default
            self.target_dir = "gnn/examples" # Example default
            # ... any other args the script's main() expects

    dummy_args = DummyArgs()
    main(dummy_args) 