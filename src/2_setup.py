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
import argparse # Added

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

def main(cmd_args=None): # Renamed 'args' to 'cmd_args' to avoid conflict with module name 'args'
    """Main function for the setup step (Step 2).

    Handles argument parsing if run standalone, then orchestrates directory
    verification and Python virtual environment setup.

    Args:
        cmd_args (argparse.Namespace | list | None):
            - If None, parses arguments from sys.argv for standalone execution.
            - If a list, assumes it's a list of string arguments to be parsed.
            - If an argparse.Namespace, uses it directly.
            Expected attributes on the Namespace object include:
            target_dir, output_dir, verbose.
    """
    # Main function for the setup step.
    
    if cmd_args is None: # Standalone execution
        script_file_path = Path(__file__).resolve()
        project_root_for_defaults = script_file_path.parent.parent
        default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples" # Consistent with 1_gnn.py
        default_output_dir = project_root_for_defaults / "output"

        parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 2: Setup.")
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
            help="Enable verbose (DEBUG level) logging."
        )
        parsed_args = parser.parse_args() # sys.argv will be used here
    elif isinstance(cmd_args, list): # If called with a list of string args
        script_file_path = Path(__file__).resolve()
        project_root_for_defaults = script_file_path.parent.parent
        default_target_dir = project_root_for_defaults / "src" / "gnn" / "examples"
        default_output_dir = project_root_for_defaults / "output"
        parser = argparse.ArgumentParser(description="GNN Processing Pipeline - Step 2: Setup.")
        parser.add_argument("--target-dir", type=Path, default=default_target_dir)
        parser.add_argument("--output-dir", type=Path, default=default_output_dir)
        parser.add_argument("--verbose", action="store_true")
        parsed_args = parser.parse_args(cmd_args)
    elif hasattr(cmd_args, 'target_dir'): # If it's already a parsed Namespace-like object
        parsed_args = cmd_args
    else:
        # Use print for initial error as logger might not be configured
        print("ERROR: Invalid 'cmd_args' type passed to 2_setup.py main(). Expected None, list, or Namespace-like object.", file=sys.stderr)
        sys.exit(1)

    # Setup logging level based on verbosity
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    # Configure root logger if not already done (e.g., by main.py)
    # If already configured, this will be a no-op for level/format if propagate=True for this logger.
    # We mainly want to ensure that if run standalone, it logs something.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    else: # If root has handlers (e.g., from main.py), just set this script's logger level
        logger.setLevel(log_level)

    logger.info("▶️ Starting Step 2: Setup")
    logger.debug(f"  Parsed arguments for setup: {parsed_args}")
    logger.debug("  Phase 1: Verifying project directories...")
    
    target_dir_abs = Path(parsed_args.target_dir).resolve()
    output_dir_abs = Path(parsed_args.output_dir).resolve()

    # Pass string paths to verify_directories as it expects strings currently
    if not verify_directories(str(target_dir_abs), str(output_dir_abs), parsed_args.verbose):
        logger.error("❌ Directory verification failed. Halting setup step.")
        sys.exit(1)
    
    logger.info("  ✅ Project directories verified successfully.")
    logger.info("  Phase 2: Setting up Python virtual environment and dependencies...")

    if project_env_setup and hasattr(project_env_setup, 'perform_full_setup'):
        try:
            logger.debug("  🐍 Attempting to run perform_full_setup from src/setup/setup.py")
            # Pass verbose to perform_full_setup. This requires perform_full_setup to accept it.
            env_setup_result = project_env_setup.perform_full_setup(verbose=parsed_args.verbose) 
            if env_setup_result != 0:
                logger.error(f"❌ Python virtual environment and dependency setup failed (returned code: {env_setup_result}).")
                sys.exit(1)
            logger.info("  ✅ Python virtual environment and dependencies setup completed.")

            logger.info("  Phase 3: Confirming PyMDP availability (informational)...")
            pymdp_confirmed_fully = False
            try:
                import pymdp # type: ignore
                logger.info("    Successfully imported the 'pymdp' module.")
                pymdp_version_str = "N/A"
                try:
                    import importlib.metadata
                    pymdp_version_str = importlib.metadata.version('inferactively-pymdp')
                    logger.info(f"    Version of 'inferactively-pymdp' via importlib.metadata: {pymdp_version_str}")
                except importlib.metadata.PackageNotFoundError:
                    logger.warning("    ⚠️ 'inferactively-pymdp' package not found by importlib.metadata. Version unknown.")
                except Exception as e_meta:
                    logger.warning(f"    ⚠️ Error getting version via importlib.metadata: {e_meta}")
                pymdp_module_location = getattr(pymdp, '__file__', 'N/A')
                logger.info(f"    'pymdp' module location: {pymdp_module_location}")
                from pymdp.agent import Agent # type: ignore
                logger.info(f"    Successfully imported 'pymdp.agent.Agent': {Agent}")
                agent_module = sys.modules.get(Agent.__module__)
                agent_module_file = getattr(agent_module, '__file__', 'N/A') if agent_module else 'N/A'
                logger.info(f"    'pymdp.agent.Agent' module ({Agent.__module__}) location: {agent_module_file}")
                logger.info("  ✅ PyMDP module and Agent class appear to be available and importable.")
                pymdp_confirmed_fully = True
            except ImportError as e_imp:
                logger.warning(f"  ⚠️ Failed to import 'pymdp' or 'pymdp.agent.Agent': {e_imp}")
                logger.warning("      This may affect PyMDP-dependent steps (e.g., 9_render, 10_execute).")
            except Exception as e_other:
                logger.warning(f"  ⚠️ An unexpected error occurred during PyMDP availability check: {e_other}", exc_info=parsed_args.verbose)
            
            if not pymdp_confirmed_fully:
                logger.warning("  ⚠️ PyMDP availability check did not fully succeed. Subsequent PyMDP steps may fail or use unexpected versions.")

        except Exception as e:
            logger.error(f"❌ Error during virtual environment setup or core dependency installation: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.warning("⚠️ Warning: 'project_env_setup.perform_full_setup' not available from src/setup/setup.py. Skipping virtual environment setup.")
        logger.error("❌ Critical setup phase (virtual environment and dependencies) was skipped due to import issues.")
        sys.exit(1)
    
    logger.info("✅ Step 2: Setup complete")
    sys.exit(0)

if __name__ == "__main__":
    # When run standalone, main() will parse sys.argv itself.
    main() 