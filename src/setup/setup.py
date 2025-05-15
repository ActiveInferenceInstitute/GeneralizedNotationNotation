"""
Main setup script for the GNN project.

This script handles the creation of a virtual environment and the installation
of project dependencies when called directly or via its perform_full_setup function.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging
import argparse

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# --- Configuration ---
VENV_DIR = ".venv"  # Name of the virtual environment directory
REQUIREMENTS_FILE = "requirements.txt"

# PROJECT_ROOT should be the 'src/' directory.
# Since this script is in 'src/setup/', Path(__file__).parent is 'src/setup'.
# So, Path(__file__).parent.parent is 'src/'.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

VENV_PATH = PROJECT_ROOT / VENV_DIR
REQUIREMENTS_PATH = PROJECT_ROOT / REQUIREMENTS_FILE

# Determine the correct Python interpreter path for the virtual environment
if sys.platform == "win32":
    VENV_PYTHON = VENV_PATH / "Scripts" / "python.exe"
    VENV_PIP = VENV_PATH / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_PATH / "bin" / "python"
    VENV_PIP = VENV_PATH / "bin" / "pip"

# --- Helper Functions ---

def run_command(command: list[str], cwd: Path = PROJECT_ROOT, check: bool = True, verbose: bool = False) -> None:
    """
    Runs a shell command and logs its output based on verbosity.

    Args:
        command: The command and its arguments as a list of strings.
        cwd: The current working directory for the command.
        check: If True, raises CalledProcessError if the command returns a non-zero exit code.
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
    """
    command_str_list = [str(c) for c in command]
    if verbose:
        logger.debug(f"Running command: '{' '.join(command_str_list)}' in {cwd}")
    else:
        logger.debug(f"Running command: '{command_str_list[0]} ...' in {cwd}")
    
    try:
        process = subprocess.run(command_str_list, cwd=cwd, check=check, capture_output=True, text=True, errors='replace')
        if verbose:
            if process.stdout:
                logger.debug(f"Stdout:\n{process.stdout.strip()}")
            if process.stderr:
                logger.debug(f"Stderr:\n{process.stderr.strip()}")
        if not check and process.returncode != 0:
            logger.warning(f"Command returned non-zero exit code: {process.returncode}")
            if process.stdout:
                logger.warning(f"Stdout:\n{process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"Stderr:\n{process.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: '{' '.join(e.cmd)}'")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"Stdout:\n{e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Stderr:\n{e.stderr.strip()}")
        if check:
            raise
    except FileNotFoundError as e:
        logger.error(f"Error: Command not found - {command_str_list[0]}. Ensure it is installed and in PATH.")
        logger.error(f"Details: {e}")
        if check:
            raise

def create_virtual_environment(verbose: bool = False) -> None:
    """
    Creates a virtual environment if it doesn't already exist.

    Args:
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
    """
    if not VENV_PATH.exists():
        logger.info(f"Creating virtual environment in {VENV_PATH}...")
        try:
            run_command([sys.executable, "-m", "venv", VENV_DIR], cwd=PROJECT_ROOT, verbose=verbose)
            logger.info(f"Virtual environment created successfully at {VENV_PATH}")
        except Exception as e:
            logger.error(f"Failed to create virtual environment: {e}", exc_info=verbose)
            raise
    else:
        logger.info(f"Virtual environment already exists at {VENV_PATH}")

def install_dependencies(verbose: bool = False) -> None:
    """
    Installs or updates dependencies from the requirements.txt file
    into the virtual environment.

    Args:
        verbose: If True, enables detailed (DEBUG level) logging for this setup process.
    """
    if not REQUIREMENTS_PATH.exists():
        logger.warning(f"{REQUIREMENTS_PATH} not found. Skipping dependency installation from this file.")
        return

    logger.info(f"Attempting to install 'inferactively-pymdp' individually into {VENV_PATH}...")
    try:
        if not VENV_PIP.exists():
            logger.error(f"Pip executable not found at {VENV_PIP}. Cannot install dependencies.")
            raise FileNotFoundError(f"Pip not found at {VENV_PIP}")
        
        run_command([str(VENV_PIP), "install", "inferactively-pymdp"], cwd=PROJECT_ROOT, check=False, verbose=verbose)
        logger.debug("Individual installation attempt for 'inferactively-pymdp' completed.")
    except Exception as e:
        logger.warning(f"An error occurred during the individual installation of 'inferactively-pymdp': {e}", exc_info=verbose)

    logger.info(f"Installing/updating dependencies from {REQUIREMENTS_PATH} into {VENV_PATH}...")
    try:
        if not VENV_PIP.exists():
            logger.error(f"Pip executable not found at {VENV_PIP}. Cannot install dependencies.")
            raise FileNotFoundError(f"Pip not found at {VENV_PIP}")

        run_command([str(VENV_PIP), "install", "-r", str(REQUIREMENTS_PATH)], cwd=PROJECT_ROOT, verbose=verbose)
        logger.info("Dependencies from requirements.txt installed successfully.")
    except subprocess.CalledProcessError:
        logger.error("Failed to install dependencies from requirements.txt.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during dependency installation from requirements.txt: {e}", exc_info=verbose)
        raise

# --- Callable Main Function ---
def perform_full_setup(verbose: bool = False):
    """
    Performs the full setup: creates virtual environment and installs dependencies.
    This function is intended to be called by other scripts.

    Args:
        verbose (bool): If True, enables detailed (DEBUG level) logging for this setup process.
    """
    
    # Configure logger for this module based on verbosity passed from caller
    # This assumes that the caller (e.g., 2_setup.py) has already configured the root logger if necessary.
    # We are setting the level for THIS logger instance.
    log_level_to_set = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level_to_set)
    # If no handlers are configured on the root logger (e.g. running this script directly without a pre-configured root logger),
    # add a basic one for this logger to output to console.
    if not logging.getLogger().hasHandlers() and not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.propagate = False # Avoid double-logging if root also gets configured later

    logger.info(f"Starting environment setup (venv and dependencies) targeting {PROJECT_ROOT}...")
    logger.debug(f"Verbose mode: {verbose}")
    logger.debug(f"Project root: {PROJECT_ROOT}")
    logger.debug(f"Venv path: {VENV_PATH}")
    logger.debug(f"Requirements path: {REQUIREMENTS_PATH}")
    logger.debug(f"Venv Python: {VENV_PYTHON}")
    logger.debug(f"Venv Pip: {VENV_PIP}")

    try:
        create_virtual_environment(verbose=verbose)
        install_dependencies(verbose=verbose)
        logger.info(f"Environment setup completed successfully for {PROJECT_ROOT}.")
        logger.info(f"To activate the virtual environment, navigate to {PROJECT_ROOT} and run:")
        if sys.platform == "win32":
            logger.info(f"  .\\{VENV_DIR}\\Scripts\\activate")
        else:
            logger.info(f"  source {VENV_DIR}/bin/activate")
        return 0 # Success
    except Exception as e:
        logger.error(f"Environment setup failed: {e}", exc_info=verbose)
        return 1 # Failure

# --- Main Execution (for direct script running) ---

if __name__ == "__main__":
    # Basic argument parsing for direct execution
    parser = argparse.ArgumentParser(description="Direct execution of GNN project setup script (venv and dependencies).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG level) logging.")
    cli_args = parser.parse_args()

    # Setup basic logging for direct run if not already configured by perform_full_setup's internal check
    # This initial basicConfig is for messages before perform_full_setup potentially reconfigures its own handler.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG if cli_args.verbose else logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

    logger.info("Running src/setup/setup.py directly...")
    exit_code = perform_full_setup(verbose=cli_args.verbose)
    if exit_code == 0:
        logger.info("Direct execution of setup.py completed.")
    else:
        logger.error("Direct execution of setup.py failed.")
    sys.exit(exit_code) 