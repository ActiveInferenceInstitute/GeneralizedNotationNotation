"""
Main setup script for the GNN project.

This script handles the creation of a virtual environment and the installation
of project dependencies when called directly or via its perform_full_setup function.
"""

import os
import subprocess
import sys
from pathlib import Path

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

def run_command(command: list[str], cwd: Path = PROJECT_ROOT, check: bool = True) -> None:
    """
    Runs a shell command and prints its output.

    Args:
        command: The command and its arguments as a list of strings.
        cwd: The current working directory for the command.
        check: If True, raises CalledProcessError if the command returns a non-zero exit code.
    """
    # Ensure command elements are strings
    command_str_list = [str(c) for c in command]
    print(f"Running command: '{' '.join(command_str_list)}' in {cwd}")
    try:
        process = subprocess.run(command_str_list, cwd=cwd, check=check, capture_output=True, text=True)
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(process.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: '{' '.join(e.cmd)}'", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        if check:
            raise
    except FileNotFoundError as e:
        print(f"Error: Command not found - {command_str_list[0]}. Ensure it is installed and in PATH.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        if check:
            raise

def create_virtual_environment() -> None:
    """
    Creates a virtual environment if it doesn't already exist.
    """
    if not VENV_PATH.exists():
        print(f"Creating virtual environment in {VENV_PATH}...")
        try:
            # Use sys.executable to ensure we're using the python that runs this script
            run_command([sys.executable, "-m", "venv", VENV_DIR], cwd=PROJECT_ROOT)
            print(f"Virtual environment created successfully at {VENV_PATH}")
        except Exception as e:
            print(f"Failed to create virtual environment: {e}", file=sys.stderr)
            # sys.exit(1) # Avoid exiting if called as a library
            raise # Re-raise the exception to be handled by the caller
    else:
        print(f"Virtual environment already exists at {VENV_PATH}")

def install_dependencies() -> None:
    """
    Installs or updates dependencies from the requirements.txt file
    into the virtual environment.
    """
    if not REQUIREMENTS_PATH.exists():
        print(f"Error: {REQUIREMENTS_PATH} not found.", file=sys.stderr)
        print(f"Please create a {REQUIREMENTS_FILE} file in the {PROJECT_ROOT} directory.", file=sys.stderr)
        # Optionally, create an empty one if it doesn't exist
        # REQUIREMENTS_PATH.touch()
        # print(f"Created an empty {REQUIREMENTS_PATH}. Please add your dependencies.")
        return # Or raise an error if it's critical

    print(f"Attempting to install 'inferactively-pymdp' individually into {VENV_PATH}...")
    try:
        if not VENV_PIP.exists():
            print(f"Error: Pip executable not found at {VENV_PIP}.", file=sys.stderr)
            print("Please ensure the virtual environment was created correctly.", file=sys.stderr)
            raise FileNotFoundError(f"Pip not found at {VENV_PIP}")
        
        run_command([str(VENV_PIP), "install", "inferactively-pymdp"], cwd=PROJECT_ROOT, check=False) # Try this first, don't stop if it fails
        print("Individual installation attempt for 'inferactively-pymdp' completed.")
    except Exception as e:
        print(f"An error occurred during the individual installation of 'inferactively-pymdp': {e}", file=sys.stderr)
        # We still proceed to requirements.txt

    print(f"Installing/updating dependencies from {REQUIREMENTS_PATH} into {VENV_PATH}...")
    try:
        if not VENV_PIP.exists():
            print(f"Error: Pip executable not found at {VENV_PIP}.", file=sys.stderr)
            print("Please ensure the virtual environment was created correctly.", file=sys.stderr)
            # sys.exit(1) # Avoid exiting
            raise FileNotFoundError(f"Pip not found at {VENV_PIP}")

        run_command([str(VENV_PIP), "install", "-r", str(REQUIREMENTS_PATH)], cwd=PROJECT_ROOT)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.", file=sys.stderr)
        # sys.exit(1) # Avoid exiting
        raise
    except Exception as e:
        print(f"An unexpected error occurred during dependency installation: {e}", file=sys.stderr)
        # sys.exit(1) # Avoid exiting
        raise

# --- Callable Main Function ---
def perform_full_setup():
    """
    Performs the full setup: creates virtual environment and installs dependencies.
    This function is intended to be called by other scripts.
    """
    print(f"Starting environment setup (venv and dependencies) targeting {PROJECT_ROOT}...")
    try:
        create_virtual_environment()
        install_dependencies()
        print(f"Environment setup completed successfully for {PROJECT_ROOT}.")
        print(f"To activate the virtual environment, navigate to {PROJECT_ROOT} and run:")
        if sys.platform == "win32":
            print(f"  .\\{VENV_DIR}\\Scripts\\activate")
        else:
            print(f"  source {VENV_DIR}/bin/activate")
        return 0 # Success
    except Exception as e:
        print(f"Environment setup failed: {e}", file=sys.stderr)
        return 1 # Failure

# --- Main Execution (for direct script running) ---

if __name__ == "__main__":
    print("Running src/setup/setup.py directly...")
    exit_code = perform_full_setup()
    if exit_code == 0:
        print("Direct execution of setup.py completed.")
    else:
        print("Direct execution of setup.py failed.", file=sys.stderr)
    sys.exit(exit_code) 