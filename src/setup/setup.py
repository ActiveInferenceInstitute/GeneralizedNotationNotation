"""
Setup script for the GNN project with UV support.

This script handles the creation of a UV environment and the installation
of project dependencies using Python packaging standards.

Functions are organized into sub-modules:
- uv_management: UV environment creation, dependency sync, validation
- dependency_setup: JAX testing, Julia setup, optional package groups
"""

import sys
import logging
import argparse
import time
from pathlib import Path

# --- Re-export everything from sub-modules for backward compatibility ---
from .uv_management import (
    # Constants
    VENV_DIR,
    PYPROJECT_FILE,
    LOCK_FILE,
    PROJECT_ROOT,
    VENV_PATH,
    PYPROJECT_PATH,
    LOCK_PATH,
    VENV_PYTHON,
    MIN_PYTHON_VERSION,
    OPTIONAL_GROUPS,
    # Functions
    run_command,
    check_system_requirements,
    check_uv_availability,
    create_uv_environment,
    install_uv_dependencies,
    get_installed_package_versions,
    add_uv_dependency,
    remove_uv_dependency,
    update_uv_dependencies,
    lock_uv_dependencies,
    setup_uv_environment,
    validate_uv_setup,
    get_uv_setup_info,
    check_environment_health,
    cleanup_uv_setup,
    save_setup_results,
    log_system_info,
)

from .dependency_setup import (
    install_jax_and_test,
    setup_julia_environment,
    install_optional_package_group,
    install_all_optional_packages,
    install_optional_dependencies,
    create_project_structure,
    setup_gnn_project,
    setup_complete_environment,
)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---


# --- Callable Main Function ---
def perform_full_setup(verbose: bool = False, recreate_venv: bool = False, dev: bool = False,
                      extras: list = None, skip_jax_test: bool = False):
    """
    Performs the full setup using UV: creates environment and installs dependencies.
    This function is intended to be called by other scripts.

    Args:
        verbose (bool): If True, enables detailed (DEBUG level) logging for this setup process.
        recreate_venv (bool): If True, recreates the UV environment even if it already exists.
        dev (bool): If True, also installs development dependencies.
        extras (list): List of optional dependency groups to install.
        skip_jax_test (bool): If True, skips JAX/Optax/Flax installation testing (faster setup).

    Returns:
        int: 0 if successful, 1 if failed
    """
    # Configure logger for this module based on verbosity passed from caller
    log_level_to_set = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level_to_set)

    # Ensure we have a console handler with a clear format
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.propagate = False

    start_time = time.time()
    logger.info("🚀 Starting UV environment setup...")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"⚙️ Configuration: verbose={verbose}, recreate_venv={recreate_venv}, dev={dev}, extras={extras}, skip_jax_test={skip_jax_test}")
    sys.stdout.flush()

    try:
        # Phase 1: System Requirements
        logger.info("\n📋 Phase 1/3: Checking system requirements...")
        sys.stdout.flush()
        if not check_system_requirements(verbose):
            logger.error("❌ System requirements check failed")
            sys.stdout.flush()
            return 1
        logger.info("✅ System requirements check passed")
        sys.stdout.flush()

        # Phase 2: UV Environment
        logger.info("\n📋 Phase 2/3: Setting up UV environment...")
        sys.stdout.flush()
        venv_start = time.time()
        if not create_uv_environment(verbose, recreate_venv):
            logger.error("❌ Failed to create UV environment")
            sys.stdout.flush()
            return 1
        venv_duration = time.time() - venv_start
        logger.info(f"✅ UV environment setup completed in {venv_duration:.1f}s")
        sys.stdout.flush()

        # Phase 3: Dependencies
        logger.info("\n📋 Phase 3/3: Installing dependencies using UV...")
        logger.info("⏳ This may take several minutes...")
        sys.stdout.flush()
        deps_start = time.time()

        if not install_uv_dependencies(verbose, dev, extras):
            logger.error("❌ Failed to install dependencies using UV")
            sys.stdout.flush()
            return 1

        deps_duration = time.time() - deps_start
        logger.info(f"✅ Dependencies installed using UV in {deps_duration:.1f}s")
        sys.stdout.flush()

        # After dependency install, ensure JAX/Optax/Flax are present and working
        if not skip_jax_test:
            if not install_jax_and_test(verbose=verbose):
                logger.warning("JAX/Optax/Flax installation or self-test failed, but continuing setup.")
        else:
            logger.warning("JAX/Optax/Flax testing was skipped. JAX functionality may not be available.")

        total_duration = time.time() - start_time
        logger.info("\n🎉 UV setup completed successfully!")
        logger.info(f"⏱️ Total time: {total_duration:.1f}s")
        logger.info("\nTo activate the UV environment:")
        if sys.platform == "win32":
            logger.info(f"  .\\{VENV_DIR}\\Scripts\\activate")
        else:
            logger.info(f"  source {VENV_DIR}/bin/activate")
        logger.info("\nTo run commands in the UV environment:")
        logger.info("  uv run python src/main.py --help")
        logger.info("  uv run pytest src/tests/")
        sys.stdout.flush()
        return 0

    except Exception as e:
        logger.error(f"❌ UV setup failed: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.stdout.flush()
        return 1

# --- Main Execution (for direct script running) ---

if __name__ == "__main__":
    # Basic argument parsing for direct execution
    parser = argparse.ArgumentParser(description="Direct execution of GNN project setup script with UV (environment and dependencies).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG level) logging.")
    parser.add_argument("--recreate-venv", action="store_true", help="Recreate UV environment even if it already exists.")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies.")
    parser.add_argument("--extras", nargs="+", help="Install optional dependency groups (e.g., ml-ai llm visualization).")
    cli_args = parser.parse_args()

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG if cli_args.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

    logger.info("Running src/setup/setup.py directly with UV...")
    exit_code = perform_full_setup(
        verbose=cli_args.verbose,
        recreate_venv=cli_args.recreate_venv,
        dev=cli_args.dev,
        extras=cli_args.extras
    )
    if exit_code == 0:
        logger.info("Direct execution of setup.py with UV completed.")
    else:
        logger.error("Direct execution of setup.py with UV failed.")
    sys.exit(exit_code)
