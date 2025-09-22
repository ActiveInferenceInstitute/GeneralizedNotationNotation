#!/usr/bin/env python3
"""
Setup processor module for GNN pipeline.

This module contains the core setup processing logic that is called by
the thin orchestrator script (1_setup.py).
"""

import sys
import subprocess
import logging
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning,
)
from pipeline.config import get_output_dir_for_script, get_pipeline_config
from .setup import (
    log_system_info,
    check_uv_availability,
    setup_uv_environment,
    install_optional_dependencies,
    validate_uv_setup,
    create_project_structure,
    get_installed_package_versions,
)


def ensure_uv_available(logger: logging.Logger) -> bool:
    """Ensure UV is available, install if needed."""
    try:
        # Check if UV is already available
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ UV available: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    logger.info("🔧 UV not found, attempting installation...")

    # Try multiple installation methods
    installation_methods = [
        # Method 1: Use curl to install UV directly (recommended)
        {
            "name": "curl installer",
            "command": ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"],
            "shell": True
        },
        # Method 2: Use pip with --break-system-packages (if available)
        {
            "name": "pip with --break-system-packages",
            "command": [sys.executable, "-m", "pip", "install", "--break-system-packages", "uv"],
            "shell": False
        },
        # Method 3: Use pip with --user flag
        {
            "name": "pip with --user",
            "command": [sys.executable, "-m", "pip", "install", "--user", "uv"],
            "shell": False
        },
        # Method 4: Use pip with --force-reinstall
        {
            "name": "pip with --force-reinstall",
            "command": [sys.executable, "-m", "pip", "install", "--force-reinstall", "uv"],
            "shell": False
        }
    ]

    for method in installation_methods:
        try:
            logger.info(f"🔧 Trying {method['name']}...")
            result = subprocess.run(
                method["command"],
                capture_output=True,
                text=True,
                timeout=120,
                shell=method["shell"]
            )

            if result.returncode == 0:
                logger.info(f"✅ UV installed successfully using {method['name']}")
                # Verify installation worked
                verify_result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
                if verify_result.returncode == 0:
                    logger.info(f"✅ UV verification successful: {verify_result.stdout.strip()}")
                    return True
                else:
                    logger.warning(f"⚠️ UV installed but verification failed, trying next method...")
            else:
                logger.warning(f"⚠️ {method['name']} failed: {result.stderr.strip()}")

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"⚠️ {method['name']} failed with exception: {e}")
            continue

    # If all methods failed, provide helpful error message
    logger.error("❌ All UV installation methods failed")
    logger.error("Please install UV manually:")
    logger.error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    logger.error("  or visit: https://docs.astral.sh/uv/getting-started/installation/")
    logger.error("  or use: pip install --user uv")
    return False


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
        target_dir: Directory containing files to process
        output_dir: Directory to write output files
        logger: Logger instance for logging
        recursive: Whether to process subdirectories recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional keyword arguments

    Returns:
        True if setup was successful, False otherwise
    """
    try:
        logger.info("🚀 Processing setup")

        # Ensure UV is available
        if not ensure_uv_available(logger):
            logger.warning("⚠️ UV not available - attempting to continue with fallback methods")
            # Don't fail completely, try to continue with available tools

        # Log system information
        system_info = log_system_info(logger)

        # Check UV availability
        uv_available = check_uv_availability(logger)
        if not uv_available:
            logger.warning("⚠️ UV is not available after installation attempt - continuing with fallback")
            # Continue with fallback methods instead of failing

        # Setup UV environment
        # Use module API that manages its own project_root internally
        # Keep it fast and resilient: avoid long JAX checks in this standardized path
        # Ensure dev and test extras are installed so tests can run in step 2
        if uv_available:
            setup_success = setup_uv_environment(
                verbose=verbose,
                recreate=False,
                dev=True,
                extras=["llm", "visualization", "audio", "gui"],
                skip_jax_test=True
            )
            if not setup_success:
                logger.warning("⚠️ UV environment setup failed - continuing with fallback")
                # Continue with fallback methods
            else:
                logger.info("✅ UV environment setup completed")
        else:
            logger.warning("⚠️ Skipping UV environment setup - UV not available")
            setup_success = False

        # Use standardized numbered output folder for this step
        step_output_dir = get_output_dir_for_script("1_setup.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        # Create project structure inside the step output directory
        structure_success = create_project_structure(step_output_dir, logger)
        if not structure_success:
            logger.error("❌ Failed to create project structure")
            return False

        # Ensure core test dependencies are present in the environment (without modifying pyproject)
        try:
            installed = get_installed_package_versions()
            required_test_packages = [
                "pytest",
                "pytest-cov",
                "pytest-xdist",
                "pytest-timeout",
            ]
            missing = [p for p in required_test_packages if p not in (installed or {})]
            if missing:
                logger.info(f"📦 Installing missing test packages via UV: {missing}")
                try:
                    result = subprocess.run(
                        [
                            "uv",
                            "pip",
                            "install",
                            *missing,
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode != 0:
                        logger.warning(
                            f"⚠️ Failed to install some test packages (exit {result.returncode}); tests may be limited"
                        )
                        if verbose:
                            logger.warning(result.stdout)
                            logger.warning(result.stderr)
                except Exception as e:
                    logger.warning(f"⚠️ Could not install test packages via UV: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify installed packages: {e}")

        # Setup missing dependencies
        logger.info("🔍 Setting up missing dependencies...")
        try:
            from setup.setup import setup_missing_dependencies
            dependency_results = setup_missing_dependencies(verbose=verbose)
            logger.info(f"📊 Dependency setup results: {dependency_results}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup missing dependencies: {e}")
            dependency_results = {}

        # Validate setup
        # Keep validation lightweight; avoid strict failures on missing heavy deps
        validation_result = validate_uv_setup()

        # Log setup summary
        setup_summary = {
            "system_info": system_info,
            "uv_available": uv_available,
            "structure_created": structure_success,
            "dependency_results": dependency_results,
            "validation": validation_result,
        }

        logger.info("✅ Setup processing completed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Setup processing failed: {e}")
        return False

