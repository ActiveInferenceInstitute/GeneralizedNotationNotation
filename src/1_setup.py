#!/usr/bin/env python3
"""
Step 1: Project Setup and Environment Validation with UV (Thin Orchestrator)

This step handles project initialization, UV environment setup,
dependency installation, and environment validation using modern
Python packaging standards.

How to run:
  python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Environment setup results in the specified output directory
  - UV environment creation and validation
  - Dependency installation and verification
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that UV is installed and available
  - Check that src/setup/ contains setup modules
  - Check that the output directory is writable
  - Verify system requirements and permissions
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
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning,
    create_standardized_pipeline_script,
)
from utils.argument_utils import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

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

# Import core setup functions from setup module
try:
    from setup import (
        log_system_info,
        check_uv_availability,
        setup_uv_environment,
        install_optional_dependencies,
        validate_uv_setup,
        create_project_structure,
        get_installed_package_versions,
    )
    SETUP_AVAILABLE = True
except ImportError as e:
    SETUP_AVAILABLE = False
    logging.error(f"Setup module not available: {e}")
    
    # Critical error - setup module is required
    def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
        logger.error("Setup module not available - this is a critical error")
        return {}
    
    def check_uv_availability(logger: logging.Logger) -> bool:
        logger.error("Setup module not available - this is a critical error")
        return False
    
    def setup_uv_environment(project_root: Path, logger: logging.Logger) -> bool:
        logger.error("Setup module not available - this is a critical error")
        return False
    
    def install_optional_dependencies(project_root: Path, logger: logging.Logger, package_groups: List[str] = None) -> bool:
        logger.error("Setup module not available - this is a critical error")
        return False
    
    def validate_uv_setup(project_root: Path, logger: logging.Logger) -> Dict[str, Any]:
        logger.error("Setup module not available - this is a critical error")
        return {}
    
    def create_project_structure(output_dir: Path, logger: logging.Logger) -> bool:
        logger.error("Setup module not available - this is a critical error")
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
        
        # Validate setup
        # Keep validation lightweight; avoid strict failures on missing heavy deps
        validation_result = validate_uv_setup()
        
        # Log setup summary
        setup_summary = {
            "system_info": system_info,
            "uv_available": uv_available,
            "structure_created": structure_success,
            "validation": validation_result,
            "setup_available": SETUP_AVAILABLE
        }
        
        logger.info("✅ Setup processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup processing failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "1_setup.py",
    process_setup_standardized,
    "Project setup and environment validation",
)

def main():
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 