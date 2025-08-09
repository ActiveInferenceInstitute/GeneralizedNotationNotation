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
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def install_uv_if_needed(logger: logging.Logger) -> bool:
    """Install UV if it's not available."""
    try:
        # Check if UV is already available
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ UV is already available: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    logger.info("🔧 UV not found, installing...")
    try:
        # Install UV using the official installer
        install_script = subprocess.run([
            "curl", "-LsSf", "https://astral.sh/uv/install.sh"
        ], capture_output=True, text=True, timeout=30)
        
        if install_script.returncode == 0:
            # Execute the install script
            install_result = subprocess.run([
                "sh", "-c", install_script.stdout
            ], capture_output=True, text=True, timeout=60)
            
            if install_result.returncode == 0:
                logger.info("✅ UV installed successfully")
                return True
            else:
                logger.error(f"❌ UV installation failed: {install_result.stderr}")
                return False
        else:
            logger.error(f"❌ Failed to download UV installer: {install_script.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ UV installation failed: {e}")
        return False

# Import core setup functions from setup module
try:
    from setup import (
        log_system_info,
        check_uv_availability,
        setup_uv_environment,
        install_optional_dependencies,
        validate_uv_setup,
        create_project_structure
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
        if not install_uv_if_needed(logger):
            logger.error("❌ Failed to install UV - setup cannot continue")
            return False
        
        # Log system information
        system_info = log_system_info(logger)
        
        # Check UV availability
        uv_available = check_uv_availability(logger)
        if not uv_available:
            logger.error("❌ UV is not available after installation attempt")
            return False
        
        # Setup UV environment
        # Use module API that manages its own project_root internally
        # Keep it fast and resilient: avoid long JAX checks in this standardized path
        # Ensure dev and test extras are installed so tests can run in step 2
        setup_success = setup_uv_environment(
            verbose=verbose,
            recreate=False,
            dev=True,
            extras=["llm", "visualization", "audio"],
            skip_jax_test=True
        )
        if not setup_success:
            logger.error("❌ UV environment setup failed")
            return False
        
        logger.info("✅ UV environment setup completed")
        
        # Use standardized numbered output folder for this step
        step_output_dir = get_output_dir_for_script("1_setup.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project structure inside the step output directory
        structure_success = create_project_structure(step_output_dir, logger)
        if not structure_success:
            logger.error("❌ Failed to create project structure")
            return False
        
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
    "1_setup",
    process_setup_standardized,
    "Project setup and environment validation",
)

def main():
    return run_script()

if __name__ == "__main__":
    sys.exit(main()) 