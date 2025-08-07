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
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

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
except ImportError:
    SETUP_AVAILABLE = False
    # Fallback function definitions if setup module is not available
    def log_system_info(logger: logging.Logger) -> Dict[str, Any]:
        logger.error("Setup module not available")
        return {}
    
    def check_uv_availability(logger: logging.Logger) -> bool:
        logger.error("Setup module not available")
        return False
    
    def setup_uv_environment(project_root: Path, logger: logging.Logger) -> bool:
        logger.error("Setup module not available")
        return False
    
    def install_optional_dependencies(project_root: Path, logger: logging.Logger, package_groups: List[str] = None) -> bool:
        logger.error("Setup module not available")
        return False
    
    def validate_uv_setup(project_root: Path, logger: logging.Logger) -> Dict[str, Any]:
        logger.error("Setup module not available")
        return {}
    
    def create_project_structure(output_dir: Path, logger: logging.Logger) -> bool:
        logger.error("Setup module not available")
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
        output_dir: Output directory for setup results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Check if setup module is available
        if not SETUP_AVAILABLE:
            log_step_warning(logger, "Setup module not available, using fallback functions")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("1_setup.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing setup")
        
        # Log system information
        system_info = log_system_info(logger)
        
        # Check UV availability
        uv_available = check_uv_availability(logger)
        if not uv_available:
            log_step_error(logger, "UV is not available. Please install UV first.")
            return False
        
        # Setup UV environment
        project_root = Path(__file__).parent.parent.parent
        uv_success = setup_uv_environment(project_root, logger)
        if not uv_success:
            log_step_error(logger, "Failed to setup UV environment")
            return False
        
        # Install optional dependencies
        optional_deps_success = install_optional_dependencies(project_root, logger)
        if not optional_deps_success:
            log_step_warning(logger, "Failed to install some optional dependencies")
        
        # Validate UV environment
        validation_results = validate_uv_setup(project_root, logger)
        
        # Create project structure
        structure_success = create_project_structure(output_dir, logger)
        if not structure_success:
            log_step_warning(logger, "Failed to create some project structure")
        
        # Save setup results
        setup_results = {
            "timestamp": time.time(),
            "system_info": system_info,
            "uv_available": uv_available,
            "uv_setup_success": uv_success,
            "optional_deps_success": optional_deps_success,
            "validation_results": validation_results,
            "structure_success": structure_success,
            "setup_module_available": SETUP_AVAILABLE
        }
        
        setup_results_file = step_output_dir / "setup_results.json"
        with open(setup_results_file, 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        if uv_success:
            log_step_success(logger, "Setup processing completed successfully")
        else:
            log_step_error(logger, "Setup processing failed")
        
        return uv_success
        
    except Exception as e:
        log_step_error(logger, f"Setup processing failed: {e}")
        return False

def main():
    """Main setup processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("1_setup.py")
    
    # Setup logging
    logger = setup_step_logging("setup", args)
    
    # Check if setup module is available
    if not SETUP_AVAILABLE:
        log_step_warning(logger, "Setup module not available, using fallback functions")
    
    # Process setup
    success = process_setup_standardized(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 