#!/usr/bin/env python3
"""
Pipeline Module Template

This template shows the recommended structure for all GNN pipeline modules.
Copy this structure for consistent argument handling, logging, and error management.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Any

# Standard import pattern for all pipeline modules
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        validate_output_directory,
        get_pipeline_utilities,
        UTILS_AVAILABLE
    )
    
    # Enhanced imports for more complex modules
    from utils import (
        EnhancedArgumentParser,
        PipelineArguments,
        PerformanceTracker,
        performance_tracker
    )
    
except ImportError as e:
    # Fallback logging setup
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create minimal compatibility functions
    def setup_step_logging(name: str, verbose: bool = False):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
    
    def log_step_start(logger, message): logger.info(f"🚀 {message}")
    def log_step_success(logger, message): logger.info(f"✅ {message}")
    def log_step_warning(logger, message): logger.warning(f"⚠️ {message}")
    def log_step_error(logger, message): logger.error(f"❌ {message}")
    def validate_output_directory(output_dir, step_name): 
        try:
            (output_dir / f"{step_name}_step").mkdir(parents=True, exist_ok=True)
            return True
        except:
            return False
    
    UTILS_AVAILABLE = False

# Initialize logger for this step
logger = setup_step_logging("template_step", verbose=False)

def process_step_logic(args: argparse.Namespace) -> bool:
    """
    Main processing logic for the pipeline step.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    # Performance tracking (if available)
    if UTILS_AVAILABLE:
        with performance_tracker.track_operation("step_processing"):
            return _do_processing(args)
    else:
        return _do_processing(args)

def _do_processing(args: argparse.Namespace) -> bool:
    """Internal processing function."""
    log_step_start(logger, f"Processing with target_dir: {args.target_dir}")
    
    try:
        # Validate output directory
        if not validate_output_directory(Path(args.output_dir), "template"):
            log_step_error(logger, "Failed to create output directory")
            return False
        
        # Your processing logic here
        # ...
        
        log_step_success(logger, "Processing completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Processing failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return False

def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main function for the pipeline step.
    
    Args:
        args: Pre-parsed arguments (from main.py) or None for standalone
        
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    # Update logger verbosity if available
    if UTILS_AVAILABLE and args and hasattr(args, 'verbose') and args.verbose:
        from utils import PipelineLogger
        PipelineLogger.set_verbosity(True)
    
    # Parse arguments if not provided (standalone execution)
    if args is None:
        args = parse_standalone_arguments()
    
    # Execute processing
    success = process_step_logic(args)
    return 0 if success else 1

def parse_standalone_arguments() -> argparse.Namespace:
    """Parse arguments for standalone execution."""
    if UTILS_AVAILABLE:
        # Use enhanced argument parser
        return EnhancedArgumentParser.parse_step_arguments("template_step")
    else:
        # Fallback basic parser
        parser = argparse.ArgumentParser(description="Template Pipeline Step")
        
        # Standard pipeline arguments
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        
        parser.add_argument("--target-dir", type=Path, 
                          default=project_root / "src" / "gnn" / "examples",
                          help="Target directory for input files")
        parser.add_argument("--output-dir", type=Path,
                          default=project_root / "output", 
                          help="Output directory")
        parser.add_argument("--recursive", action="store_true",
                          help="Process files recursively")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose logging")
        
        return parser.parse_args()

if __name__ == "__main__":
    sys.exit(main()) 