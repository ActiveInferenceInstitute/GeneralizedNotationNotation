#!/usr/bin/env python3
"""
Step 5: Type Checking and Validation

This step performs type checking and validation on GNN files.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main type checking function."""
    args = EnhancedArgumentParser.parse_step_arguments("5_type_checker.py")
    
    # Setup logging
    logger = setup_step_logging("type_checker", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("5_type_checker.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run type checking from type_checker module
        from type_checker import validate_gnn_files
        
        log_step_start(logger, "Performing type checking and validation")
        
        success = validate_gnn_files(
            target_dir=args.target_dir,
            output_dir=output_dir,
            strict=args.strict,
            estimate_resources=args.estimate_resources,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "Type checking completed successfully")
            return 0
        else:
            log_step_error(logger, "Type checking failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
