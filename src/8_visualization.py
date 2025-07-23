#!/usr/bin/env python3
"""
Step 8: Visualization Processing

This step handles visualization processing for GNN files.
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
    """Main visualization processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("8_visualization.py")
    
    # Setup logging
    logger = setup_step_logging("visualization", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("8_visualization.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run visualization processing
        from visualization import process_visualization
        
        log_step_start(logger, "Processing visualization")
        
        success = process_visualization(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "Visualization processing completed successfully")
            return 0
        else:
            log_step_error(logger, "Visualization processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
