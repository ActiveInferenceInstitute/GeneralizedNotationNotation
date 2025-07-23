#!/usr/bin/env python3
"""
Step 11: Render Processing

This step handles render processing for GNN files.
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
    """Main render processing function."""
    parser = EnhancedArgumentParser.parse_step_arguments("11_render")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_step_logging("render", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("11_render.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run render processing
        from render import process_render
        
        log_step_start("Processing render")
        
        success = process_render(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success("Render processing completed successfully")
            return 0
        else:
            log_step_error("Render processing failed")
            return 1
            
    except Exception as e:
        log_step_error("Render processing failed", {"error": str(e)})
        return 1

if __name__ == "__main__":
    sys.exit(main())
