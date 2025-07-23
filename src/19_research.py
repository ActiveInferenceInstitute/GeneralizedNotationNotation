#!/usr/bin/env python3
"""
Step 19: Research Processing

This step handles research processing for GNN files.
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
    """Main research processing function."""
    parser = EnhancedArgumentParser.parse_step_arguments("19_research")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_step_logging("research", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("19_research.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run research processing
        from research import process_research
        
        log_step_start("Processing research")
        
        success = process_research(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success("Research processing completed successfully")
            return 0
        else:
            log_step_error("Research processing failed")
            return 1
            
    except Exception as e:
        log_step_error("Research processing failed", {"error": str(e)})
        return 1

if __name__ == "__main__":
    sys.exit(main())
