#!/usr/bin/env python3
"""
Step 16: Analysis Processing

This step handles analysis processing for GNN files.
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
    """Main analysis processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("16_analysis")
    
    # Setup logging
    logger = setup_step_logging("analysis", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("16_analysis.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run analysis processing
        from analysis import process_analysis
        
        log_step_start(logger, "Processing analysis")
        
        success = process_analysis(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "Analysis processing completed successfully")
            return 0
        else:
            log_step_error(logger, "Analysis processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Analysis processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
