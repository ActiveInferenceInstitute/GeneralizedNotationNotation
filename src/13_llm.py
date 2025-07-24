#!/usr/bin/env python3
"""
Step 13: LLM Processing

This step handles LLM processing for GNN files.
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
    """Main LLM processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("13_llm")
    
    # Setup logging
    logger = setup_step_logging("llm", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("13_llm.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run LLM processing
        from llm import process_llm
        
        log_step_start(logger, "Processing LLM")
        
        success = process_llm(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "LLM processing completed successfully")
            return 0
        else:
            log_step_error(logger, "LLM processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, "LLM processing failed", {"error": str(e)})
        return 1

if __name__ == "__main__":
    sys.exit(main())
