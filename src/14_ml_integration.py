#!/usr/bin/env python3
"""
Step 14: ML Integration

This step handles machine learning integration and model training.
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
    """Main ML integration function."""
    args = EnhancedArgumentParser.parse_step_arguments("14_ml_integration.py")
    
    # Setup logging
    logger = setup_step_logging("ml_integration", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("14_ml_integration.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "ML integration processing")
        
        # Import and call ML integration from ml_integration module
        from ml_integration import process_ml_integration
        
        # Call the actual ML integration function
        success = process_ml_integration(
            target_dir=args.target_dir,
            output_dir=output_dir,
            recursive=args.recursive,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "ML integration completed successfully")
            return 0
        else:
            log_step_error(logger, "ML integration failed")
            return 1
        
    except Exception as e:
        log_step_error(logger, f"ML integration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 