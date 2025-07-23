#!/usr/bin/env python3
"""
Step 7: Multi-format Export Generation

This step generates exports in multiple formats (JSON, XML, GraphML, GEXF, Pickle).
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
    """Main export generation function."""
    args = EnhancedArgumentParser.parse_step_arguments("7_export.py")
    
    # Setup logging
    logger = setup_step_logging("export", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("7_export.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run export from export module
        from export import generate_exports
        
        log_step_start(logger, "Generating multi-format exports")
        
        success = generate_exports(
            target_dir=args.target_dir,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "Exports generated successfully")
            return 0
        else:
            log_step_error(logger, "Export generation failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Export generation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
