#!/usr/bin/env python3
"""
Step 22: Model Context Protocol (MCP) Processing

This step handles Model Context Protocol processing for GNN files.
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
    """Main MCP processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("22_mcp")
    
    # Setup logging
    logger = setup_step_logging("mcp", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("22_mcp.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run MCP processing
        from mcp.processor import process_mcp_operations
        
        log_step_start(logger, "Processing MCP operations")
        
        success = process_mcp_operations(
            target_dir=args.target_dir,
            output_dir=output_dir,
            logger=logger,
            verbose=args.verbose
        )
        
        if success:
            log_step_success(logger, "MCP processing completed successfully")
            return 0
        else:
            log_step_error(logger, "MCP processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 