#!/usr/bin/env python3
"""
Step 3: GNN File Discovery and Parsing

This step discovers and parses GNN files from the target directory.
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
    """Main GNN processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("3_gnn.py")
    
    # Setup logging
    logger = setup_step_logging("gnn_processing", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("3_gnn.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import and run GNN processing from gnn module
        from gnn import process_gnn_directory
        
        log_step_start(logger, "Processing GNN files")
        
        # Call the actual GNN processing function
        results = process_gnn_directory(
            directory=args.target_dir,
            recursive=args.recursive
        )
        
        # Save results to output directory
        results_file = output_dir / "gnn_processing_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Determine success based on results
        success = len(results.get('errors', [])) == 0 and results.get('total_files', 0) > 0
        
        if success:
            log_step_success(logger, "GNN files processed successfully")
            return 0
        else:
            log_step_error(logger, "GNN file processing failed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"GNN processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 