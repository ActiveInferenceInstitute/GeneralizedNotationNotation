#!/usr/bin/env python3
"""
Step 12: Execute Processing

This step handles execute processing for GNN files.
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
    """Main execute processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("12_execute")
    
    # Setup logging
    logger = setup_step_logging("execute", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("12_execute.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing execute")
        
        # Check for rendered PyMDP scripts from step 11
        render_output_dir = get_output_dir_for_script("11_render.py", Path(args.output_dir))
        pymdp_scripts = list(render_output_dir.glob("**/*_pymdp*.py"))
        
        success = False
        
        if pymdp_scripts:
            # Execute PyMDP simulations
            logger.info(f"Found {len(pymdp_scripts)} PyMDP scripts to execute")
            
            from execute.pymdp import execute_pymdp_simulation
            from gnn import parse_gnn_file
            
            executed_count = 0
            
            for script_path in pymdp_scripts:
                try:
                    logger.info(f"Executing PyMDP script: {script_path}")
                    
                    # Try to find corresponding GNN file to get full spec
                    gnn_files = list(Path(args.target_dir).glob("*.md"))
                    
                    if gnn_files:
                        # Parse first GNN file as example
                        gnn_spec = parse_gnn_file(gnn_files[0])
                        
                        if gnn_spec:
                            # Create execution output directory  
                            exec_output_dir = output_dir / f"pymdp_execution_{script_path.stem}"
                            
                            # Execute using pipeline PyMDP module
                            exec_success, results = execute_pymdp_simulation(
                                gnn_spec=gnn_spec.to_dict(),
                                output_dir=exec_output_dir
                            )
                            
                            if exec_success:
                                executed_count += 1
                                logger.info(f"✓ Successfully executed {script_path.name}")
                            else:
                                logger.error(f"✗ Failed to execute {script_path.name}: {results.get('error')}")
                        else:
                            logger.warning(f"Could not parse GNN file for {script_path}")
                    else:
                        logger.warning(f"No GNN files found for {script_path}")
                        
                except Exception as e:
                    logger.error(f"Error executing {script_path}: {e}")
            
            success = executed_count > 0
            
            if success:
                log_step_success(logger, f"Executed {executed_count} PyMDP simulations successfully")
            else:
                log_step_error(logger, "No PyMDP simulations executed successfully")
        else:
            # Fallback to generic execute processing
            try:
                from execute import process_execute
                
                success = process_execute(
                    target_dir=args.target_dir,
                    output_dir=output_dir,
                    verbose=args.verbose
                )
            except ImportError:
                logger.warning("No rendered scripts found and generic execute module not available")
                success = True  # Don't fail if no execution needed
        
        return 0 if success else 1
            
    except Exception as e:
        log_step_error(logger, "Execute processing failed", {"error": str(e)})
        return 1

if __name__ == "__main__":
    sys.exit(main())
