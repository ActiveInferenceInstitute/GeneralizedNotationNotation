#!/usr/bin/env python3
"""
Step 20: Website Generation (Thin Orchestrator)

This step orchestrates static HTML website generation from pipeline artifacts.
It is a thin orchestrator that delegates core functionality to the website module.

How to run:
  python src/20_website.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Static HTML website in the specified output directory
  - Interactive visualizations and documentation
  - Pipeline summary and results presentation
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that website dependencies are installed
  - Check that src/website/ contains website modules
  - Check that the output directory is writable
  - Verify website template configuration and asset setup
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import the website processor
from website import process_website

def process_website_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized website generation function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for website files
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Update logger verbosity if needed
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Get configuration
        config = get_pipeline_config()
        step_config = config.get_step_config("20_website") if hasattr(config, 'get_step_config') else None
        
        # Set up output directory
        step_output_dir = get_output_dir_for_script("20_website.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log processing parameters
        logger.info(f"Processing GNN files from: {target_dir}")
        logger.info(f"Output directory: {step_output_dir}")
        logger.info(f"Recursive processing: {recursive}")
        
        # Extract website-specific parameters
        website_html_filename = kwargs.get('website_html_filename', 'gnn_pipeline_summary_website.html')
        
        logger.info(f"Website HTML filename: {website_html_filename}")
        
        # Validate input directory
        if not target_dir.exists():
            log_step_error(logger, f"Input directory does not exist: {target_dir}")
            return False
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True  # Not an error, just no files to process
        
        logger.info(f"Found {len(gnn_files)} GNN files to process for website generation")
        
        # Process files using the website module
        success = process_website(
            target_dir=target_dir,
            output_dir=step_output_dir,
            verbose=verbose,
            **kwargs
        )
        
        if success:
            log_step_success(logger, f"Successfully generated website for {len(gnn_files)} GNN models")
        else:
            log_step_error(logger, "Website generation failed")
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Website generation failed: {e}")
        if verbose:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main website generation function."""
    args = EnhancedArgumentParser.parse_step_arguments("20_website")
    
    # Setup logging
    logger = setup_step_logging("website", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        # Pass the root output directory here; the standardized function will derive the step directory.
        output_dir = Path(args.output_dir)
        
        # Process using standardized pattern
        success = process_website_standardized(
            target_dir=Path(args.target_dir) if hasattr(args, 'target_dir') else Path("input/gnn_files"),
            output_dir=output_dir,
            logger=logger,
            recursive=getattr(args, 'recursive', False),
            verbose=getattr(args, 'verbose', False),
            website_html_filename=getattr(args, 'website_html_filename', 'gnn_pipeline_summary_website.html')
        )
        
        return 0 if success else 1
        
    except Exception as e:
        log_step_error(logger, f"Website generation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
