#!/usr/bin/env python3
"""
Step 10: Ontology Processing (Thin Orchestrator)

This step orchestrates ontology processing and validation for GNN models.
It is a thin orchestrator that delegates core functionality to the ontology module.

How to run:
  python src/10_ontology.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Ontology processing results in the specified output directory
  - Ontology validation and compliance reports
  - Term mapping and relationship analysis
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that ontology dependencies are installed
  - Check that src/ontology/ contains ontology modules
  - Check that the output directory is writable
  - Verify ontology configuration and term mapping setup
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
from utils import performance_tracker
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import the ontology processor and its methods
try:
    from ontology import (
        process_ontology,
        extract_ontology_terms,
        validate_ontology_compliance,
        generate_ontology_mapping,
        process_ontology_file,
        generate_ontology_report,
        process_ontology_fallback
    )
    ONTOLOGY_AVAILABLE = True
except ImportError:
    ONTOLOGY_AVAILABLE = False

def process_ontology_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized ontology processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for ontology results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("ontology_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Get configuration
            config = get_pipeline_config()
            step_config = config.get_step_config("10_ontology.py")
            
            # Set up output directory
            step_output_dir = get_output_dir_for_script("10_ontology.py", output_dir)
            step_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log processing parameters
            logger.info(f"Processing GNN files from: {target_dir}")
            logger.info(f"Output directory: {step_output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Extract ontology-specific parameters
            ontology_terms_file = kwargs.get('ontology_terms_file', None)
            
            if ontology_terms_file:
                logger.info(f"Ontology terms file: {ontology_terms_file}")
            
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
            
            logger.info(f"Found {len(gnn_files)} GNN files to process")
            
            # Process ontology if available
            if ONTOLOGY_AVAILABLE:
                logger.info("Ontology module available, processing files...")
                
                # Process each file
                processed_files = []
                ontology_results = []
                
                for gnn_file in gnn_files:
                    try:
                        logger.debug(f"Processing ontology for: {gnn_file}")
                        
                        # Process ontology for this file using module function
                        file_result = process_ontology_file(gnn_file, step_output_dir, logger)
                        
                        if file_result:
                            processed_files.append(gnn_file)
                            ontology_results.append(file_result)
                        
                    except Exception as e:
                        log_step_error(logger, f"Failed to process ontology for {gnn_file}: {e}")
                        continue
                
                # Generate comprehensive ontology report using module function
                if processed_files:
                    generate_ontology_report(processed_files, ontology_results, step_output_dir, logger)
                    log_step_success(logger, f"Successfully processed ontology for {len(processed_files)} files")
                else:
                    log_step_warning(logger, "No files were successfully processed")
                
                return len(processed_files) > 0
            else:
                logger.warning("Ontology module not available, using fallback processing")
                return process_ontology_fallback(gnn_files, step_output_dir, logger)
                
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False

def main():
    """Main ontology processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("10_ontology")
    
    # Setup logging
    logger = setup_step_logging("ontology", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("10_ontology.py", Path(args.output_dir))
        
        # Process using standardized pattern
        success = process_ontology_standardized(
            target_dir=Path(args.target_dir) if hasattr(args, 'target_dir') else Path("input/gnn_files"),
            output_dir=output_dir,
            logger=logger,
            recursive=getattr(args, 'recursive', False),
            verbose=getattr(args, 'verbose', False),
            ontology_terms_file=getattr(args, 'ontology_terms_file', None)
        )
        
        return 0 if success else 1
        
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
