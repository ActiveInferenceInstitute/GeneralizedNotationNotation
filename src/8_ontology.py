#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 8: Ontology Processing

This script handles ontology-related operations:
- Processes Active Inference Ontology annotations
- Validates ontology mappings in GNN files
- Generates ontology analysis reports

Usage:
    python 8_ontology.py [options]
    (Typically called by main.py)
"""

import sys
from pathlib import Path

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    UTILS_AVAILABLE
)

from pipeline import (
    STEP_METADATA,
    get_output_dir_for_script
)

from ontology.processor import process_ontology_operations

# Initialize logger for this step
logger = setup_step_logging("8_ontology", verbose=False)

# Attempt to import MCP functionalities from the ontology module
try:
    from ontology import mcp as ontology_mcp
    logger.debug("Successfully imported ontology MCP module")
except ImportError as e:
    log_step_error(logger, f"Could not import 'mcp' from src/ontology/mcp.py: {e}")
    logger.error("Ensure src/ontology/mcp.py exists and src/ is discoverable.")
    ontology_mcp = None

def process_ontology_operations_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized ontology operations processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for ontology results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options (ontology_terms_file, etc.)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Call the existing process_ontology_operations function
        success = process_ontology_operations(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            recursive=recursive
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False

def main(parsed_args):
    """Main function for ontology processing."""
    
    # Log step metadata from centralized configuration
    step_info = STEP_METADATA.get("8_ontology.py", {})
    log_step_start(logger, f"{step_info.get('description', 'Ontology processing and validation')}")
    
    # Update logger verbosity if needed
    if parsed_args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get ontology terms file
    ontology_terms_file = None
    if hasattr(parsed_args, 'ontology_terms_file') and parsed_args.ontology_terms_file:
        ontology_terms_file = Path(parsed_args.ontology_terms_file)
    
    # Process ontology operations
    success = process_ontology_operations_standardized(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        logger=logger,
        recursive=getattr(parsed_args, 'recursive', False),
        verbose=getattr(parsed_args, 'verbose', False),
        ontology_terms_file=ontology_terms_file
    )
    
    if success:
        log_step_success(logger, "Ontology processing completed successfully")
        return 0
    else:
        log_step_error(logger, "Ontology processing failed")
        return 1

if __name__ == '__main__':
    # Use centralized argument parsing
    if UTILS_AVAILABLE:
        parsed_args = EnhancedArgumentParser.parse_step_arguments("8_ontology")
    else:
        # Fallback argument parsing
        import argparse
        parser = argparse.ArgumentParser(description="Ontology processing and validation")
        parser.add_argument("--target-dir", type=Path, required=True,
                          help="Target directory containing GNN files")
        parser.add_argument("--output-dir", type=Path, required=True,
                          help="Output directory for generated artifacts")
        parser.add_argument("--recursive", action="store_true",
                          help="Search recursively in subdirectories")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--ontology-terms-file", type=Path,
                          help="Path to ontology terms JSON file")
        parsed_args = parser.parse_args()
    
    exit_code = main(parsed_args)
    sys.exit(exit_code) 