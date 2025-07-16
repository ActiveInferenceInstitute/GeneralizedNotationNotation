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
import logging
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
from utils.pipeline_template import create_standardized_pipeline_script

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
        # Extract and resolve ontology_terms_file from kwargs
        ontology_terms_file = kwargs.get('ontology_terms_file')
        
        # If no ontology terms file provided, use default
        if ontology_terms_file is None:
            # Try to find default ontology terms file in expected locations
            script_dir = Path(__file__).parent
            default_paths = [
                script_dir / "ontology" / "act_inf_ontology_terms.json",
                script_dir.parent / "src" / "ontology" / "act_inf_ontology_terms.json",
                Path("src/ontology/act_inf_ontology_terms.json"),
                Path("ontology/act_inf_ontology_terms.json")
            ]
            
            for default_path in default_paths:
                if default_path.exists():
                    ontology_terms_file = default_path
                    logger.info(f"Using default ontology terms file: {ontology_terms_file}")
                    break
            
            if ontology_terms_file is None:
                logger.warning("No ontology terms file found in default locations")
        
        # Convert to Path if it's a string
        if isinstance(ontology_terms_file, str):
            ontology_terms_file = Path(ontology_terms_file)
        
        # Call the existing process_ontology_operations function
        success = process_ontology_operations(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            recursive=recursive,
            ontology_terms_file=ontology_terms_file
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "8_ontology.py",
    process_ontology_operations_standardized,
    "Ontology processing and validation"
)

if __name__ == '__main__':
    sys.exit(run_script()) 