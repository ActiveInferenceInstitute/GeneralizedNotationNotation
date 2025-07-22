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
import json
import datetime

# Import centralized utilities and configuration
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    validate_output_directory,
    EnhancedArgumentParser,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
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
        target_dir: Directory containing GNN files
        output_dir: Output directory for ontology results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if ontology processing succeeded, False otherwise
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("ontology_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Setup output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load ontology terms
            ontology_terms_file = kwargs.get('ontology_terms_file')
            if ontology_terms_file and Path(ontology_terms_file).exists():
                with open(ontology_terms_file, 'r') as f:
                    ontology_terms = json.load(f)
                logger.info(f"Loaded {len(ontology_terms)} ontology terms")
            else:
                logger.warning("No ontology terms file found, using minimal defaults")
                ontology_terms = {
                    "active_inference": "Active Inference framework concepts",
                    "pomdp": "Partially Observable Markov Decision Process",
                    "belief_state": "Agent's belief about hidden states"
                }
            
            # Process GNN files for ontology mapping
            gnn_files = []
            if recursive:
                gnn_files = list(target_dir.rglob("*.md")) + list(target_dir.rglob("*.gnn"))
            else:
                gnn_files = list(target_dir.glob("*.md")) + list(target_dir.glob("*.gnn"))
            
            logger.info(f"Processing {len(gnn_files)} files for ontology mapping")
            
            # Generate ontology mapping results
            ontology_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "files_processed": len(gnn_files),
                "ontology_terms_count": len(ontology_terms),
                "mappings": [],
                "coverage_analysis": {
                    "files_with_mappings": 0,
                    "total_mappings_found": 0,
                    "unmapped_concepts": []
                }
            }
            
            # Analyze each file for ontology concepts
            files_with_mappings = 0
            total_mappings = 0
            
            for gnn_file in gnn_files:
                try:
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    file_mappings = []
                    for term, description in ontology_terms.items():
                        if term.lower() in content:
                            file_mappings.append({
                                "term": term,
                                "description": description,
                                "file": str(gnn_file.relative_to(target_dir))
                            })
                            total_mappings += 1
                    
                    if file_mappings:
                        files_with_mappings += 1
                        ontology_results["mappings"].extend(file_mappings)
                        
                except Exception as e:
                    logger.warning(f"Could not process {gnn_file}: {e}")
            
            # Update coverage analysis
            ontology_results["coverage_analysis"]["files_with_mappings"] = files_with_mappings
            ontology_results["coverage_analysis"]["total_mappings_found"] = total_mappings
            
            # Save ontology processing results
            results_file = output_dir / "ontology_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(ontology_results, f, indent=2)
            
            # Generate summary report
            summary_file = output_dir / "ontology_summary.md"
            with open(summary_file, 'w') as f:
                f.write(f"# Ontology Analysis Summary\n\n")
                f.write(f"- Files processed: {len(gnn_files)}\n")
                f.write(f"- Files with ontology mappings: {files_with_mappings}\n")
                f.write(f"- Total mappings found: {total_mappings}\n")
                f.write(f"- Ontology terms analyzed: {len(ontology_terms)}\n\n")
                
                if ontology_results["mappings"]:
                    f.write("## Mappings Found\n\n")
                    for mapping in ontology_results["mappings"][:10]:  # Show first 10
                        f.write(f"- **{mapping['term']}** in {mapping['file']}\n")
            
            logger.info(f"Ontology analysis completed: {results_file}")
            logger.info(f"Summary report: {summary_file}")
            
            log_step_success(logger, f"Processed {len(gnn_files)} files with {total_mappings} ontology mappings")
            return True
            
    except Exception as e:
        log_step_error(logger, f"Ontology failed: {e}")
        return False

run_script = create_standardized_pipeline_script(
    "8_ontology.py",
    process_ontology_standardized,
    "Ontology processing and validation",
    additional_arguments={
        "ontology_terms_file": {
            "type": Path, 
            "help": "Path to ontology terms JSON file",
            "flag": "--ontology-terms-file",
            "dest": "ontology_terms_file"
        }
    }
)

if __name__ == '__main__':
    sys.exit(run_script()) 