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

import os
import json
import datetime 
import logging
from pathlib import Path
import sys
import argparse

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

def process_ontology_operations(target_dir: Path, output_dir: Path, ontology_terms_file: Path = None, recursive: bool = False):
    """Process ontology operations for GNN files."""
    log_step_start(logger, f"Processing ontology operations for: {target_dir}")
    
    # Use centralized output directory configuration
    ontology_output_dir = get_output_dir_for_script("8_ontology.py", output_dir)
    ontology_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files to process for ontology operations")
        
        # Load ontology terms if file provided
        ontology_terms = {}
        if ontology_terms_file and ontology_terms_file.exists():
            try:
                with open(ontology_terms_file, 'r') as f:
                    ontology_terms = json.load(f)
                logger.info(f"Loaded {len(ontology_terms)} ontology terms from {ontology_terms_file}")
            except Exception as e:
                log_step_warning(logger, f"Failed to load ontology terms file: {e}")
        
        # Process each GNN file
        processed_files = []
        for gnn_file in gnn_files:
            try:
                logger.debug(f"Processing ontology for file: {gnn_file}")
                
                # Read file content
                with open(gnn_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract ontology annotations if MCP module available
                annotations = {}
                if ontology_mcp:
                    try:
                        annotations = ontology_mcp.parse_gnn_ontology_section(content, verbose=True)
                    except Exception as e:
                        log_step_warning(logger, f"Failed to parse ontology annotations in {gnn_file}: {e}")
                
                # Create file-specific output
                file_output = {
                    "file": str(gnn_file.relative_to(target_dir)),
                    "annotations": annotations,
                    "terms_matched": [],
                    "terms_missing": []
                }
                
                # Match annotations with ontology terms
                if annotations and ontology_terms:
                    for annotation in annotations.get('terms', []):
                        if annotation in ontology_terms:
                            file_output["terms_matched"].append(annotation)
                        else:
                            file_output["terms_missing"].append(annotation)
                
                processed_files.append(file_output)
                
            except Exception as e:
                log_step_error(logger, f"Failed to process ontology for {gnn_file}: {e}")
        
        # Generate summary report
        report_content = {
            "timestamp": datetime.datetime.now().isoformat(),
            "target_directory": str(target_dir),
            "files_processed": len(processed_files),
            "ontology_terms_loaded": len(ontology_terms),
            "files": processed_files
        }
        
        # Save JSON report
        json_report_file = ontology_output_dir / "ontology_processing_report.json"
        with open(json_report_file, 'w') as f:
            json.dump(report_content, f, indent=2)
        
        # Save Markdown report
        md_report_file = ontology_output_dir / "ontology_processing_report.md"
        with open(md_report_file, 'w') as f:
            f.write("# Ontology Processing Report\n\n")
            f.write(f"**Generated:** {report_content['timestamp']}\n")
            f.write(f"**Target Directory:** {report_content['target_directory']}\n")
            f.write(f"**Files Processed:** {report_content['files_processed']}\n")
            f.write(f"**Ontology Terms Loaded:** {report_content['ontology_terms_loaded']}\n\n")
            
            f.write("## File Details\n\n")
            for file_info in processed_files:
                f.write(f"### {file_info['file']}\n")
                f.write(f"- **Annotations Found:** {len(file_info['annotations'])}\n")
                f.write(f"- **Terms Matched:** {len(file_info['terms_matched'])}\n")
                f.write(f"- **Terms Missing:** {len(file_info['terms_missing'])}\n\n")
        
        log_step_success(logger, f"Ontology processing completed. Reports saved to {ontology_output_dir}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False

def main(parsed_args: argparse.Namespace):
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
    success = process_ontology_operations(
        target_dir=Path(parsed_args.target_dir),
        output_dir=Path(parsed_args.output_dir),
        ontology_terms_file=ontology_terms_file,
        recursive=getattr(parsed_args, 'recursive', False)
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