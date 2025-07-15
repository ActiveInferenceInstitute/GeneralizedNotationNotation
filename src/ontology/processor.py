from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json

# Import ontology functionality
try:
    from .ontology_validator import OntologyValidator
    from .ontology_mapper import OntologyMapper
    ONTOLOGY_AVAILABLE = True
except ImportError as e:
    ONTOLOGY_AVAILABLE = False

def process_ontology_operations(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False):
    """Process Active Inference ontology operations."""
    log_step_start(logger, "Processing Active Inference ontology operations")
    
    # Use centralized output directory configuration
    ontology_output_dir = get_output_dir_for_script("8_ontology.py", output_dir)
    ontology_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ONTOLOGY_AVAILABLE:
        log_step_error(logger, "Ontology functionality not available")
        return False
    
    try:
        # Initialize ontology components
        validator = OntologyValidator()
        mapper = OntologyMapper()
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files for ontology processing")
        
        # Process files with ontology tools
        successful_operations = 0
        failed_operations = 0
        
        with performance_tracker.track_operation("process_ontology_operations"):
            for gnn_file in gnn_files:
                try:
                    logger.debug(f"Processing {gnn_file.name} with ontology tools")
                    
                    # Create file-specific output directory
                    file_output_dir = ontology_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Read GNN file content
                    with open(gnn_file, 'r', encoding='utf-8') as f:
                        gnn_content = f.read()
                    
                    # Validate ontology mappings
                    validation_results = validator.validate_gnn_ontology(gnn_content)
                    
                    # Map to ontology terms
                    mapping_results = mapper.map_gnn_to_ontology(gnn_content)
                    
                    # Save results
                    results = {
                        "file": str(gnn_file),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "validation": validation_results,
                        "mapping": mapping_results,
                        "success": validation_results.get("valid", False) and mapping_results.get("success", False)
                    }
                    
                    # Save individual file results
                    results_file = file_output_dir / "ontology_results.json"
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    if results["success"]:
                        successful_operations += 1
                        logger.debug(f"Ontology processing completed for {gnn_file.name}")
                    else:
                        failed_operations += 1
                        log_step_warning(logger, f"Ontology processing failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_operations += 1
                    log_step_error(logger, f"Failed to process {gnn_file.name} with ontology: {e}")
        
        # Generate ontology summary report
        summary_file = ontology_output_dir / "ontology_processing_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(gnn_files),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / len(gnn_files) * 100 if gnn_files else 0
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log results summary
        if successful_operations == len(gnn_files):
            log_step_success(logger, f"All {len(gnn_files)} files processed successfully with ontology")
            return True
        elif successful_operations > 0:
            log_step_warning(logger, f"Partial success: {successful_operations}/{len(gnn_files)} files processed successfully")
            return True
        else:
            log_step_error(logger, "No files were processed successfully with ontology")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Ontology processing failed: {e}")
        return False 