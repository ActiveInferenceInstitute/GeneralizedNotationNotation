from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json

# Import ontology functionality from mcp.py
try:
    from .mcp import (
        parse_gnn_ontology_section,
        load_defined_ontology_terms,
        validate_annotations,
        generate_ontology_report_for_file
    )
    ONTOLOGY_AVAILABLE = True
except ImportError as e:
    # Fallback implementation
    ONTOLOGY_AVAILABLE = False
    
    def parse_gnn_ontology_section(content):
        return {}
    
    def load_defined_ontology_terms(path):
        return {}
    
    def validate_annotations(annotations, terms):
        return {"valid_mappings": {}, "invalid_terms": {}, "unmapped_model_vars": []}
    
    def generate_ontology_report_for_file(file_path, annotations, validation_results=None):
        return f"# Ontology Report for {file_path}\n\nOntology functionality not available."

def process_ontology_operations(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False, ontology_terms_file: Path = None):
    """Process Active Inference ontology operations."""
    log_step_start(logger, "Processing Active Inference ontology operations")
    
    # Use centralized output directory configuration
    ontology_output_dir = get_output_dir_for_script("8_ontology.py", output_dir)
    ontology_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ONTOLOGY_AVAILABLE:
        log_step_error(logger, "Ontology functionality not available")
        return False
    
    try:
        # Load ontology terms if provided
        defined_terms = {}
        if ontology_terms_file and ontology_terms_file.exists():
            logger.info(f"Loading ontology terms from: {ontology_terms_file}")
            defined_terms = load_defined_ontology_terms(str(ontology_terms_file))
            logger.info(f"Loaded {len(defined_terms)} ontology terms")
        else:
            logger.warning("No ontology terms file provided or file not found")
        
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
        all_results = []
        
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
                    
                    # Parse ontology annotations
                    annotations = parse_gnn_ontology_section(gnn_content, verbose=logger.isEnabledFor(logging.DEBUG))
                    
                    # Validate annotations if we have defined terms
                    validation_results = None
                    if defined_terms:
                        validation_results = validate_annotations(annotations, defined_terms)
                    
                    # Generate ontology report
                    report_content = generate_ontology_report_for_file(
                        str(gnn_file), 
                        annotations, 
                        validation_results
                    )
                    
                    # Save report
                    report_file = file_output_dir / "ontology_report.md"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    # Prepare results
                    results = {
                        "file": str(gnn_file),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "annotations_found": len(annotations),
                        "annotations": annotations,
                        "validation_results": validation_results,
                        "success": True
                    }
                    
                    # Add validation summary if available
                    if validation_results:
                        results["valid_mappings"] = len(validation_results.get("valid_mappings", {}))
                        results["invalid_terms"] = len(validation_results.get("invalid_terms", {}))
                        results["success"] = len(validation_results.get("invalid_terms", {})) == 0
                    
                    # Save individual file results
                    results_file = file_output_dir / "ontology_results.json"
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    
                    all_results.append(results)
                    
                    if results["success"]:
                        successful_operations += 1
                        logger.debug(f"Ontology processing completed for {gnn_file.name}")
                    else:
                        failed_operations += 1
                        log_step_warning(logger, f"Ontology processing failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_operations += 1
                    log_step_error(logger, f"Failed to process {gnn_file.name} with ontology: {e}")
        
        # Generate comprehensive ontology summary report
        summary_file = ontology_output_dir / "ontology_processing_summary.json"
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(gnn_files),
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": successful_operations / len(gnn_files) * 100 if gnn_files else 0,
            "ontology_terms_loaded": len(defined_terms),
            "files_processed": all_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate markdown summary report
        markdown_summary = generate_markdown_summary(summary, ontology_output_dir)
        summary_md_file = ontology_output_dir / "ontology_summary_report.md"
        with open(summary_md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        
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

def generate_markdown_summary(summary: dict, output_dir: Path) -> str:
    """Generate a comprehensive markdown summary report."""
    lines = [
        "# Active Inference Ontology Processing Summary\n",
        f"**Generated:** {summary['timestamp']}\n",
        f"**Total Files Processed:** {summary['total_files']}\n",
        f"**Successful Operations:** {summary['successful_operations']}\n",
        f"**Failed Operations:** {summary['failed_operations']}\n",
        f"**Success Rate:** {summary['success_rate']:.1f}%\n",
        f"**Ontology Terms Loaded:** {summary['ontology_terms_loaded']}\n\n",
        "## File Processing Results\n"
    ]
    
    for file_result in summary.get('files_processed', []):
        lines.append(f"### {Path(file_result['file']).name}\n")
        lines.append(f"- **File:** {file_result['file']}\n")
        lines.append(f"- **Annotations Found:** {file_result['annotations_found']}\n")
        
        if file_result.get('validation_results'):
            lines.append(f"- **Valid Mappings:** {file_result.get('valid_mappings', 0)}\n")
            lines.append(f"- **Invalid Terms:** {file_result.get('invalid_terms', 0)}\n")
        
        lines.append(f"- **Status:** {'✅ Success' if file_result['success'] else '❌ Failed'}\n\n")
        
        # Add annotation details
        if file_result.get('annotations'):
            lines.append("#### Annotations:\n")
            for var, term in file_result['annotations'].items():
                lines.append(f"- `{var}` → `{term}`\n")
            lines.append("\n")
    
    lines.append("---\n")
    lines.append("*Report generated by GNN Ontology Processing Module*\n")
    
    return "".join(lines) 