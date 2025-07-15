from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error

# Import format exporters
try:
    from .format_exporters import (
        _gnn_model_to_dict,
        export_to_json_gnn,
        export_to_xml_gnn,
        export_to_plaintext_summary,
        export_to_plaintext_dsl,
        export_to_gexf,
        export_to_graphml,
        export_to_json_adjacency_list,
        export_to_python_pickle,
        HAS_NETWORKX
    )
    FORMAT_EXPORTERS_LOADED = True
except ImportError as e:
    FORMAT_EXPORTERS_LOADED = False
    HAS_NETWORKX = False

def export_gnn_files(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Export GNN files to multiple formats.
    
    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional export options
        
    Returns:
        True if export succeeded, False otherwise
    """
    log_step_start(logger, f"Exporting GNN files from: {target_dir}")
    
    if not FORMAT_EXPORTERS_LOADED:
        log_step_error(logger, "Format exporters not available")
        return False
    
    # Use centralized output directory configuration
    export_output_dir = get_output_dir_for_script("5_export.py", output_dir)
    export_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir}")
        return True
    
    logger.info(f"Found {len(gnn_files)} GNN files to export")
    
    success_count = 0
    total_files = len(gnn_files)
    
    for gnn_file in gnn_files:
        try:
            logger.debug(f"Processing file: {gnn_file}")
            
            # Parse GNN file to dictionary
            gnn_dict = _gnn_model_to_dict(gnn_file)
            
            # Create file-specific output directory
            file_output_dir = export_output_dir / gnn_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export to various formats (functions raise exceptions on failure)
            export_success = True
            export_errors = []
            
            # JSON export
            try:
                export_to_json_gnn(gnn_dict, file_output_dir / f"{gnn_file.stem}.json")
            except Exception as e:
                export_success = False
                export_errors.append(f"JSON: {e}")
            
            # XML export
            try:
                export_to_xml_gnn(gnn_dict, file_output_dir / f"{gnn_file.stem}.xml")
            except Exception as e:
                export_success = False
                export_errors.append(f"XML: {e}")
            
            # Plaintext exports
            try:
                export_to_plaintext_summary(gnn_dict, file_output_dir / f"{gnn_file.stem}_summary.txt")
            except Exception as e:
                export_success = False
                export_errors.append(f"Summary: {e}")
            
            try:
                export_to_plaintext_dsl(gnn_dict, file_output_dir / f"{gnn_file.stem}_dsl.txt")
            except Exception as e:
                export_success = False
                export_errors.append(f"DSL: {e}")
            
            # Graph exports (if NetworkX available)
            if HAS_NETWORKX:
                try:
                    export_to_gexf(gnn_dict, file_output_dir / f"{gnn_file.stem}.gexf")
                except Exception as e:
                    export_success = False
                    export_errors.append(f"GEXF: {e}")
                
                try:
                    export_to_graphml(gnn_dict, file_output_dir / f"{gnn_file.stem}.graphml")
                except Exception as e:
                    export_success = False
                    export_errors.append(f"GraphML: {e}")
                
                try:
                    export_to_json_adjacency_list(gnn_dict, file_output_dir / f"{gnn_file.stem}_adjacency.json")
                except Exception as e:
                    export_success = False
                    export_errors.append(f"Adjacency: {e}")
            
            # Pickle export
            try:
                export_to_python_pickle(gnn_dict, file_output_dir / f"{gnn_file.stem}.pkl")
            except Exception as e:
                export_success = False
                export_errors.append(f"Pickle: {e}")
            
            if export_success:
                success_count += 1
                logger.debug(f"Successfully exported {gnn_file.name}")
            else:
                log_step_warning(logger, f"Some exports failed for {gnn_file.name}: {'; '.join(export_errors)}")
                
        except Exception as e:
            log_step_error(logger, f"Failed to export {gnn_file}: {e}")
    
    # Log summary
    logger.info(f"Export completed: {success_count}/{total_files} files exported successfully")
    
    if success_count == total_files:
        log_step_success(logger, "All GNN files exported successfully")
        return True
    elif success_count > 0:
        log_step_warning(logger, f"Partial success: {success_count}/{total_files} files exported")
        return True
    else:
        log_step_error(logger, "No files were exported successfully")
        return False 