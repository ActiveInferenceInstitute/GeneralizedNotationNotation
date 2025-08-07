#!/usr/bin/env python3
"""
Step 7: Multi-format Export Generation (Thin Orchestrator)

This step generates exports in multiple formats (JSON, XML, GraphML, GEXF, Pickle).

How to run:
  python src/7_export.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Multi-format exports in the specified output directory
  - JSON, XML, GraphML, GEXF, and Pickle format files
  - Comprehensive export reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that export dependencies are installed
  - Check that src/export/ contains export modules
  - Check that the output directory is writable
  - Verify export configuration and format requirements
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

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

# Import core export functions from export module
try:
    from export import (
        export_to_json,
        export_to_xml,
        export_to_graphml,
        export_to_gexf,
        export_to_pickle,
        export_gnn_model
    )
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    # Fallback function definitions if export module is not available
    def export_to_json(model_data: Dict, output_path: Path) -> bool:
        print("Export module not available")
        return False
    
    def export_to_xml(model_data: Dict, output_path: Path) -> bool:
        print("Export module not available")
        return False
    
    def export_to_graphml(model_data: Dict, output_path: Path) -> bool:
        print("Export module not available")
        return False
    
    def export_to_gexf(model_data: Dict, output_path: Path) -> bool:
        print("Export module not available")
        return False
    
    def export_to_pickle(model_data: Dict, output_path: Path) -> bool:
        print("Export module not available")
        return False
    
    def export_gnn_model(model_data: Dict, output_dir: Path, formats: List[str] = None) -> Dict[str, Any]:
        return {"error": "Export module not available"}

def process_export_standardized(
    target_dir: Path,
    output_dir: Path,
    logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized export processing function.
    
    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for export results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Check if export module is available
        if not EXPORT_AVAILABLE:
            log_step_warning(logger, "Export module not available, using fallback functions")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("7_export.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing export")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return False
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Export results
        export_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(step_output_dir),
            "files_exported": [],
            "summary": {
                "total_files": 0,
                "successful_exports": 0,
                "failed_exports": 0,
                "formats_generated": {
                    "json": 0,
                    "xml": 0,
                    "graphml": 0,
                    "gexf": 0,
                    "pickle": 0
                }
            }
        }
        
        # Export formats
        export_formats = [
            ("json", export_to_json, ".json"),
            ("xml", export_to_xml, ".xml"),
            ("graphml", export_to_graphml, ".graphml"),
            ("gexf", export_to_gexf, ".gexf"),
            ("pickle", export_to_pickle, ".pkl")
        ]
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Exporting: {file_name}")
            
            # Load the actual parsed GNN specification
            parsed_model_file = file_result.get("parsed_model_file")
            if parsed_model_file and Path(parsed_model_file).exists():
                try:
                    with open(parsed_model_file, 'r') as f:
                        actual_gnn_spec = json.load(f)
                    logger.info(f"Loaded parsed GNN specification from {parsed_model_file}")
                    model_data = actual_gnn_spec
                except Exception as e:
                    logger.error(f"Failed to load parsed GNN spec from {parsed_model_file}: {e}")
                    model_data = file_result
            else:
                logger.warning(f"Parsed model file not found for {file_name}, using summary data")
                model_data = file_result
            
            # Create file-specific output directory
            file_output_dir = step_output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_export_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "exports": {},
                "success": True
            }
            
            # Generate exports for each format
            for format_name, export_func, extension in export_formats:
                try:
                    # Generate export file
                    export_file = file_output_dir / f"{file_name.replace('.md', '')}_{format_name}{extension}"
                    success = export_func(model_data, export_file)
                    
                    if success:
                        file_export_result["exports"][format_name] = {
                            "success": True,
                            "export_file": str(export_file),
                            "file_size": export_file.stat().st_size if export_file.exists() else 0
                        }
                        export_results["summary"]["formats_generated"][format_name] += 1
                        logger.info(f"Generated {format_name} export for {file_name}")
                    else:
                        file_export_result["exports"][format_name] = {
                            "success": False,
                            "error": f"Export function returned False"
                        }
                        file_export_result["success"] = False
                        
                except Exception as e:
                    logger.error(f"Failed to generate {format_name} export for {file_name}: {e}")
                    file_export_result["exports"][format_name] = {
                        "success": False,
                        "error": str(e)
                    }
                    file_export_result["success"] = False
            
            export_results["files_exported"].append(file_export_result)
            export_results["summary"]["total_files"] += 1
            
            if file_export_result["success"]:
                export_results["summary"]["successful_exports"] += 1
            else:
                export_results["summary"]["failed_exports"] += 1
        
        # Save export results
        export_results_file = step_output_dir / "export_results.json"
        with open(export_results_file, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        # Save export summary
        export_summary_file = step_output_dir / "export_summary.json"
        with open(export_summary_file, 'w') as f:
            json.dump(export_results["summary"], f, indent=2)
        
        logger.info(f"Export processing completed:")
        logger.info(f"  Total files: {export_results['summary']['total_files']}")
        logger.info(f"  Successful exports: {export_results['summary']['successful_exports']}")
        logger.info(f"  Failed exports: {export_results['summary']['failed_exports']}")
        logger.info(f"  Formats generated: {export_results['summary']['formats_generated']}")
        
        log_step_success(logger, "Export processing completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Export processing failed: {e}")
        return False

def main():
    """Main export processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("7_export.py")
    
    # Setup logging
    logger = setup_step_logging("export", args)
    
    # Check if export module is available
    if not EXPORT_AVAILABLE:
        log_step_warning(logger, "Export module not available, using fallback functions")
    
    # Process export
    success = process_export_standardized(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
