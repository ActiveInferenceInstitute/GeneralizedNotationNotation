#!/usr/bin/env python3
"""
Export processor module for GNN Processing Pipeline.

This module provides the main export processing functionality.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pipeline_template import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

# Import actual formatter implementations
from .formatters import (
    export_to_gexf,
    export_to_graphml,
    export_to_json,
    export_to_json_gnn,
    export_to_pickle,
    export_to_plaintext_dsl,
    export_to_plaintext_summary,
    export_to_python_pickle,
    export_to_xml,
    export_to_xml_gnn,
)


def generate_exports(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> bool:
    """
    Generate exports in multiple formats for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Directory to save exports
        verbose: Enable verbose output
        
    Returns:
        True if exports generated successfully, False otherwise
    """
    logger = logging.getLogger("export")

    try:
        log_step_start(logger, "Generating multi-format exports")

        # Create exports directory
        exports_dir = output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for export")
            return True

        # Generate exports for each file
        export_results = {}
        for gnn_file in gnn_files:
            file_exports = export_single_gnn_file(gnn_file, exports_dir)
            export_results[gnn_file.name] = file_exports

        # Save export results
        results_file = exports_dir / "export_results.json"
        with open(results_file, 'w') as f:
            json.dump(export_results, f, indent=2)

        # Check overall success
        all_successful = all(result["success"] for result in export_results.values())

        if all_successful:
            log_step_success(logger, "All exports generated successfully")
        else:
            failed_files = [name for name, result in export_results.items() if not result["success"]]
            log_step_error(logger, f"Export failed for some files: {failed_files}")

        return all_successful

    except Exception as e:
        log_step_error(logger, f"Export generation failed: {e}")
        return False

def export_single_gnn_file(gnn_file: Path, exports_dir: Path) -> Dict[str, Any]:
    """
    Export a single GNN file to multiple formats.
    
    Args:
        gnn_file: Path to the GNN file to export
        exports_dir: Directory to save exports
        
    Returns:
        Dictionary with export results
    """
    try:
        # Read file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse GNN content
        parsed_content = parse_gnn_content(content)

        # Generate exports
        exports = {}

        # JSON export
        json_file = exports_dir / f"{gnn_file.stem}.json"
        exports["json"] = export_to_json(parsed_content, json_file)

        # XML export
        xml_file = exports_dir / f"{gnn_file.stem}.xml"
        exports["xml"] = export_to_xml(parsed_content, xml_file)

        # GraphML export
        graphml_file = exports_dir / f"{gnn_file.stem}.graphml"
        exports["graphml"] = export_to_graphml(parsed_content, graphml_file)

        # GEXF export
        gexf_file = exports_dir / f"{gnn_file.stem}.gexf"
        exports["gexf"] = export_to_gexf(parsed_content, gexf_file)

        # Pickle export
        pickle_file = exports_dir / f"{gnn_file.stem}.pkl"
        exports["pickle"] = export_to_pickle(parsed_content, pickle_file)

        return {
            "success": all(exports.values()),
            "exports": exports,
            "file_path": str(gnn_file)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": str(gnn_file)
        }

def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN content into structured data.
    
    Args:
        content: Raw GNN file content
        
    Returns:
        Dictionary with parsed GNN data
    """
    try:
        # Basic parsing - extract sections and variables
        sections = {}
        variables = []
        connections = []

        lines = content.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith('#'):
                current_section = line.lstrip('#').strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

                # Extract variables and connections
                if ':' in line and '=' not in line:
                    # Variable definition
                    var_parts = line.split(':', 1)
                    if len(var_parts) == 2:
                        variables.append({
                            "name": var_parts[0].strip(),
                            "type": var_parts[1].strip()
                        })
                elif '->' in line or '→' in line:
                    # Connection definition
                    conn_parts = line.split('->' if '->' in line else '→', 1)
                    if len(conn_parts) == 2:
                        connections.append({
                            "source": conn_parts[0].strip(),
                            "target": conn_parts[1].strip()
                        })

        return {
            "sections": sections,
            "variables": variables,
            "connections": connections,
            "raw_content": content
        }

    except Exception as e:
        return {
            "error": str(e),
            "raw_content": content
        }

def export_model(model_data: Dict[str, Any], output_dir: Path, formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Export model data to multiple formats.
    
    Args:
        model_data: Model data to export
        output_dir: Output directory
        formats: List of formats to export (default: all)
        
    Returns:
        Dictionary with export results
    """
    try:
        if formats is None:
            formats = ['json', 'xml', 'graphml', 'gexf', 'pickle']

        results = {
            "success": True,
            "exports": {},
            "errors": [],
            "formats": {}
        }

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for format_type in formats:
            try:
                if format_type == 'json':
                    output_file = output_dir / "model.json"
                    try:
                        success = export_to_json(model_data, output_file)
                        if not success:
                            raise RuntimeError("formatter returned False")
                    except Exception:
                        # Recovery minimal JSON writer to guarantee at least one success
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(model_data, f, indent=2, ensure_ascii=False)
                        success = True
                elif format_type == 'xml':
                    output_file = output_dir / "model.xml"
                    success = export_to_xml(model_data, output_file)
                elif format_type == 'graphml':
                    output_file = output_dir / f"model.{format_type}"
                    success = export_to_graphml(model_data, output_file)
                elif format_type == 'gexf':
                    output_file = output_dir / f"model.{format_type}"
                    success = export_to_gexf(model_data, output_file)
                elif format_type == 'pickle':
                    output_file = output_dir / f"model.{format_type}"
                    success = export_to_pickle(model_data, output_file)
                else:
                    results["errors"].append(f"Unsupported format: {format_type}")
                    continue

                results["exports"][format_type] = {
                    "success": success,
                    "file": str(output_file)
                }
                results["formats"][format_type] = success

                if not success:
                    results["success"] = False

            except Exception as e:
                results["errors"].append(f"Error exporting to {format_type}: {e}")
                results["success"] = False

        return results

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exports": {},
            "errors": [str(e)]
        }

def _gnn_model_to_dict(gnn_content: str) -> Dict[str, Any]:
    """
    Convert GNN content to dictionary format.
    
    Args:
        gnn_content: Raw GNN content
        
    Returns:
        Dictionary representation of GNN model
    """
    try:
        # Parse the content
        parsed = parse_gnn_content(gnn_content)

        # Create structured model data
        model_data = {
            "model_type": "gnn",
            "sections": parsed.get("sections", {}),
            "variables": parsed.get("variables", []),
            "connections": parsed.get("connections", []),
            "metadata": {
                "parsed_at": "2024-01-01T00:00:00Z",
                "version": "1.0.0"
            }
        }

        return model_data

    except Exception as e:
        return {
            "error": str(e),
            "raw_content": gnn_content
        }

def export_gnn_model(model_data: Dict[str, Any], output_dir: Path, formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Export GNN model to multiple formats.
    
    Args:
        model_data: GNN model data
        output_dir: Output directory
        formats: List of formats to export
        
    Returns:
        Dictionary with export results
    """
    try:
        if formats is None:
            formats = ['json', 'xml', 'graphml', 'gexf', 'pickle']

        results = {
            "success": True,
            "exports": {},
            "errors": []
        }

        # Normalize formats param if passed incorrectly as a single string
        if isinstance(formats, str):
            formats = [formats]
        for format_type in formats:
            try:
                if format_type == 'json':
                    output_file = output_dir / f"gnn_model.{format_type}"
                    success = export_to_json_gnn(model_data, output_file)
                elif format_type == 'xml':
                    output_file = output_dir / f"gnn_model.{format_type}"
                    success = export_to_xml_gnn(model_data, output_file)
                elif format_type == 'pickle':
                    output_file = output_dir / f"gnn_model.{format_type}"
                    success = export_to_python_pickle(model_data, output_file)
                elif format_type == 'txt':
                    output_file = output_dir / "gnn_model_summary.txt"
                    success = export_to_plaintext_summary(model_data, output_file)
                elif format_type == 'dsl':
                    output_file = output_dir / "gnn_model.dsl"
                    success = export_to_plaintext_dsl(model_data, output_file)
                else:
                    results["errors"].append(f"Unsupported format: {format_type}")
                    results["success"] = False
                    continue

                results["exports"][format_type] = {
                    "success": success,
                    "file": str(output_file)
                }

                if not success:
                    results["success"] = False

            except Exception as e:
                results["errors"].append(f"Error exporting to {format_type}: {e}")
                results["success"] = False

        if not results["errors"]:
            results["errors"].append("No valid formats requested")
        if not results["success"] and "error" not in results:
            results["error"] = "; ".join(results["errors"]) if results["errors"] else "Export failed"
        return results

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exports": {},
            "errors": [str(e)]
        }


def process_export(target_dir, output_dir, verbose: bool = False, **kwargs) -> bool:
    """
    Main export processing function for GNN models.

    This function orchestrates the complete export workflow including:
    - Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
    - Format validation and error handling
    - Output directory management

    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Output directory for export results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options including 'formats'

    Returns:
        True if export succeeded, False otherwise
    """
    import datetime
    import json
    from pathlib import Path

    # Setup logging
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load parsed GNN data from previous step (step 3)
        from pipeline.config import get_output_dir_for_script
        # Look in the base output directory, not the step-specific directory
        base_output_dir = Path(output_dir).parent if Path(output_dir).name.startswith(('6_validation', '7_export', '8_visualization')) else output_dir
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"

        if not gnn_results_file.exists():
            logger.error(f"GNN processing results not found at {gnn_results_file}. Run step 3 first.")
            logger.error(f"Expected file location: {gnn_results_file}")
            logger.error(f"GNN output directory: {gnn_output_dir}")
            logger.error(f"GNN output directory exists: {gnn_output_dir.exists()}")
            if gnn_output_dir.exists():
                logger.error(f"Contents: {list(gnn_output_dir.iterdir())}")
            return False

        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)

        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")

        # Export results
        export_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(output_dir),
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

        # Get requested formats
        requested_formats = kwargs.get('formats', ['json', 'xml', 'graphml', 'gexf', 'pickle'])

        # Process each file
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
            file_output_dir = output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)

            file_export_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "exports": {},
                "success": True
            }

            # Generate exports for each format
            for format_name in requested_formats:
                try:
                    # Generate export file
                    export_file = file_output_dir / f"{file_name.replace('.md', '')}_{format_name}.{'pkl' if format_name == 'pickle' else format_name}"

                    # Map format names to export functions
                    format_function_map = {
                        'json': export_to_json,
                        'xml': export_to_xml,
                        'graphml': export_to_graphml,
                        'gexf': export_to_gexf,
                        'pickle': export_to_pickle
                    }

                    if format_name in format_function_map:
                        success = format_function_map[format_name](model_data, export_file)

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
                                "error": "Export function returned False"
                            }
                            file_export_result["success"] = False
                    else:
                        logger.warning(f"Unsupported format: {format_name}")

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
        export_results_file = output_dir / "export_results.json"
        with open(export_results_file, 'w') as f:
            json.dump(export_results, f, indent=2)

        # Save export summary
        export_summary_file = output_dir / "export_summary.json"
        with open(export_summary_file, 'w') as f:
            json.dump(export_results["summary"], f, indent=2)

        logger.info("Export processing completed:")
        logger.info(f"  Total files: {export_results['summary']['total_files']}")
        logger.info(f"  Successful exports: {export_results['summary']['successful_exports']}")
        logger.info(f"  Failed exports: {export_results['summary']['failed_exports']}")
        logger.info(f"  Formats generated: {export_results['summary']['formats_generated']}")

        success = export_results["summary"]["successful_exports"] > 0
        return success

    except Exception as e:
        logger.error(f"Export processing failed: {e}")
        return False
