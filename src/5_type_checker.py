#!/usr/bin/env python3
"""
Step 5: Type Checking and Validation (Thin Orchestrator)

This step performs type checking and validation on GNN files.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/type_checker/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the type_checker module.

Pipeline Flow:
    main.py → 5_type_checker.py (this script) → type_checker/ (modular implementation)
"""

import sys
import json
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
    log_step_warning,
    create_standardized_pipeline_script,
)
from utils.argument_utils import ArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

from type_checker.analysis_utils import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity,
)

run_script = create_standardized_pipeline_script(
    "5_type_checker.py",
    lambda target_dir, output_dir, logger, **kwargs: _run_type_check(
        target_dir, output_dir, logger, **kwargs
    ),
    "Type checking and validation of GNN files",
)


def _run_type_check(target_dir: Path, output_dir: Path, logger, **kwargs) -> bool:
    args = ArgumentParser.parse_step_arguments("5_type_checker.py")
    
    # Load configuration for POMDP settings
    config_file = Path("input/config.yaml")
    pomdp_config = {}
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            pomdp_config = config.get("validation", {}).get("pomdp", {})
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
    
    # Check for POMDP mode configuration
    pomdp_mode = getattr(args, "pomdp_mode", pomdp_config.get("enabled", False))
    ontology_file = None
    if pomdp_mode:
        ontology_file_path = pomdp_config.get("ontology_file", "input/ontology_terms.json")
        ontology_file = Path(ontology_file_path)
        if not ontology_file.exists():
            logger.warning(f"POMDP mode enabled but ontology file not found at {ontology_file}, using default terms")
            ontology_file = None
    
    # Load parsed GNN data from previous step
    gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
    # Step 3 uses double-nested output directory structure
    gnn_nested_dir = gnn_output_dir / "3_gnn_output"
    gnn_results_file = gnn_nested_dir / "gnn_processing_results.json"

    if not gnn_results_file.exists():
        log_step_error(logger, "GNN processing results not found. Run step 3 first.")
        return False

    with open(gnn_results_file, "r") as f:
        gnn_results = json.load(f)

    logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")

    # Ensure output directory exists with expected structure
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create the expected prerequisite directory structure
    prereq_dir = Path(args.output_dir) / "5_type_output"
    prereq_dir.mkdir(parents=True, exist_ok=True)

    type_check_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source_directory": str(args.target_dir),
        "output_directory": str(output_dir),
        "strict_mode": getattr(args, "strict", False),
        "resource_estimation": getattr(args, "estimate_resources", False),
        "pomdp_mode": pomdp_mode,
        "ontology_file": str(ontology_file) if ontology_file else None,
        "files_analyzed": [],
        "summary": {
            "total_files": 0,
            "valid_files": 0,
            "type_errors": 0,
            "warnings": 0,
        },
        "global_analysis": {
            "type_analysis": {},
            "connection_analysis": {},
            "complexity_analysis": {},
        },
    }
    
    # Add POMDP-specific summary if in POMDP mode
    if pomdp_mode:
        type_check_results["summary"]["pomdp_metrics"] = {
            "total_state_space_size": 0,
            "total_observation_space_size": 0,
            "total_action_space_size": 0,
            "pomdp_models_valid": 0
        }

    all_variables: List[Dict[str, Any]] = []
    all_connections: List[Dict[str, Any]] = []

    for file_result in gnn_results.get("processed_files", []):
        if not file_result.get("parse_success"):
            continue

        file_name = file_result["file_name"]
        logger.info(f"Analyzing types for: {file_name}")

        parsed_model_file = Path(file_result.get("parsed_model_file", ""))
        if not parsed_model_file.exists():
            logger.info(f"Parsed model file not found: {parsed_model_file}")
            continue

        try:
            with open(parsed_model_file, "r") as f:
                parsed_model = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load parsed model from {parsed_model_file}: {e}")
            continue

        variables = parsed_model.get("variables", [])
        connections = parsed_model.get("connections", [])

        type_analysis = analyze_variable_types(variables)
        connection_analysis = analyze_connections(connections)
        complexity_analysis = estimate_computational_complexity(type_analysis, connection_analysis)
        
        # POMDP-specific analysis if enabled
        pomdp_analysis = None
        if pomdp_mode:
            try:
                from type_checker.processor import GNNTypeChecker
                pomdp_checker = GNNTypeChecker(pomdp_mode=True, ontology_file=ontology_file)
                # Try different possible file names for POMDP analysis
                pomdp_file_candidates = [
                    parsed_model_file.parent / file_name,
                    parsed_model_file.parent / f"{file_name.replace('.md', '_markdown.md')}",
                    parsed_model_file.parent / f"{file_name.replace('.md', '_processed.md')}",
                    parsed_model_file.parent / file_name.replace('.md', '_markdown.md'),
                    parsed_model_file.parent / file_name.replace('.md', '_processed.md')
                ]
                
                pomdp_analysis = None
                for candidate in pomdp_file_candidates:
                    if candidate.exists():
                        pomdp_analysis = pomdp_checker.validate_pomdp_file(candidate)
                        break
                
                if pomdp_analysis is None:
                    # Fallback to using the original file if no processed version found
                    pomdp_analysis = pomdp_checker.validate_pomdp_file(Path(file_result.get("file_path", "")))
                
                # Add POMDP metrics to summary
                if pomdp_analysis and "pomdp_analysis" in pomdp_analysis:
                    inner_pomdp_analysis = pomdp_analysis["pomdp_analysis"]
                    if "pomdp_specific" in inner_pomdp_analysis:
                        pomdp_metrics = inner_pomdp_analysis["pomdp_specific"]
                        type_check_results["summary"]["pomdp_metrics"]["total_state_space_size"] += pomdp_metrics.get("state_space_size", 0)
                        type_check_results["summary"]["pomdp_metrics"]["total_observation_space_size"] += pomdp_metrics.get("observation_space_size", 0)
                        type_check_results["summary"]["pomdp_metrics"]["total_action_space_size"] += pomdp_metrics.get("action_space_size", 0)
                        
                        if inner_pomdp_analysis.get("overall_valid", False):
                            type_check_results["summary"]["pomdp_metrics"]["pomdp_models_valid"] += 1
                        
            except Exception as e:
                logger.warning(f"POMDP analysis failed for {file_name}: {e}")
                pomdp_analysis = {"error": str(e)}

        type_errors: List[str] = []
        warnings: List[str] = []

        for var in variables:
            if not var.get("description"):
                warnings.append(f"Variable '{var['name']}' missing description")

        for var in variables:
            dimensions = var.get("dimensions", [])
            if any(d <= 0 for d in dimensions):
                type_errors.append(
                    f"Variable '{var['name']}' has invalid dimensions: {dimensions}"
                )

        var_names = {var.get("name") for var in variables}
        connected_vars: set[str] = set()
        for conn in connections:
            connected_vars.update(conn.get("source_variables", []))
            connected_vars.update(conn.get("target_variables", []))

        orphaned = var_names - connected_vars
        
        # Filter out variables that are allowed to be standalone/global
        # These variables typically represent global state, time, or computed quantities
        allowed_standalone = set()
        for var_name in orphaned:
            # Find variable info (variables might be a list of dicts)
            var_info = {}
            if isinstance(variables, list):
                for var in variables:
                    if var.get("name") == var_name:
                        var_info = var
                        break
            elif isinstance(variables, dict):
                var_info = variables.get(var_name, {})
            
            var_comment = var_info.get('description', '').lower() if var_info else ''
            
            # Time variables are typically global/standalone
            if var_name.lower() in ['t', 'time', 'step', 'timestep']:
                allowed_standalone.add(var_name)
                continue
            
            # Free energy variables are often computed quantities that don't need explicit connections
            if var_name.upper() in ['F', 'FREE_ENERGY', 'VARIATIONAL_FREE_ENERGY']:
                allowed_standalone.add(var_name)
                continue
            
            # Check comment patterns for standalone indicators
            standalone_patterns = [
                'time', 'global', 'computed', 'derived', 'output',
                'free energy', 'standalone', 'independent', 'discrete time'
            ]
            
            if any(pattern in var_comment for pattern in standalone_patterns):
                allowed_standalone.add(var_name)
                continue
        
        actual_orphaned = orphaned - allowed_standalone
        if actual_orphaned:
            warnings.append(f"Orphaned variables: {list(actual_orphaned)}")

        file_analysis: Dict[str, Any] = {
            "file_name": file_name,
            "file_path": file_result.get("file_path"),
            "parsed_model_file": str(parsed_model_file),
            "type_analysis": type_analysis,
            "connection_analysis": connection_analysis,
            "complexity_analysis": complexity_analysis,
            "type_errors": type_errors,
            "warnings": warnings,
            "model_info": file_result.get("model_info", {}),
        }
        
        # Add POMDP analysis if available
        if pomdp_analysis:
            file_analysis["pomdp_analysis"] = pomdp_analysis

        type_check_results["files_analyzed"].append(file_analysis)
        all_variables.extend(variables)
        all_connections.extend(connections)

        type_check_results["summary"]["total_files"] += 1
        if not type_errors:
            type_check_results["summary"]["valid_files"] += 1
        type_check_results["summary"]["type_errors"] += len(type_errors)
        type_check_results["summary"]["warnings"] += len(warnings)

        logger.info(
            f"Analyzed {file_name}: {len(variables)} variables, {len(connections)} connections, "
            f"{len(type_errors)} errors, {len(warnings)} warnings"
        )

    if all_variables:
        type_check_results["global_analysis"]["type_analysis"] = analyze_variable_types(all_variables)
    if all_connections:
        type_check_results["global_analysis"]["connection_analysis"] = analyze_connections(all_connections)
    if all_variables and all_connections:
        type_check_results["global_analysis"]["complexity_analysis"] = (
            estimate_computational_complexity(
                type_check_results["global_analysis"]["type_analysis"],
                type_check_results["global_analysis"]["connection_analysis"],
            )
        )

    results_file = output_dir / "type_check_results.json"
    with open(results_file, "w") as f:
        json.dump(type_check_results, f, indent=2, default=str)

    summary_file = output_dir / "type_check_summary.json"
    with open(summary_file, "w") as f:
        json.dump(type_check_results["summary"], f, indent=2)

    global_analysis_file = output_dir / "global_type_analysis.json"
    with open(global_analysis_file, "w") as f:
        json.dump(type_check_results["global_analysis"], f, indent=2, default=str)
    
    # Generate POMDP-specific summary if in POMDP mode
    if pomdp_mode:
        try:
            from type_checker.processor import GNNTypeChecker
            pomdp_checker = GNNTypeChecker(pomdp_mode=True, ontology_file=ontology_file)
            
            # Extract POMDP analyses from files_analyzed
            pomdp_analyses = []
            for file_analysis in type_check_results.get("files_analyzed", []):
                if "pomdp_analysis" in file_analysis:
                    pomdp_analyses.append(file_analysis["pomdp_analysis"])
            
            # Create a results structure for summary generation
            pomdp_results = {
                "pomdp_analysis": pomdp_analyses,
                "summary_statistics": type_check_results.get("summary", {})
            }
            
            pomdp_summary = pomdp_checker.get_pomdp_analysis_summary(pomdp_results)
            
            pomdp_summary_file = output_dir / "pomdp_analysis_summary.md"
            with open(pomdp_summary_file, "w") as f:
                f.write(pomdp_summary)
            
            logger.info("POMDP analysis summary generated")
        except Exception as e:
            logger.warning(f"Failed to generate POMDP summary: {e}")
    
    # Copy results to prerequisite directory for validation step
    import shutil
    shutil.copy2(results_file, prereq_dir / "type_check_results.json")
    shutil.copy2(summary_file, prereq_dir / "type_check_summary.json")
    shutil.copy2(global_analysis_file, prereq_dir / "global_type_analysis.json")
    
    if pomdp_mode:
        try:
            shutil.copy2(pomdp_summary_file, prereq_dir / "pomdp_analysis_summary.md")
        except:
            pass  # POMDP summary might not have been generated

    success = type_check_results["summary"].get("valid_files", 0) > 0
    warnings_count = type_check_results["summary"].get("warnings", 0)
    
    if success:
        if warnings_count > 0:
            log_step_warning(
                logger,
                f"Type checking completed: {type_check_results['summary']['valid_files']} valid files, "
                f"{warnings_count} warnings",
            )
        else:
            log_step_success(
                logger,
                f"Type checking completed: {type_check_results['summary']['valid_files']} valid files, "
                f"{warnings_count} warnings",
            )
    else:
        log_step_error(logger, "No valid files found during type checking")

    return success


def main():
    return run_script()


if __name__ == "__main__":
    sys.exit(main())
