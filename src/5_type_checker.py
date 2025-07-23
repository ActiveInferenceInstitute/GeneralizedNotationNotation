#!/usr/bin/env python3
"""
Step 5: Type Checking and Validation

This step performs type checking and validation on GNN files.
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
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def analyze_variable_types(variables: List[Dict]) -> Dict[str, Any]:
    """Analyze variable types and dimensions."""
    type_analysis = {
        "total_variables": len(variables),
        "type_distribution": {},
        "dimension_analysis": {
            "max_dimensions": 0,
            "avg_dimensions": 0,
            "dimension_distribution": {}
        },
        "data_type_distribution": {},
        "complexity_metrics": {
            "total_elements": 0,
            "estimated_memory_bytes": 0,
            "estimated_memory_mb": 0
        }
    }
    
    total_dimensions = 0
    total_elements = 0
    
    for var in variables:
        # Type distribution
        var_type = var.get("type", "unknown")
        type_analysis["type_distribution"][var_type] = type_analysis["type_distribution"].get(var_type, 0) + 1
        
        # Data type distribution
        data_type = var.get("data_type", "unknown")
        type_analysis["data_type_distribution"][data_type] = type_analysis["data_type_distribution"].get(data_type, 0) + 1
        
        # Dimension analysis
        dimensions = var.get("dimensions", [1])
        dim_count = len(dimensions)
        total_dimensions += dim_count
        type_analysis["dimension_analysis"]["max_dimensions"] = max(
            type_analysis["dimension_analysis"]["max_dimensions"], 
            dim_count
        )
        
        # Dimension distribution
        dim_key = f"{dim_count}D"
        type_analysis["dimension_analysis"]["dimension_distribution"][dim_key] = \
            type_analysis["dimension_analysis"]["dimension_distribution"].get(dim_key, 0) + 1
        
        # Calculate elements
        elements = 1
        for dim in dimensions:
            elements *= dim
        total_elements += elements
    
    # Calculate averages
    if variables:
        type_analysis["dimension_analysis"]["avg_dimensions"] = total_dimensions / len(variables)
    
    # Memory estimation (assuming 8 bytes per element for float64)
    type_analysis["complexity_metrics"]["total_elements"] = total_elements
    type_analysis["complexity_metrics"]["estimated_memory_bytes"] = total_elements * 8
    type_analysis["complexity_metrics"]["estimated_memory_mb"] = (total_elements * 8) / (1024 * 1024)
    
    return type_analysis

def analyze_connections(connections: List[Dict]) -> Dict[str, Any]:
    """Analyze connection patterns and complexity."""
    connection_analysis = {
        "total_connections": len(connections),
        "connection_type_distribution": {},
        "connectivity_metrics": {
            "avg_connections_per_variable": 0,
            "max_connections_per_variable": 0,
            "isolated_variables": 0
        },
        "graph_metrics": {
            "in_degree_distribution": {},
            "out_degree_distribution": {},
            "cycles_detected": False
        }
    }
    
    # Track variable connectivity
    variable_connections = {}
    
    for conn in connections:
        conn_type = conn.get("type", "unknown")
        connection_analysis["connection_type_distribution"][conn_type] = \
            connection_analysis["connection_type_distribution"].get(conn_type, 0) + 1
        
        # Track source and target variables
        sources = conn.get("source_variables", [])
        targets = conn.get("target_variables", [])
        
        for source in sources:
            if source not in variable_connections:
                variable_connections[source] = {"in": 0, "out": 0}
            variable_connections[source]["out"] += 1
        
        for target in targets:
            if target not in variable_connections:
                variable_connections[target] = {"in": 0, "out": 0}
            variable_connections[target]["in"] += 1
    
    # Calculate connectivity metrics
    if variable_connections:
        total_connections = sum(v["out"] for v in variable_connections.values())
        connection_analysis["connectivity_metrics"]["avg_connections_per_variable"] = \
            total_connections / len(variable_connections)
        connection_analysis["connectivity_metrics"]["max_connections_per_variable"] = \
            max(v["out"] for v in variable_connections.values())
        connection_analysis["connectivity_metrics"]["isolated_variables"] = \
            sum(1 for v in variable_connections.values() if v["in"] == 0 and v["out"] == 0)
    
    return connection_analysis

def estimate_computational_complexity(type_analysis: Dict, connection_analysis: Dict) -> Dict[str, Any]:
    """Estimate computational complexity of the model."""
    complexity = {
        "inference_complexity": {
            "operations_per_step": 0,
            "memory_bandwidth_gb_s": 0,
            "parallelization_potential": "low"
        },
        "learning_complexity": {
            "gradient_operations": 0,
            "parameter_updates": 0
        },
        "resource_requirements": {
            "cpu_cores_recommended": 1,
            "ram_gb_recommended": 1,
            "gpu_memory_gb_recommended": 0
        }
    }
    
    # Estimate operations based on variables and connections
    total_elements = type_analysis["complexity_metrics"]["total_elements"]
    total_connections = connection_analysis["total_connections"]
    
    # Basic inference operations (matrix operations)
    complexity["inference_complexity"]["operations_per_step"] = total_elements * total_connections
    
    # Memory bandwidth (assuming each element accessed once per step)
    complexity["inference_complexity"]["memory_bandwidth_gb_s"] = \
        (total_elements * 8) / (1024 * 1024 * 1024)  # GB/s
    
    # Parallelization potential
    if total_elements > 1000:
        complexity["inference_complexity"]["parallelization_potential"] = "high"
    elif total_elements > 100:
        complexity["inference_complexity"]["parallelization_potential"] = "medium"
    
    # Resource recommendations
    memory_mb = type_analysis["complexity_metrics"]["estimated_memory_mb"]
    if memory_mb > 1000:
        complexity["resource_requirements"]["ram_gb_recommended"] = 4
        complexity["resource_requirements"]["gpu_memory_gb_recommended"] = 2
    elif memory_mb > 100:
        complexity["resource_requirements"]["ram_gb_recommended"] = 2
    else:
        complexity["resource_requirements"]["ram_gb_recommended"] = 1
    
    if complexity["inference_complexity"]["parallelization_potential"] == "high":
        complexity["resource_requirements"]["cpu_cores_recommended"] = 4
    
    return complexity

def main():
    """Main type checking function."""
    args = EnhancedArgumentParser.parse_step_arguments("5_type_checker.py")
    
    # Setup logging
    logger = setup_step_logging("type_checker", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("5_type_checker.py", config.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Performing type checking and validation")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", config.base_output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return 1
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Process each parsed file
        type_check_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "strict_mode": args.strict,
            "resource_estimation": args.estimate_resources,
            "files_analyzed": [],
            "summary": {
                "total_files": 0,
                "valid_files": 0,
                "type_errors": 0,
                "warnings": 0
            },
            "global_analysis": {
                "type_analysis": {},
                "connection_analysis": {},
                "complexity_analysis": {}
            }
        }
        
        all_variables = []
        all_connections = []
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Analyzing types for: {file_name}")
            
            # Load the parsed model from the individual JSON file
            parsed_model_file = Path(file_result["parsed_model_file"])
            if not parsed_model_file.exists():
                logger.warning(f"Parsed model file not found: {parsed_model_file}")
                continue
            
            try:
                with open(parsed_model_file, 'r') as f:
                    parsed_model = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load parsed model from {parsed_model_file}: {e}")
                continue
            
            # Extract variables and connections from the parsed model
            variables = parsed_model.get("variables", [])
            connections = parsed_model.get("connections", [])
            
            # Analyze variables
            type_analysis = analyze_variable_types(variables)
            
            # Analyze connections
            connection_analysis = analyze_connections(connections)
            
            # Estimate complexity
            complexity_analysis = estimate_computational_complexity(type_analysis, connection_analysis)
            
            # Check for type issues
            type_errors = []
            warnings = []
            
            # Check for missing descriptions
            for var in variables:
                if not var.get("description"):
                    warnings.append(f"Variable '{var['name']}' missing description")
            
            # Check for unusual dimensions
            for var in variables:
                dimensions = var.get("dimensions", [])
                if any(d <= 0 for d in dimensions):
                    type_errors.append(f"Variable '{var['name']}' has invalid dimensions: {dimensions}")
            
            # Check for orphaned variables
            var_names = {var["name"] for var in variables}
            connected_vars = set()
            for conn in connections:
                connected_vars.update(conn.get("source_variables", []))
                connected_vars.update(conn.get("target_variables", []))
            
            orphaned = var_names - connected_vars
            if orphaned:
                warnings.append(f"Orphaned variables: {list(orphaned)}")
            
            file_analysis = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "parsed_model_file": str(parsed_model_file),
                "type_analysis": type_analysis,
                "connection_analysis": connection_analysis,
                "complexity_analysis": complexity_analysis,
                "type_errors": type_errors,
                "warnings": warnings,
                "model_info": file_result.get("model_info", {})
            }
            
            type_check_results["files_analyzed"].append(file_analysis)
            
            # Accumulate global statistics
            all_variables.extend(variables)
            all_connections.extend(connections)
            
            type_check_results["summary"]["total_files"] += 1
            if not type_errors:
                type_check_results["summary"]["valid_files"] += 1
            type_check_results["summary"]["type_errors"] += len(type_errors)
            type_check_results["summary"]["warnings"] += len(warnings)
            
            logger.info(f"Analyzed {file_name}: {len(variables)} variables, {len(connections)} connections, {len(type_errors)} errors, {len(warnings)} warnings")
        
        # Perform global analysis
        if all_variables:
            type_check_results["global_analysis"]["type_analysis"] = analyze_variable_types(all_variables)
        if all_connections:
            type_check_results["global_analysis"]["connection_analysis"] = analyze_connections(all_connections)
        if all_variables and all_connections:
            type_check_results["global_analysis"]["complexity_analysis"] = estimate_computational_complexity(
                type_check_results["global_analysis"]["type_analysis"],
                type_check_results["global_analysis"]["connection_analysis"]
            )
        
        # Save results
        results_file = output_dir / "type_check_results.json"
        with open(results_file, 'w') as f:
            json.dump(type_check_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "type_check_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(type_check_results["summary"], f, indent=2)
        
        # Save global analysis
        global_analysis_file = output_dir / "global_type_analysis.json"
        with open(global_analysis_file, 'w') as f:
            json.dump(type_check_results["global_analysis"], f, indent=2, default=str)
        
        # Determine success
        success = type_check_results["summary"]["valid_files"] > 0
        
        if success:
            log_step_success(logger, f"Type checking completed: {type_check_results['summary']['valid_files']} valid files, {type_check_results['summary']['warnings']} warnings")
            return 0
        else:
            log_step_error(logger, "No valid files found during type checking")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
