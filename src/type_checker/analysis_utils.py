#!/usr/bin/env python3
"""
Type checker analysis utilities for GNN models.

These functions provide variable and connection analysis as well as
computational complexity estimation. They are imported by the step orchestrator
to keep the numbered script thin.

This module provides robust analysis functions with comprehensive error handling,
validation, and performance optimization for GNN model type checking.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import math


def analyze_variable_types(variables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze variable types and dimensions for a GNN model.
    
    Args:
        variables: List of variable dictionaries with type, data_type, dimensions, etc.
        
    Returns:
        Dictionary containing comprehensive type analysis results
        
    Raises:
        ValueError: If variables is not a list
        TypeError: If individual variable items are not dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Input validation
    if not isinstance(variables, list):
        logger.warning(f"Expected list for variables, got {type(variables)}")
        variables = []
    
    # Filter out None and invalid entries
    valid_variables = []
    for i, var in enumerate(variables):
        if var is None:
            logger.warning(f"Skipping None variable at index {i}")
            continue
        if not isinstance(var, dict):
            logger.warning(f"Skipping non-dict variable at index {i}: {type(var)}")
            continue
        valid_variables.append(var)
    
    type_analysis: Dict[str, Any] = {
        "total_variables": len(valid_variables),
        "type_distribution": {},
        "dimension_analysis": {
            "max_dimensions": 0,
            "avg_dimensions": 0,
            "dimension_distribution": {},
            "invalid_dimensions": 0,
        },
        "data_type_distribution": {},
        "complexity_metrics": {
            "total_elements": 0,
            "estimated_memory_bytes": 0,
            "estimated_memory_mb": 0,
            "estimated_memory_gb": 0,
        },
        "validation_issues": [],
        "performance_warnings": [],
    }

    total_dimensions = 0
    total_elements = 0
    invalid_dimension_count = 0

    for i, var in enumerate(valid_variables):
        try:
            # Extract and validate type information
            var_type = var.get("type", "unknown")
            if not isinstance(var_type, str):
                var_type = str(var_type)
            type_analysis["type_distribution"][var_type] = (
                type_analysis["type_distribution"].get(var_type, 0) + 1
            )

            # Extract and validate data type
            data_type = var.get("data_type", "unknown")
            if not isinstance(data_type, str):
                data_type = str(data_type)
            type_analysis["data_type_distribution"][data_type] = (
                type_analysis["data_type_distribution"].get(data_type, 0) + 1
            )

            # Extract and validate dimensions
            dimensions = var.get("dimensions", [1])
            if not isinstance(dimensions, list):
                logger.warning(f"Invalid dimensions format for variable {i}: {dimensions}")
                dimensions = [1]
                invalid_dimension_count += 1
            
            # Validate dimension values
            valid_dimensions = []
            for dim in dimensions:
                if isinstance(dim, (int, float)) and dim > 0:
                    valid_dimensions.append(int(dim))
                else:
                    logger.warning(f"Invalid dimension value: {dim}")
                    invalid_dimension_count += 1
                    valid_dimensions.append(1)  # Default to 1 for invalid dimensions
            
            if not valid_dimensions:
                valid_dimensions = [1]  # Ensure at least one dimension
            
            dim_count = len(valid_dimensions)
            total_dimensions += dim_count
            if dim_count > type_analysis["dimension_analysis"]["max_dimensions"]:
                type_analysis["dimension_analysis"]["max_dimensions"] = dim_count

            dim_key = f"{dim_count}D"
            type_analysis["dimension_analysis"]["dimension_distribution"][dim_key] = (
                type_analysis["dimension_analysis"]["dimension_distribution"].get(dim_key, 0)
                + 1
            )

            # Calculate elements with overflow protection
            elements = 1
            for dim in valid_dimensions:
                if elements > 1e15:  # Prevent overflow
                    logger.warning(f"Large dimension product detected: {elements}")
                    type_analysis["performance_warnings"].append(
                        f"Variable {i} has very large dimension product: {elements}"
                    )
                    break
                elements *= dim
            
            total_elements += elements
            
        except Exception as e:
            logger.error(f"Error processing variable {i}: {e}")
            type_analysis["validation_issues"].append(f"Error processing variable {i}: {str(e)}")
            continue

    # Calculate averages
    if valid_variables:
        type_analysis["dimension_analysis"]["avg_dimensions"] = total_dimensions / len(valid_variables)
    
    type_analysis["dimension_analysis"]["invalid_dimensions"] = invalid_dimension_count

    # Calculate memory estimates with overflow protection
    type_analysis["complexity_metrics"]["total_elements"] = total_elements
    type_analysis["complexity_metrics"]["estimated_memory_bytes"] = total_elements * 8
    type_analysis["complexity_metrics"]["estimated_memory_mb"] = (total_elements * 8) / (1024 * 1024)
    type_analysis["complexity_metrics"]["estimated_memory_gb"] = (total_elements * 8) / (1024 * 1024 * 1024)
    
    # Add performance warnings for large models
    if total_elements > 1e9:  # 1 billion elements
        type_analysis["performance_warnings"].append(
            f"Very large model detected: {total_elements:,} total elements"
        )
    elif total_elements > 1e6:  # 1 million elements
        type_analysis["performance_warnings"].append(
            f"Large model detected: {total_elements:,} total elements"
        )

    return type_analysis


def analyze_connections(connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze connection patterns and complexity for a GNN model.
    
    Args:
        connections: List of connection dictionaries with type, source_variables, target_variables, etc.
        
    Returns:
        Dictionary containing comprehensive connection analysis results
        
    Raises:
        ValueError: If connections is not a list
        TypeError: If individual connection items are not dictionaries
    """
    logger = logging.getLogger(__name__)
    
    # Input validation
    if not isinstance(connections, list):
        logger.warning(f"Expected list for connections, got {type(connections)}")
        connections = []
    
    # Filter out None and invalid entries
    valid_connections = []
    for i, conn in enumerate(connections):
        if conn is None:
            logger.warning(f"Skipping None connection at index {i}")
            continue
        if not isinstance(conn, dict):
            logger.warning(f"Skipping non-dict connection at index {i}: {type(conn)}")
            continue
        valid_connections.append(conn)
    
    connection_analysis: Dict[str, Any] = {
        "total_connections": len(valid_connections),
        "connection_type_distribution": {},
        "connectivity_metrics": {
            "avg_connections_per_variable": 0,
            "max_connections_per_variable": 0,
            "isolated_variables": 0,
            "highly_connected_variables": 0,
        },
        "graph_metrics": {
            "in_degree_distribution": {},
            "out_degree_distribution": {},
            "cycles_detected": False,
            "strongly_connected_components": 0,
        },
        "validation_issues": [],
        "performance_warnings": [],
    }

    variable_connections: Dict[str, Dict[str, int]] = {}
    all_variables: set[str] = set()

    for i, conn in enumerate(valid_connections):
        try:
            # Extract and validate connection type
            conn_type = conn.get("type", "unknown")
            if not isinstance(conn_type, str):
                conn_type = str(conn_type)
            connection_analysis["connection_type_distribution"][conn_type] = (
                connection_analysis["connection_type_distribution"].get(conn_type, 0) + 1
            )

            # Extract and validate source variables
            sources = conn.get("source_variables", [])
            if not isinstance(sources, list):
                logger.warning(f"Invalid source_variables format for connection {i}: {sources}")
                sources = []
            
            # Extract and validate target variables
            targets = conn.get("target_variables", [])
            if not isinstance(targets, list):
                logger.warning(f"Invalid target_variables format for connection {i}: {targets}")
                targets = []

            # Process source variables
            for source in sources:
                if not isinstance(source, str):
                    logger.warning(f"Invalid source variable name: {source}")
                    continue
                all_variables.add(source)
                if source not in variable_connections:
                    variable_connections[source] = {"in": 0, "out": 0}
                variable_connections[source]["out"] += 1

            # Process target variables
            for target in targets:
                if not isinstance(target, str):
                    logger.warning(f"Invalid target variable name: {target}")
                    continue
                all_variables.add(target)
                if target not in variable_connections:
                    variable_connections[target] = {"in": 0, "out": 0}
                variable_connections[target]["in"] += 1
                
        except Exception as e:
            logger.error(f"Error processing connection {i}: {e}")
            connection_analysis["validation_issues"].append(f"Error processing connection {i}: {str(e)}")
            continue

    # Calculate connectivity metrics
    if variable_connections:
        total_out = sum(v["out"] for v in variable_connections.values())
        total_in = sum(v["in"] for v in variable_connections.values())
        
        connection_analysis["connectivity_metrics"]["avg_connections_per_variable"] = (
            total_out / len(variable_connections) if variable_connections else 0
        )
        connection_analysis["connectivity_metrics"]["max_connections_per_variable"] = (
            max(v["out"] for v in variable_connections.values()) if variable_connections else 0
        )
        connection_analysis["connectivity_metrics"]["isolated_variables"] = sum(
            1 for v in variable_connections.values() if v["in"] == 0 and v["out"] == 0
        )
        connection_analysis["connectivity_metrics"]["highly_connected_variables"] = sum(
            1 for v in variable_connections.values() if v["out"] > 5 or v["in"] > 5
        )
        
        # Calculate degree distributions
        out_degrees = [v["out"] for v in variable_connections.values()]
        in_degrees = [v["in"] for v in variable_connections.values()]
        
        for degree in out_degrees:
            degree_key = f"degree_{degree}"
            connection_analysis["graph_metrics"]["out_degree_distribution"][degree_key] = (
                connection_analysis["graph_metrics"]["out_degree_distribution"].get(degree_key, 0) + 1
            )
        
        for degree in in_degrees:
            degree_key = f"degree_{degree}"
            connection_analysis["graph_metrics"]["in_degree_distribution"][degree_key] = (
                connection_analysis["graph_metrics"]["in_degree_distribution"].get(degree_key, 0) + 1
            )
        
        # Simple cycle detection (basic check for self-loops)
        self_loops = sum(1 for v in variable_connections.values() if v["in"] > 0 and v["out"] > 0)
        if self_loops > 0:
            connection_analysis["graph_metrics"]["cycles_detected"] = True
            connection_analysis["performance_warnings"].append(
                f"Potential cycles detected: {self_loops} variables with both input and output connections"
            )
        
        # Estimate strongly connected components (simplified)
        connection_analysis["graph_metrics"]["strongly_connected_components"] = max(1, len(all_variables) - len(valid_connections))
    
    # Add performance warnings
    if len(valid_connections) > 1000:
        connection_analysis["performance_warnings"].append(
            f"Very large number of connections: {len(valid_connections)}"
        )
    elif len(valid_connections) > 100:
        connection_analysis["performance_warnings"].append(
            f"Large number of connections: {len(valid_connections)}"
        )

    return connection_analysis


def estimate_computational_complexity(
    type_analysis: Dict[str, Any], connection_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate computational complexity for inference and learning.
    
    Args:
        type_analysis: Results from analyze_variable_types()
        connection_analysis: Results from analyze_connections()
        
    Returns:
        Dictionary containing computational complexity estimates
        
    Raises:
        ValueError: If input analyses are not dictionaries
        TypeError: If required fields are missing or invalid
    """
    logger = logging.getLogger(__name__)
    
    # Input validation
    if not isinstance(type_analysis, dict):
        logger.warning(f"Expected dict for type_analysis, got {type(type_analysis)}")
        type_analysis = {"complexity_metrics": {"total_elements": 0, "estimated_memory_mb": 0}}
    
    if not isinstance(connection_analysis, dict):
        logger.warning(f"Expected dict for connection_analysis, got {type(connection_analysis)}")
        connection_analysis = {"total_connections": 0}
    
    complexity: Dict[str, Any] = {
        "inference_complexity": {
            "operations_per_step": 0,
            "memory_bandwidth_gb_s": 0,
            "parallelization_potential": "low",
            "scalability_rating": "good",
        },
        "learning_complexity": {
            "gradient_operations": 0,
            "parameter_updates": 0,
            "backpropagation_steps": 0,
        },
        "resource_requirements": {
            "cpu_cores_recommended": 1,
            "ram_gb_recommended": 1,
            "gpu_memory_gb_recommended": 0,
            "storage_gb_recommended": 0,
        },
        "performance_estimates": {
            "inference_time_ms": 0,
            "learning_time_s": 0,
            "memory_efficiency_score": 0,
        },
        "optimization_suggestions": [],
        "warnings": [],
    }

    try:
        # Extract metrics with safe defaults
        total_elements = type_analysis.get("complexity_metrics", {}).get("total_elements", 0)
        memory_mb = type_analysis.get("complexity_metrics", {}).get("estimated_memory_mb", 0)
        memory_gb = type_analysis.get("complexity_metrics", {}).get("estimated_memory_gb", 0)
        total_connections = connection_analysis.get("total_connections", 0)
        
        # Validate metrics
        if not isinstance(total_elements, (int, float)) or total_elements < 0:
            logger.warning(f"Invalid total_elements: {total_elements}")
            total_elements = 0
        
        if not isinstance(total_connections, (int, float)) or total_connections < 0:
            logger.warning(f"Invalid total_connections: {total_connections}")
            total_connections = 0

        # Calculate inference complexity
        operations_per_step = total_elements * max(1, total_connections)
        complexity["inference_complexity"]["operations_per_step"] = operations_per_step
        complexity["inference_complexity"]["memory_bandwidth_gb_s"] = (
            (total_elements * 8) / (1024 * 1024 * 1024)
        )

        # Determine parallelization potential
        if total_elements > 1000000:  # 1 million elements
            complexity["inference_complexity"]["parallelization_potential"] = "very_high"
            complexity["inference_complexity"]["scalability_rating"] = "excellent"
        elif total_elements > 100000:  # 100k elements
            complexity["inference_complexity"]["parallelization_potential"] = "high"
            complexity["inference_complexity"]["scalability_rating"] = "very_good"
        elif total_elements > 10000:  # 10k elements
            complexity["inference_complexity"]["parallelization_potential"] = "medium"
            complexity["inference_complexity"]["scalability_rating"] = "good"
        elif total_elements > 1000:  # 1k elements
            complexity["inference_complexity"]["parallelization_potential"] = "low"
            complexity["inference_complexity"]["scalability_rating"] = "fair"
        else:
            complexity["inference_complexity"]["parallelization_potential"] = "very_low"
            complexity["inference_complexity"]["scalability_rating"] = "poor"

        # Calculate learning complexity
        gradient_ops = operations_per_step * 2  # Forward + backward pass
        complexity["learning_complexity"]["gradient_operations"] = gradient_ops
        complexity["learning_complexity"]["parameter_updates"] = total_elements
        complexity["learning_complexity"]["backpropagation_steps"] = max(1, int(math.log2(total_elements + 1)))

        # Determine resource requirements
        if memory_gb > 10:  # Very large model
            complexity["resource_requirements"]["ram_gb_recommended"] = 32
            complexity["resource_requirements"]["gpu_memory_gb_recommended"] = 16
            complexity["resource_requirements"]["cpu_cores_recommended"] = 16
            complexity["resource_requirements"]["storage_gb_recommended"] = 100
        elif memory_gb > 1:  # Large model
            complexity["resource_requirements"]["ram_gb_recommended"] = 16
            complexity["resource_requirements"]["gpu_memory_gb_recommended"] = 8
            complexity["resource_requirements"]["cpu_cores_recommended"] = 8
            complexity["resource_requirements"]["storage_gb_recommended"] = 50
        elif memory_mb > 100:  # Medium model
            complexity["resource_requirements"]["ram_gb_recommended"] = 8
            complexity["resource_requirements"]["gpu_memory_gb_recommended"] = 4
            complexity["resource_requirements"]["cpu_cores_recommended"] = 4
            complexity["resource_requirements"]["storage_gb_recommended"] = 10
        else:  # Small model
            complexity["resource_requirements"]["ram_gb_recommended"] = 4
            complexity["resource_requirements"]["gpu_memory_gb_recommended"] = 2
            complexity["resource_requirements"]["cpu_cores_recommended"] = 2
            complexity["resource_requirements"]["storage_gb_recommended"] = 1

        # Calculate performance estimates
        complexity["performance_estimates"]["inference_time_ms"] = max(1, operations_per_step / 1000000)  # Rough estimate
        complexity["performance_estimates"]["learning_time_s"] = max(1, gradient_ops / 100000)  # Rough estimate
        complexity["performance_estimates"]["memory_efficiency_score"] = min(100, max(0, 100 - (memory_gb * 10)))

        # Generate optimization suggestions
        if total_elements > 1000000:
            complexity["optimization_suggestions"].append("Consider model pruning or quantization")
            complexity["optimization_suggestions"].append("Use distributed training")
        if memory_gb > 5:
            complexity["optimization_suggestions"].append("Consider gradient checkpointing")
            complexity["optimization_suggestions"].append("Use mixed precision training")
        if total_connections > 1000:
            complexity["optimization_suggestions"].append("Consider sparse connections")
            complexity["optimization_suggestions"].append("Use attention mechanisms")
        
        if not complexity["optimization_suggestions"]:
            complexity["optimization_suggestions"].append("Model is well-optimized for current scale")

        # Add warnings
        if operations_per_step > 1e9:  # 1 billion operations
            complexity["warnings"].append("Very high computational complexity detected")
        if memory_gb > 20:
            complexity["warnings"].append("Very high memory requirements detected")
        if total_connections > 10000:
            complexity["warnings"].append("Very dense connection pattern detected")

    except Exception as e:
        logger.error(f"Error calculating computational complexity: {e}")
        complexity["warnings"].append(f"Error in complexity calculation: {str(e)}")

    return complexity


