#!/usr/bin/env python3
"""
Type checker analysis utilities for GNN models.

These functions provide variable and connection analysis as well as
computational complexity estimation. They are imported by the step orchestrator
to keep the numbered script thin.
"""

from typing import Any, Dict, List


def analyze_variable_types(variables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze variable types and dimensions for a GNN model."""
    type_analysis: Dict[str, Any] = {
        "total_variables": len(variables),
        "type_distribution": {},
        "dimension_analysis": {
            "max_dimensions": 0,
            "avg_dimensions": 0,
            "dimension_distribution": {},
        },
        "data_type_distribution": {},
        "complexity_metrics": {
            "total_elements": 0,
            "estimated_memory_bytes": 0,
            "estimated_memory_mb": 0,
        },
    }

    total_dimensions = 0
    total_elements = 0

    for var in variables:
        var_type = var.get("type", "unknown")
        type_analysis["type_distribution"][var_type] = (
            type_analysis["type_distribution"].get(var_type, 0) + 1
        )

        data_type = var.get("data_type", "unknown")
        type_analysis["data_type_distribution"][data_type] = (
            type_analysis["data_type_distribution"].get(data_type, 0) + 1
        )

        dimensions = var.get("dimensions", [1])
        dim_count = len(dimensions)
        total_dimensions += dim_count
        if dim_count > type_analysis["dimension_analysis"]["max_dimensions"]:
            type_analysis["dimension_analysis"]["max_dimensions"] = dim_count

        dim_key = f"{dim_count}D"
        type_analysis["dimension_analysis"]["dimension_distribution"][dim_key] = (
            type_analysis["dimension_analysis"]["dimension_distribution"].get(dim_key, 0)
            + 1
        )

        elements = 1
        for dim in dimensions:
            elements *= dim
        total_elements += elements

    if variables:
        type_analysis["dimension_analysis"]["avg_dimensions"] = total_dimensions / len(variables)

    type_analysis["complexity_metrics"]["total_elements"] = total_elements
    type_analysis["complexity_metrics"]["estimated_memory_bytes"] = total_elements * 8
    type_analysis["complexity_metrics"]["estimated_memory_mb"] = (
        total_elements * 8
    ) / (1024 * 1024)

    return type_analysis


def analyze_connections(connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze connection patterns and complexity for a GNN model."""
    connection_analysis: Dict[str, Any] = {
        "total_connections": len(connections),
        "connection_type_distribution": {},
        "connectivity_metrics": {
            "avg_connections_per_variable": 0,
            "max_connections_per_variable": 0,
            "isolated_variables": 0,
        },
        "graph_metrics": {
            "in_degree_distribution": {},
            "out_degree_distribution": {},
            "cycles_detected": False,
        },
    }

    variable_connections: Dict[str, Dict[str, int]] = {}

    for conn in connections:
        conn_type = conn.get("type", "unknown")
        connection_analysis["connection_type_distribution"][conn_type] = (
            connection_analysis["connection_type_distribution"].get(conn_type, 0) + 1
        )

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

    if variable_connections:
        total_out = sum(v["out"] for v in variable_connections.values())
        connection_analysis["connectivity_metrics"]["avg_connections_per_variable"] = (
            total_out / len(variable_connections)
        )
        connection_analysis["connectivity_metrics"]["max_connections_per_variable"] = (
            max(v["out"] for v in variable_connections.values())
        )
        connection_analysis["connectivity_metrics"]["isolated_variables"] = sum(
            1 for v in variable_connections.values() if v["in"] == 0 and v["out"] == 0
        )

    return connection_analysis


def estimate_computational_complexity(
    type_analysis: Dict[str, Any], connection_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Estimate computational complexity for inference and learning."""
    complexity: Dict[str, Any] = {
        "inference_complexity": {
            "operations_per_step": 0,
            "memory_bandwidth_gb_s": 0,
            "parallelization_potential": "low",
        },
        "learning_complexity": {
            "gradient_operations": 0,
            "parameter_updates": 0,
        },
        "resource_requirements": {
            "cpu_cores_recommended": 1,
            "ram_gb_recommended": 1,
            "gpu_memory_gb_recommended": 0,
        },
    }

    total_elements = type_analysis["complexity_metrics"].get("total_elements", 0)
    total_connections = connection_analysis.get("total_connections", 0)

    complexity["inference_complexity"]["operations_per_step"] = (
        total_elements * total_connections
    )
    complexity["inference_complexity"]["memory_bandwidth_gb_s"] = (
        (total_elements * 8) / (1024 * 1024 * 1024)
    )

    if total_elements > 1000:
        complexity["inference_complexity"]["parallelization_potential"] = "high"
    elif total_elements > 100:
        complexity["inference_complexity"]["parallelization_potential"] = "medium"

    memory_mb = type_analysis["complexity_metrics"].get("estimated_memory_mb", 0)
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


