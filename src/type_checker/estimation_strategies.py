#!/usr/bin/env python3
"""
Estimation Strategy Functions for GNN Resource Estimator.

Standalone functions extracted from GNNResourceEstimator that compute
memory, inference, storage, FLOPS, and complexity estimates for GNN models.
"""

import logging
import math
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def estimate_memory(
    variables: Dict[str, Any],
    memory_factors: Dict[str, int],
) -> float:
    """
    Estimate memory requirements based on variables.

    Returns:
        Memory estimate in KB.
    """
    total_memory = 0.0

    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "float")
        dims = var_info.get("dimensions", [1])

        size_factor = memory_factors.get(var_type, memory_factors["float"])

        try:
            dimension_values = _parse_dimensions(dims)
            total_size = size_factor * math.prod(dimension_values)
        except Exception as e:
            logger.warning(f"Error calculating size for variable {var_name}: {e}")
            total_size = size_factor * 2

        total_memory += total_size

    return total_memory / 1024.0


def detailed_memory_breakdown(
    variables: Dict[str, Any],
    memory_factors: Dict[str, int],
) -> Dict[str, Any]:
    """Create detailed memory breakdown by variable and type."""
    breakdown: Dict[str, Any] = {
        "by_variable": {},
        "by_type": {t: 0 for t in memory_factors},
        "total_bytes": 0,
        "representation_overhead": 1024,
    }

    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "float")
        dims = var_info.get("dimensions", [1])
        size_factor = memory_factors.get(var_type, memory_factors["float"])

        try:
            dimension_values = _parse_dimensions(dims)
            element_count = math.prod(dimension_values)
            var_size = size_factor * element_count
            var_overhead = len(var_name) + 24

            breakdown["by_variable"][var_name] = {
                "size_bytes": var_size,
                "elements": element_count,
                "dimensions": dimension_values,
                "type": var_type,
                "overhead_bytes": var_overhead,
                "total_bytes": var_size + var_overhead,
            }

            breakdown["by_type"][var_type] = breakdown["by_type"].get(var_type, 0) + var_size
            breakdown["total_bytes"] += var_size + var_overhead
            breakdown["representation_overhead"] += var_overhead
        except Exception as e:
            logger.warning(f"Error in memory breakdown for {var_name}: {e}")

    breakdown["total_kb"] = breakdown["total_bytes"] / 1024.0
    breakdown["overhead_kb"] = breakdown["representation_overhead"] / 1024.0
    return breakdown


def estimate_flops(
    variables: Dict[str, Any],
    edges: List[Dict[str, Any]],
    equations: str,
    model_type: str,
    operation_costs: Dict[str, int],
) -> Dict[str, Any]:
    """Estimate floating-point operations (FLOPS) required for inference."""
    flops_estimate: Dict[str, Any] = {
        "total_flops": 0,
        "matrix_operations": 0,
        "element_operations": 0,
        "nonlinear_operations": 0,
    }

    matrices: Dict[str, list] = {}
    for var_name, var_info in variables.items():
        dims = var_info.get("dimensions", [1])
        if len(dims) >= 2:
            matrices[var_name] = dims

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")

        if source in matrices and target in matrices:
            source_dims = matrices[source]
            target_dims = matrices[target]

            if len(source_dims) >= 2 and len(target_dims) >= 2:
                try:
                    m = _safe_dim_int(source_dims[0])
                    n = _safe_dim_int(source_dims[1])
                    p = _safe_dim_int(target_dims[1]) if len(target_dims) > 1 else 2
                    flops = m * n * p * operation_costs["matrix_multiply"]
                    flops_estimate["matrix_operations"] += flops
                    flops_estimate["total_flops"] += flops
                except (ValueError, TypeError):
                    flops_estimate["matrix_operations"] += 100
                    flops_estimate["total_flops"] += 100

    for line in equations.split("\n"):
        element_ops = 0
        nonlinear_ops = 0

        element_ops += line.count("+") * operation_costs["addition"]
        element_ops += line.count("*") * operation_costs["scalar_multiply"]
        element_ops += line.count("/") * operation_costs["division"]

        nonlinear_ops += line.count("exp") * operation_costs["exp"]
        nonlinear_ops += line.count("log") * operation_costs["log"]
        nonlinear_ops += line.count("softmax") * operation_costs["softmax"]
        nonlinear_ops += line.count("sigma") * operation_costs["sigmoid"]

        flops_estimate["element_operations"] += element_ops
        flops_estimate["nonlinear_operations"] += nonlinear_ops
        flops_estimate["total_flops"] += element_ops + nonlinear_ops

    if model_type == "Dynamic":
        flops_estimate["total_flops"] *= 2.5
    elif model_type == "Hierarchical":
        flops_estimate["total_flops"] *= 3.5

    if flops_estimate["total_flops"] == 0:
        base_flops = len(variables) * 20
        multiplier = {"Static": 1.0, "Dynamic": 2.5}.get(model_type, 3.5)
        flops_estimate["total_flops"] = base_flops * multiplier

    return flops_estimate


def estimate_inference_time(
    flops_estimate: Dict[str, Any],
    hardware_specs: Dict[str, float],
) -> Dict[str, float]:
    """Estimate inference time based on FLOPS and hardware specs."""
    total_flops = flops_estimate["total_flops"]
    cpu_time_seconds = total_flops / hardware_specs["cpu_flops_per_second"]
    return {
        "cpu_time_seconds": cpu_time_seconds,
        "cpu_time_ms": cpu_time_seconds * 1000,
        "cpu_time_us": cpu_time_seconds * 1_000_000,
    }


def estimate_batched_inference(
    variables: Dict[str, Any],
    model_type: str,
    flops_estimate: Dict[str, Any],
    hardware_specs: Dict[str, float],
) -> Dict[str, Any]:
    """Estimate batched inference performance."""
    total_flops = flops_estimate["total_flops"]
    batch_sizes = [1, 8, 32, 128, 512]
    batch_estimates: Dict[str, Any] = {}

    for batch_size in batch_sizes:
        if batch_size == 1:
            batch_flops = total_flops
        else:
            scale_factor = 0.7 + 0.3 / math.log2(batch_size + 1)
            batch_flops = total_flops * batch_size * scale_factor

        time_seconds = batch_flops / hardware_specs["cpu_flops_per_second"]
        throughput = batch_size / time_seconds if time_seconds > 0 else 0

        batch_estimates[f"batch_{batch_size}"] = {
            "flops": batch_flops,
            "time_seconds": time_seconds,
            "throughput_per_second": throughput,
        }

    return batch_estimates


def estimate_matrix_operation_costs(
    variables: Dict[str, Any],
    edges: List[Dict[str, Any]],
    equations: str,
    operation_costs: Dict[str, int],
) -> Dict[str, Any]:
    """Provide detailed estimates of matrix operation costs."""
    costs: Dict[str, Any] = {
        "matrix_multiply": [],
        "matrix_transpose": [],
        "matrix_inversion": [],
        "element_wise": [],
        "total_matrix_flops": 0,
    }

    matrices: Dict[str, list] = {}
    for var_name, var_info in variables.items():
        dims = var_info.get("dimensions", [1])
        if len(dims) >= 2:
            int_dims = [_safe_dim_int(d) for d in dims]
            matrices[var_name] = int_dims

    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")

        if source in matrices and target in matrices:
            source_dims = matrices[source]
            target_dims = matrices[target]

            if len(source_dims) >= 2 and len(target_dims) >= 2:
                m = source_dims[0]
                n = source_dims[1] if len(source_dims) > 1 else 1
                p = target_dims[1] if len(target_dims) > 1 else 1
                flops = m * n * p * operation_costs["matrix_multiply"]

                costs["matrix_multiply"].append({
                    "operation": f"{source} \u00d7 {target}",
                    "dimensions": f"{m}\u00d7{n} * {n}\u00d7{p}",
                    "flops": flops,
                })
                costs["total_matrix_flops"] += flops

    for line in equations.split("\n"):
        if "^T" in line or "transpose" in line:
            for matrix_name, dims in matrices.items():
                if matrix_name in line and (f"{matrix_name}^T" in line or f"transpose({matrix_name})" in line):
                    m, n = dims[0], dims[1] if len(dims) > 1 else 1
                    flops = m * n
                    costs["matrix_transpose"].append({
                        "operation": f"{matrix_name}^T",
                        "dimensions": f"{m}\u00d7{n}",
                        "flops": flops,
                    })
                    costs["total_matrix_flops"] += flops

        if "^-1" in line or "inv(" in line:
            for matrix_name, dims in matrices.items():
                if matrix_name in line and (f"{matrix_name}^-1" in line or f"inv({matrix_name})" in line):
                    n = dims[0]
                    flops = n ** 3
                    costs["matrix_inversion"].append({
                        "operation": f"{matrix_name}^-1",
                        "dimensions": f"{n}\u00d7{n}",
                        "flops": flops,
                    })
                    costs["total_matrix_flops"] += flops

    return costs


def estimate_model_overhead(
    variables: Dict[str, Any],
    edges: List[Dict[str, Any]],
    equations: str,
) -> Dict[str, Any]:
    """Estimate model overhead including compile-time and optimization costs."""
    var_count = len(variables)
    edge_count = len(edges)
    eq_count = len(equations.split("\n"))

    compilation_ms = 10 + (var_count * 2) + (edge_count * 1) + (eq_count * 5)
    optimization_ms = 20 + (var_count ** 2 * 0.5)
    memory_overhead_bytes = 1024 + (var_count * 50) + (edge_count * 30) + (eq_count * 100)

    return {
        "compilation_ms": compilation_ms,
        "optimization_ms": optimization_ms,
        "memory_overhead_kb": memory_overhead_bytes / 1024.0,
    }


def estimate_inference(
    variables: Dict[str, Any],
    model_type: str,
    edges: List[Dict[str, Any]],
    equations: str,
    inference_factors: Dict[str, float],
) -> float:
    """Estimate inference time requirements based on model complexity."""
    base_time = inference_factors.get(model_type, inference_factors["Static"])

    var_time = 0.0
    for var_name, var_info in variables.items():
        var_type = var_info.get("type", "float")
        dims = var_info.get("dimensions", [1])

        try:
            element_count = 1
            for d in dims:
                element_count *= _parse_single_dim(d)
            type_factor = inference_factors.get(var_type, inference_factors["float"])
            var_time += type_factor * math.log2(element_count + 1)
        except Exception:
            var_time += inference_factors.get(var_type, 1.0)

    edge_time = 0.0
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        if "+" in source or "+" in target:
            edge_time += 1.0
        else:
            edge_time += 0.5

    equation_lines = equations.split("\n")
    equation_time = 0.0
    for line in equation_lines:
        eq_cost = 2.0
        if "softmax" in line or "sigma" in line:
            eq_cost += 1.5
        if "^" in line:
            eq_cost += 1.0
        if "sum" in line or "\u2211" in line:
            eq_cost += 1.0
        equation_time += eq_cost

    if equation_time == 0 and len(equation_lines) > 0:
        equation_time = len(equation_lines) * 2.0

    weighted_time = (
        base_time * 4.0 + var_time * 2.5 + edge_time * 1.0 + equation_time * 2.5
    ) / 10.0

    return weighted_time * 10.0


def estimate_storage(
    variables: Dict[str, Any],
    edges: List[Dict[str, Any]],
    equations: str,
    memory_factors: Dict[str, int],
) -> float:
    """Estimate storage requirements in KB."""
    memory_est = estimate_memory(variables, memory_factors)

    structural_overhead_kb = 1.0
    var_overhead_kb = 0.0
    for var_name, var_info in variables.items():
        var_desc_length = len(var_info.get("comment", ""))
        var_overhead_kb += (len(var_name) + 24 + var_desc_length) / 1024.0

    edge_overhead_kb = len(edges) * 0.1
    equation_overhead_kb = len(equations) * 0.001
    format_overhead_kb = 0.5

    total_storage_kb = (
        memory_est * 1.2
        + structural_overhead_kb
        + var_overhead_kb
        + edge_overhead_kb
        + equation_overhead_kb
        + format_overhead_kb
    )

    return max(total_storage_kb, 1.0)


def calculate_complexity(
    variables: Dict[str, Any],
    edges: List[Dict[str, Any]],
    equations: str,
) -> Dict[str, float]:
    """Calculate detailed complexity metrics for the model."""
    total_dims = 0
    max_dim = 0
    for var_info in variables.values():
        dims = var_info.get("dimensions", [1])
        int_dims = [_safe_dim_int(d) for d in dims]
        dim_size = math.prod(int_dims)
        total_dims += dim_size
        max_dim = max(max_dim, dim_size)

    var_count = len(variables)
    edge_count = len(edges)

    density = 0.0
    if var_count > 1:
        max_possible_edges = var_count * (var_count - 1)
        density = edge_count / max_possible_edges if max_possible_edges > 0 else 0

    in_degree: Dict[str, int] = {}
    out_degree: Dict[str, int] = {}
    for edge in edges:
        source = edge.get("source", "").split("+")[0]
        target = edge.get("target", "").split("+")[0]
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1

    avg_in_degree = sum(in_degree.values()) / max(len(in_degree), 1)
    avg_out_degree = sum(out_degree.values()) / max(len(out_degree), 1)
    max_in_degree = max(in_degree.values()) if in_degree else 0
    max_out_degree = max(out_degree.values()) if out_degree else 0

    cyclic_score = 0.0
    if edge_count > var_count:
        cyclic_score = (edge_count - var_count) / max(var_count, 1)

    temporal_edges = 0
    for edge in edges:
        if "+" in edge.get("source", "") or "+" in edge.get("target", ""):
            temporal_edges += 1
    temporal_complexity = temporal_edges / max(edge_count, 1)

    eq_lines = equations.split("\n")
    avg_eq_length = sum(len(line) for line in eq_lines) / max(len(eq_lines), 1)

    operators = 0
    for line in eq_lines:
        operators += line.count("+") + line.count("-") + line.count("*") + line.count("/") + line.count("^")

    higher_order_ops = 0
    for line in eq_lines:
        higher_order_ops += line.count("sum") + line.count("prod") + line.count("log") + line.count("exp")
        higher_order_ops += line.count("softmax") + line.count("tanh") + line.count("sigma")

    equation_complexity = 0.0
    if eq_lines:
        equation_complexity = (avg_eq_length + operators + 3 * higher_order_ops) / len(eq_lines)

    state_space_complexity = math.log2(total_dims + 1) if total_dims > 0 else 0

    overall_complexity = (
        state_space_complexity * 0.25
        + (density + cyclic_score) * 0.25
        + temporal_complexity * 0.2
        + equation_complexity * 0.3
    )
    overall_complexity = min(10, overall_complexity * 2)

    return {
        "state_space_complexity": state_space_complexity,
        "graph_density": density,
        "avg_in_degree": avg_in_degree,
        "avg_out_degree": avg_out_degree,
        "max_in_degree": max_in_degree,
        "max_out_degree": max_out_degree,
        "cyclic_complexity": cyclic_score,
        "temporal_complexity": temporal_complexity,
        "equation_complexity": equation_complexity,
        "overall_complexity": overall_complexity,
        "variable_count": var_count,
        "edge_count": edge_count,
        "total_state_space_dim": total_dims,
        "max_variable_dim": max_dim,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_single_dim(d: Any) -> int:
    """Parse a single dimension value to int."""
    if isinstance(d, (int, float)):
        return int(d)
    if isinstance(d, str):
        if d.isdigit():
            return int(d)
        if "len" in d and "\u03c0" in d:
            return 3
        if d.startswith("="):
            matches = re.findall(r"\d+", d)
            return int(matches[0]) if matches else 1
    return 1


def _parse_dimensions(dims: list) -> list:
    """Parse a list of dimension values to ints."""
    return [_parse_single_dim(d) for d in dims]


def _safe_dim_int(d: Any) -> int:
    """Convert a dimension to int, defaulting to 2 for unparseable values."""
    if isinstance(d, int):
        return d
    if isinstance(d, str) and d.isdigit():
        return int(d)
    return 2
