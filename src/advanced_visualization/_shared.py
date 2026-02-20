"""
Shared utilities for advanced visualization sub-modules.

Contains dataclasses, validation, and helper functions used by both
processor.py, network_viz.py, and statistical_viz.py. Exists to
avoid circular imports between processor and sub-modules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


@dataclass
class AdvancedVisualizationAttempt:
    """Track individual visualization attempts"""
    viz_type: str
    model_name: str
    status: str  # "success", "failed", "skipped"
    duration_ms: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass
class AdvancedVisualizationResults:
    """Aggregate results for advanced visualization processing"""
    total_attempts: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    attempts: List[AdvancedVisualizationAttempt] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _normalize_connection_format(conn_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize connection format to handle both old and new formats."""
    if "source_variables" in conn_info and "target_variables" in conn_info:
        return conn_info
    elif "source" in conn_info and "target" in conn_info:
        return {
            "source_variables": [conn_info["source"]],
            "target_variables": [conn_info["target"]],
            **{k: v for k, v in conn_info.items() if k not in ["source", "target"]}
        }
    else:
        return conn_info


def _calculate_semantic_positions(variables: List[Dict], connections: List[Dict]):
    """
    Calculate meaningful 3D positions for variables based on semantic relationships.

    Args:
        variables: List of variable dictionaries
        connections: List of connection dictionaries

    Returns:
        Array of 3D positions for each variable
    """
    if not NUMPY_AVAILABLE or np is None:
        return []

    if not variables:
        return np.array([])

    n_vars = len(variables)
    positions = np.zeros((n_vars, 3))

    np.random.seed(42)
    positions = np.random.rand(n_vars, 3) * 10

    var_names = [var.get("name", f"var_{i}") for i, var in enumerate(variables)]
    connection_matrix = np.zeros((n_vars, n_vars))

    for conn_info in connections:
        normalized_conn = _normalize_connection_format(conn_info)
        source_vars = normalized_conn.get("source_variables", [])
        target_vars = normalized_conn.get("target_variables", [])

        for source_var in source_vars:
            for target_var in target_vars:
                if source_var != target_var:
                    source_idx = None
                    target_idx = None

                    for idx, name in enumerate(var_names):
                        if name == source_var:
                            source_idx = idx
                        if name == target_var:
                            target_idx = idx

                    if source_idx is not None and target_idx is not None:
                        connection_matrix[source_idx, target_idx] = 1

    for _ in range(50):
        forces = np.zeros_like(positions)

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += (diff / distance) * (1 / distance)

        for i in range(n_vars):
            for j in range(n_vars):
                if connection_matrix[i, j] > 0:
                    diff = positions[j] - positions[i]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += diff * (distance / 10)

        positions += forces * 0.01

    positions = (positions - positions.min()) / (positions.max() - positions.min()) * 10

    return positions


def _generate_fallback_report(
    model_name: str,
    viz_type: str,
    output_dir: Path,
    model_data: Dict,
    logger: logging.Logger
):
    """Generate fallback HTML report when advanced libraries unavailable"""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{model_name} - {viz_type.upper()} Visualization (Fallback)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        .data {{ background: #fff; border: 1px solid #ddd; padding: 10px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{model_name} - {viz_type.upper()} Visualization</h1>
    <div class="info">
        <p><strong>Note:</strong> Advanced visualization libraries not available.
        Showing basic model information instead.</p>
    </div>
    <div class="data">
        <h2>Model Structure</h2>
        <pre>{json.dumps(model_data, indent=2)}</pre>
    </div>
</body>
</html>"""

    output_file = output_dir / f"{model_name}_{viz_type}_fallback.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    logger.info(f"Generated fallback report: {output_file}")


def validate_visualization_data(model_data: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """
    Validate that visualization data is complete and meaningful.

    Args:
        model_data: Parsed model data
        logger: Logger instance

    Returns:
        Validation results dictionary
    """
    validation_results = {
        "overall_valid": True,
        "warnings": [],
        "errors": [],
        "data_quality": {},
        "recommendations": []
    }

    try:
        if not isinstance(model_data, dict):
            validation_results["errors"].append("Model data is not a dictionary")
            validation_results["overall_valid"] = False
            return validation_results

        required_keys = ["variables", "connections"]
        for key in required_keys:
            if key not in model_data:
                validation_results["warnings"].append(f"Missing key: {key}")
            elif not model_data[key]:
                validation_results["warnings"].append(f"Empty data for key: {key}")

        variables = model_data.get("variables", [])
        if not variables:
            validation_results["errors"].append("No variables found in model")
            validation_results["overall_valid"] = False
        else:
            validation_results["data_quality"]["total_variables"] = len(variables)

            valid_vars = 0
            for var in variables:
                if isinstance(var, dict) and "name" in var and "var_type" in var:
                    valid_vars += 1
                else:
                    validation_results["warnings"].append(f"Invalid variable structure: {var}")

            validation_results["data_quality"]["valid_variables"] = valid_vars
            validation_results["data_quality"]["variable_validity_rate"] = valid_vars / len(variables)

            if valid_vars < len(variables) * 0.8:
                validation_results["warnings"].append("Low variable validity rate")

        connections = model_data.get("connections", [])
        if not connections:
            validation_results["warnings"].append("No connections found in model")
        else:
            validation_results["data_quality"]["total_connections"] = len(connections)

            valid_connections = 0
            for conn in connections:
                if isinstance(conn, dict) and ("source_variables" in conn or "target_variables" in conn):
                    valid_connections += 1
                else:
                    validation_results["warnings"].append(f"Invalid connection structure: {conn}")

            validation_results["data_quality"]["valid_connections"] = valid_connections
            validation_results["data_quality"]["connection_validity_rate"] = valid_connections / len(connections)

        pomdp_indicators = {
            "likelihood_matrix": 0,
            "transition_matrix": 0,
            "preference_vector": 0,
            "prior_vector": 0,
            "hidden_state": 0,
            "observation": 0,
            "policy": 0
        }

        for var in variables:
            if isinstance(var, dict):
                var_type = var.get("var_type", "")
                for indicator in pomdp_indicators:
                    if indicator in var_type:
                        pomdp_indicators[indicator] += 1

        validation_results["data_quality"]["pomdp_indicators"] = pomdp_indicators

        pomdp_score = sum(pomdp_indicators.values())
        if pomdp_score >= 3:
            validation_results["data_quality"]["is_pomdp_model"] = True
            validation_results["data_quality"]["pomdp_completeness"] = pomdp_score / len(pomdp_indicators)
        else:
            validation_results["data_quality"]["is_pomdp_model"] = False
            validation_results["warnings"].append("Model does not appear to be a complete POMDP")

        if validation_results["data_quality"].get("variable_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append("Review variable parsing - high invalidity rate")

        if validation_results["data_quality"].get("connection_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append("Review connection parsing - high invalidity rate")

        if pomdp_score < 3:
            validation_results["recommendations"].append("Model may not be a complete POMDP - check GNN structure")

        if validation_results["errors"]:
            validation_results["overall_valid"] = False
        elif len(validation_results["warnings"]) > 2:
            validation_results["overall_valid"] = False
            validation_results["warnings"].append("Too many warnings - data quality may be poor")

        if logger:
            logger.info(f"Validation completed: {validation_results['overall_valid']} (errors: {len(validation_results['errors'])}, warnings: {len(validation_results['warnings'])})")

    except Exception as e:
        validation_results["errors"].append(f"Validation error: {e}")
        validation_results["overall_valid"] = False
        if logger:
            logger.error(f"Validation failed: {e}")

    return validation_results
