#!/usr/bin/env python3
"""
Statistical helpers (per-element statistics, distributions, correlations) for GNN Step 16 analysis.

Extracted from ``analysis.analyzer``.
"""

import logging
from typing import (
    Any,
    Dict,
    List,
    cast,
)

import numpy as np

from .analysis_complexity import calculate_cyclomatic_complexity

logger = logging.getLogger(__name__)


# Import visualization libraries with error handling
try:
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = cast(Any, None)


def calculate_variable_statistics(variables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for variables."""
    if not variables:
        return {"count": 0, "types": {}, "average_line": 0}

    stats: dict[str, Any] = {
        "count": len(variables),
        "types": {},
        "average_line": np.mean([var.get("line", 0) for var in variables]),
        "line_std": np.std([var.get("line", 0) for var in variables]),
    }

    # Count types
    for var in variables:
        var_type = var.get("type", "unknown")
        stats["types"][var_type] = stats["types"].get(var_type, 0) + 1

    return stats


def calculate_connection_statistics(
    connections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate statistics for connections."""
    if not connections:
        return {"count": 0, "average_line": 0}

    stats: dict[str, Any] = {
        "count": len(connections),
        "average_line": np.mean([conn.get("line", 0) for conn in connections]),
        "line_std": np.std([conn.get("line", 0) for conn in connections]),
    }

    return stats


def calculate_section_statistics(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for sections."""
    if not sections:
        return {"count": 0, "average_line": 0}

    stats: dict[str, Any] = {
        "count": len(sections),
        "average_line": np.mean([section.get("line", 0) for section in sections]),
        "line_std": np.std([section.get("line", 0) for section in sections]),
    }

    return stats


def analyze_distributions(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze distributions of model elements."""
    analysis: dict[str, Any] = {
        "variable_distribution": {},
        "connection_distribution": {},
        "complexity_metrics": {},
    }

    # Analyze variable distribution
    if variables:
        var_lines = [var.get("line", 0) for var in variables]
        analysis["variable_distribution"] = {
            "mean": float(np.mean(var_lines)),
            "std": float(np.std(var_lines)),
            "min": float(np.min(var_lines)),
            "max": float(np.max(var_lines)),
            "median": float(np.median(var_lines)),
        }
        if SCIPY_AVAILABLE and len(var_lines) > 2:
            analysis["variable_distribution"]["skewness"] = float(stats.skew(var_lines))
            analysis["variable_distribution"]["kurtosis"] = float(
                stats.kurtosis(var_lines)
            )
            counts = np.unique(var_lines, return_counts=True)[1]
            analysis["variable_distribution"]["entropy"] = float(stats.entropy(counts))

    # Analyze connection distribution
    if connections:
        conn_lines = [conn.get("line", 0) for conn in connections]
        analysis["connection_distribution"] = {
            "mean": float(np.mean(conn_lines)),
            "std": float(np.std(conn_lines)),
            "min": float(np.min(conn_lines)),
            "max": float(np.max(conn_lines)),
            "median": float(np.median(conn_lines)),
        }
        if SCIPY_AVAILABLE and len(conn_lines) > 2:
            analysis["connection_distribution"]["skewness"] = float(
                stats.skew(conn_lines)
            )
            analysis["connection_distribution"]["kurtosis"] = float(
                stats.kurtosis(conn_lines)
            )
            counts = np.unique(conn_lines, return_counts=True)[1]
            analysis["connection_distribution"]["entropy"] = float(
                stats.entropy(counts)
            )

    # Calculate complexity metrics
    analysis["complexity_metrics"] = {
        "total_elements": len(variables) + len(connections),
        "variable_complexity": len(variables),
        "connection_complexity": len(connections),
        "density": len(connections) / max(len(variables), 1),
        "cyclomatic_complexity": calculate_cyclomatic_complexity(
            variables, connections
        ),
    }

    return analysis


def calculate_correlations(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate correlations between model elements."""
    correlations: dict[str, Any] = {
        "variable_connection_correlation": 0.0,
        "line_position_correlation": 0.0,
    }

    if variables and connections:
        # Calculate correlation between number of variables and connections
        var_count = len(variables)
        conn_count = len(connections)
        correlations["variable_connection_correlation"] = conn_count / max(var_count, 1)

        # Calculate line position correlation
        var_lines = [var.get("line", 0) for var in variables]
        conn_lines = [conn.get("line", 0) for conn in connections]

        if len(var_lines) > 1 and len(conn_lines) > 1:
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    correlation_matrix = np.corrcoef(var_lines, conn_lines)
                val = correlation_matrix[0, 1]
                correlations["line_position_correlation"] = (
                    0.0 if np.isnan(val) else float(val)
                )
            except Exception as e:
                logger.debug(f"Correlation computation failed, defaulting to 0.0: {e}")
                correlations["line_position_correlation"] = 0.0

    return correlations


# Explicit re-export surface (no_implicit_reexport).
__all__ = [
    "SCIPY_AVAILABLE",
    "analyze_distributions",
    "calculate_connection_statistics",
    "calculate_correlations",
    "calculate_section_statistics",
    "calculate_variable_statistics",
    "stats",
]
