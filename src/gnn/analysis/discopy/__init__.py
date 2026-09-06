"""
DisCoPy Analysis Module

Per-framework analysis and visualization for DisCoPy categorical diagrams.
"""

from typing import Any

from .analyzer import (
    analyze_diagram_structure,
    create_discopy_visualizations,
    extract_circuit_data,
    generate_analysis_from_logs,
)

__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_discopy_visualizations",
    "extract_circuit_data",
    "analyze_diagram_structure",
]
