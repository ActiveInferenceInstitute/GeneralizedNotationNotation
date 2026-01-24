"""
DisCoPy Analysis Module

Per-framework analysis and visualization for DisCoPy categorical diagrams.
"""

from .analyzer import (
    generate_analysis_from_logs,
    create_discopy_visualizations,
    extract_circuit_data,
    analyze_diagram_structure,
)

__all__ = [
    "generate_analysis_from_logs",
    "create_discopy_visualizations",
    "extract_circuit_data",
    "analyze_diagram_structure",
]
