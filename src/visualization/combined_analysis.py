"""Facade: combined analysis lives in visualization.analysis."""

from typing import Any

from .analysis import (
    generate_combined_analysis,
    generate_combined_visualizations,
)

__all__: list[Any] = ["generate_combined_analysis", "generate_combined_visualizations"]
