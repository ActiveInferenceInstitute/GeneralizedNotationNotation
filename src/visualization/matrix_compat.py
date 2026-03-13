"""
Matrix helper functions for visualization.

These functions operate on visualization data and live in visualization/
where they belong. analysis/ should not be imported from visualization/.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import re

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    _NP_AVAILABLE = False


def parse_matrix_data(matrix_str: str) -> Optional[Any]:
    """Parse a matrix from its string representation into a numpy array."""
    try:
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', matrix_str)
        if numbers and _NP_AVAILABLE:
            return np.array([float(n) for n in numbers])
        return None
    except Exception:
        return None


def generate_matrix_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """Generate matrix visualizations; delegates to matrix_visualizer when available."""
    try:
        from .matrix_visualizer import MatrixVisualizer
        mv = MatrixVisualizer()
        return mv.generate_visualizations(parsed_data, output_dir, model_name)
    except Exception:
        return []
