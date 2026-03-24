"""Matrix string parsing and batch matrix viz entrypoint (keeps analysis/ free of visualization)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment,misc]
    _NP_AVAILABLE = False


def parse_matrix_data(matrix_str: str) -> Optional[Any]:
    """Parse a matrix from its string representation into a numpy array."""
    try:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", matrix_str)
        if numbers and _NP_AVAILABLE and np is not None:
            return np.array([float(n) for n in numbers])
        return None
    except Exception:
        return None


def generate_matrix_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """Run matrix analysis + stats plots from parsed GNN data."""
    try:
        from visualization.matrix.visualizer import generate_matrix_visualizations as _run

        return _run(parsed_data, output_dir, model_name)
    except Exception:
        return []
