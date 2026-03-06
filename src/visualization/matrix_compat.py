"""
Compatibility shim for analysis.analyzer matrix functions.

Provides stub implementations when the analysis module is unavailable, so that
visualization/processor.py does not need to know about analysis/ internals directly.
"""
try:
    from analysis.analyzer import parse_matrix_data, generate_matrix_visualizations
except ImportError:
    from typing import Any, Dict, List
    from pathlib import Path
    import re
    import numpy as np

    def parse_matrix_data(matrix_str: str) -> Any:
        try:
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', matrix_str)
            if len(numbers) >= 1:
                return np.array([float(n) for n in numbers])
            return None
        except Exception:
            return None

    def generate_matrix_visualizations(
        parsed_data: Dict[str, Any], output_dir: Path, model_name: str
    ) -> List[str]:
        return []
