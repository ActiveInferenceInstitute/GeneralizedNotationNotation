#!/usr/bin/env python3
"""
Code metrics counting for generated files.

Counts lines of code and simple structural elements (functions, classes) in a
source file. Language-aware only to the extent the pipeline needs: Python
(``def``/``class``), Julia (``function``), and JAX-decorated functions.
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def count_code_metrics(file_path: Path) -> Dict[str, int]:
    """
    Calculate code metrics for a generated file.

    Args:
        file_path: Path to the code file

    Returns:
        Dictionary with lines_of_code, functions, classes counts
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Count non-empty, non-comment lines
        loc = sum(
            1 for line in lines if line.strip() and not line.strip().startswith("#")
        )

        # Count functions (Python: def, Julia: function)
        functions = sum(
            1
            for line in lines
            if line.strip().startswith("def ")
            or line.strip().startswith("function ")
            or "@jit" in line
        )  # JAX decorated functions

        # Count classes (Python: class)
        classes = sum(1 for line in lines if line.strip().startswith("class "))

        return {
            "lines_of_code": loc,
            "total_lines": len(lines),
            "functions": functions,
            "classes": classes,
        }
    except Exception as e:
        logger.warning(f"Could not count code metrics for {file_path}: {e}")
        return {"lines_of_code": 0, "total_lines": 0, "functions": 0, "classes": 0}
