#!/usr/bin/env python3
"""
PyTorch Analyzer for GNN Pipeline

Reads simulation_results.json produced by PyTorch runner and generates
belief trajectory, action distribution, and EFE analysis plots.

The analysis pipeline is shared with the NumPyro analyzer via
``analysis.flat_payload_analyzer``; this module binds it to the PyTorch
spec (framework name, file patterns, plot labels) and re-exports the
public ``generate_analysis_from_logs`` / ``_generate_plots`` entry points.

@Web: https://pytorch.org/docs/stable/
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..flat_payload_analyzer import (
    FlatPayloadSpec,
)
from ..flat_payload_analyzer import (
    generate_analysis_from_logs as _generate_from_spec,
)

# Public spec: PyTorch-prefixed result files, PyTorch bar color, PyTorch labels.
PYTORCH_SPEC = FlatPayloadSpec(
    framework="pytorch",
    file_patterns=(
        "**/pytorch/**/simulation_results.json",
        "**/pytorch_simulation_results.json",
    ),
    analysis_filename="pytorch_analysis.json",
    title_prefix="PyTorch",
    bar_color="#4285F4",
    log_label="PyTorch",
)


def generate_analysis_from_logs(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> List[str]:
    """Generate analysis from PyTorch simulation results.

    Searches recursively for simulation_results.json files under results_dir
    (including model/pytorch/simulation_data subdirectories).

    Args:
        results_dir: Root directory to search for results (e.g. 12_execute_output).
        output_dir: Directory for analysis artifacts. Defaults to results_dir.
        verbose: Enable verbose logging.

    Returns:
        List of generated output file paths.
    """
    return _generate_from_spec(PYTORCH_SPEC, results_dir, output_dir, verbose)


def _generate_plots(
    beliefs: np.ndarray,
    actions: list,
    observations: list,
    efe: np.ndarray,
    output_dir: Path,
) -> bool:
    """Generate analysis plots using matplotlib.

    Returns True if at least one plot artifact was written, False otherwise
    (e.g. matplotlib unavailable or no plottable data).
    """
    from ..flat_payload_analyzer import _generate_plots as _shared_plots

    return _shared_plots(PYTORCH_SPEC, beliefs, actions, observations, efe, output_dir)


__all__ = [
    "PYTORCH_SPEC",
    "generate_analysis_from_logs",
]
