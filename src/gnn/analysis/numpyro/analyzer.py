#!/usr/bin/env python3
"""
NumPyro Analyzer for GNN Pipeline

Reads simulation_results.json produced by NumPyro runner and generates
belief trajectory, action distribution, and EFE analysis plots.

The analysis pipeline is shared with the PyTorch analyzer via
``analysis.flat_payload_analyzer``; this module binds it to the NumPyro
spec (framework name, file patterns, plot labels) and re-exports the
public ``generate_analysis_from_logs`` / ``_generate_plots`` entry points.

@Web: https://num.pyro.ai/
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

# Public spec: NumPyro-prefixed result files, NumPyro bar color, NumPyro labels.
NUMPYRO_SPEC = FlatPayloadSpec(
    framework="numpyro",
    file_patterns=(
        "**/numpyro/**/simulation_results.json",
        "**/numpyro_simulation_results.json",
    ),
    analysis_filename="numpyro_analysis.json",
    title_prefix="NumPyro",
    bar_color="#EA4335",
    log_label="NumPyro",
)


def generate_analysis_from_logs(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> List[str]:
    """Generate analysis from NumPyro simulation results.

    Searches recursively for simulation_results.json files under results_dir
    (including model/numpyro/simulation_data subdirectories).

    Args:
        results_dir: Root directory to search for results (e.g. 12_execute_output).
        output_dir: Directory for analysis artifacts. Defaults to results_dir.
        verbose: Enable verbose logging.

    Returns:
        List of generated output file paths.
    """
    return _generate_from_spec(NUMPYRO_SPEC, results_dir, output_dir, verbose)


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

    return _shared_plots(NUMPYRO_SPEC, beliefs, actions, observations, efe, output_dir)


__all__ = [
    "NUMPYRO_SPEC",
    "generate_analysis_from_logs",
]
