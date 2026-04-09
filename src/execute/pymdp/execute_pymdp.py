#!/usr/bin/env python3
"""
Thin pipeline entry point for PyMDP simulations (pymdp 1.0.0 / JAX).

This module used to contain a second, shadowed ``execute_pymdp_simulation``
definition with a different signature from the one exported by
``src/execute/pymdp/__init__.py`` (via ``executor.py``). That duplicate
led to subtle bugs whenever a caller imported this file directly.

It is now a small shim that re-exports the canonical entry points and
provides two convenience helpers that operate on GNN files and batches.
The actual simulation work is done by ``simple_simulation`` + ``pymdp_simulation``,
both of which call real pymdp 1.0.0.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

from .executor import (
    execute_pymdp_simulation,
    execute_pymdp_simulation_from_gnn,
)
from .pymdp_simulation import PyMDPSimulation  # noqa: F401 - re-exported
from .pymdp_utils import extract_gnn_dimensions, validate_gnn_pomdp_structure

logger = logging.getLogger(__name__)


def execute_from_gnn_file(
    gnn_file: Path,
    output_dir: Path,
    correlation_id: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """Parse a GNN file and hand it to the canonical executor."""
    try:
        logger.info("Parsing GNN file: %s", gnn_file)
        try:
            from ...gnn import parse_gnn_file  # type: ignore[attr-defined]
        except ImportError:
            from src.gnn import parse_gnn_file  # type: ignore[no-redef]

        gnn_spec = parse_gnn_file(gnn_file)
        if not gnn_spec:
            msg = f"Failed to parse GNN file: {gnn_file}"
            logger.error(msg)
            return False, {"error": msg}

        if hasattr(gnn_spec, "to_dict"):
            gnn_spec_dict = gnn_spec.to_dict()
        else:
            gnn_spec_dict = gnn_spec

        return execute_pymdp_simulation(
            gnn_spec=gnn_spec_dict,
            output_dir=output_dir,
            correlation_id=correlation_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to execute from GNN file %s: %s", gnn_file, e)
        return False, {
            "error": str(e),
            "exception": str(e),
            "traceback": traceback.format_exc(),
        }


def batch_execute_pymdp(
    gnn_specs: list,
    base_output_dir: Path,
) -> Dict[str, Any]:
    """Execute several GNN specifications via the canonical executor."""
    logger.info("Starting batch execution of %d PyMDP simulations", len(gnn_specs))

    batch_results: Dict[str, Any] = {
        "total_simulations": len(gnn_specs),
        "successful_simulations": 0,
        "failed_simulations": 0,
        "results": [],
        "errors": [],
    }

    for i, gnn_spec in enumerate(gnn_specs):
        model_name = gnn_spec.get("model_name", f"simulation_{i + 1}")
        sim_output_dir = base_output_dir / model_name
        try:
            success, results = execute_pymdp_simulation(
                gnn_spec=gnn_spec,
                output_dir=sim_output_dir,
                correlation_id=f"batch_{i}",
            )
            batch_results["results"].append(
                {
                    "simulation_index": i,
                    "model_name": model_name,
                    "success": success,
                    "results": results,
                }
            )
            if success:
                batch_results["successful_simulations"] += 1
            else:
                batch_results["failed_simulations"] += 1
                batch_results["errors"].append(
                    {
                        "simulation_index": i,
                        "model_name": model_name,
                        "error": results.get("error"),
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.error("Exception in simulation %d: %s", i + 1, e)
            batch_results["failed_simulations"] += 1
            batch_results["errors"].append(
                {
                    "simulation_index": i,
                    "error": str(e),
                    "exception": True,
                }
            )

    total = max(batch_results["total_simulations"], 1)
    rate = batch_results["successful_simulations"] / total
    logger.info(
        "Batch completed: %d success / %d fail (%.1f%%)",
        batch_results["successful_simulations"],
        batch_results["failed_simulations"],
        rate * 100.0,
    )
    return batch_results


__all__ = [
    "execute_pymdp_simulation",
    "execute_pymdp_simulation_from_gnn",
    "execute_from_gnn_file",
    "batch_execute_pymdp",
    "PyMDPSimulation",
    "extract_gnn_dimensions",
    "validate_gnn_pomdp_structure",
]
