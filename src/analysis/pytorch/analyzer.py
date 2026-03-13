#!/usr/bin/env python3
"""
PyTorch Analyzer for GNN Pipeline

Reads simulation_results.json produced by PyTorch runner and generates
belief trajectory, action distribution, and EFE analysis plots.

@Web: https://pytorch.org/docs/stable/
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[str] = []

    # Search recursively for PyTorch simulation results
    pytorch_results = list(results_dir.rglob("**/pytorch/**/simulation_results.json"))
    # Also check for pytorch-prefixed results
    pytorch_results += list(results_dir.rglob("**/pytorch_simulation_results.json"))
    # Recovery: check root
    root_result = results_dir / "simulation_results.json"
    if root_result.exists() and root_result not in pytorch_results:
        pytorch_results.append(root_result)

    if not pytorch_results:
        if verbose:
            logger.debug(f"No PyTorch simulation_results.json found under {results_dir}")
        return generated_files

    for results_file in pytorch_results:

        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read results: {e}")
            continue

        # Determine model name from path
        path_parts = results_file.parts
        model_name = results.get("model_name", "unknown")
        for i, part in enumerate(path_parts):
            if part == "pytorch" and i >= 1:
                model_name = path_parts[i - 1]
                break

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        beliefs = np.array(results.get("beliefs", []))
        actions = results.get("actions", [])
        observations = results.get("observations", [])
        efe = np.array(results.get("efe_history", []))
        validation = results.get("validation", {})

        analysis = {
            "framework": "pytorch",
            "model_name": model_name,
            "num_timesteps": len(actions),
            "num_states": beliefs.shape[1] if beliefs.ndim == 2 else 0,
            "validation": validation,
            "metrics": {},
        }

        # Compute metrics
        if beliefs.ndim == 2 and beliefs.shape[0] > 0:
            entropy = -np.sum(beliefs * np.log(beliefs + 1e-16), axis=1)
            analysis["metrics"]["mean_belief_entropy"] = float(np.mean(entropy))
            analysis["metrics"]["final_belief_entropy"] = float(entropy[-1])
            confidence = np.max(beliefs, axis=1)
            analysis["metrics"]["mean_confidence"] = float(np.mean(confidence))
            analysis["metrics"]["final_confidence"] = float(confidence[-1])

        if actions:
            unique, counts = np.unique(actions, return_counts=True)
            analysis["metrics"]["action_distribution"] = {
                int(a): int(c) for a, c in zip(unique, counts)
            }

        if efe.ndim == 2 and efe.shape[0] > 0:
            analysis["metrics"]["mean_efe"] = float(np.mean(efe))

        # Generate plots
        try:
            _generate_plots(beliefs, actions, observations, efe, model_output_dir)
            analysis["plots_generated"] = True
        except Exception as e:
            logger.warning(f"Plot generation failed for {model_name}: {e}")
            analysis["plots_generated"] = False

        # Save analysis
        analysis_file = model_output_dir / "pytorch_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        generated_files.append(str(analysis_file))
        logger.info(f"✅ PyTorch analysis saved: {model_name}")

    return generated_files


def _generate_plots(
    beliefs: np.ndarray,
    actions: list,
    observations: list,
    efe: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate analysis plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    # Belief trajectory
    if beliefs.ndim == 2 and beliefs.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for s in range(beliefs.shape[1]):
            ax.plot(beliefs[:, s], label=f"State {s}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("Belief", fontsize=14)
        ax.set_title("PyTorch — Belief Trajectory", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "belief_trajectory.png", dpi=150)
        plt.close(fig)

    # Action distribution
    if actions:
        fig, ax = plt.subplots(figsize=(6, 4))
        unique, counts = np.unique(actions, return_counts=True)
        ax.bar(unique.astype(str), counts, color="#4285F4")
        ax.set_xlabel("Action", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title("PyTorch — Action Distribution", fontsize=16)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "action_distribution.png", dpi=150)
        plt.close(fig)

    # EFE history
    if efe.ndim == 2 and efe.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for a_idx in range(efe.shape[1]):
            ax.plot(efe[:, a_idx], label=f"Action {a_idx}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("EFE", fontsize=14)
        ax.set_title("PyTorch — Expected Free Energy", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "efe_history.png", dpi=150)
        plt.close(fig)

    logger.info(f"✅ PyTorch analysis plots saved to: {output_dir}")
