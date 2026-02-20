"""
Framework-specific data extractors for post-simulation analysis.

Provides extract_*_data() functions for PyMDP, RxInfer.jl, ActiveInference.jl,
JAX, and DisCoPy execution results.

Extracted from post_simulation.py for maintainability.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def extract_pymdp_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PyMDP-specific data from execution result.
    Enhanced to read from collected files if available.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)

            # Read simulation_results.json if available
            sim_data_dir = impl_path / "simulation_data"
            if sim_data_dir.exists():
                results_files = list(sim_data_dir.glob("*simulation_results.json"))
                if results_files:
                    try:
                        logger.info(f"Reading PyMDP simulation data from {results_files[0].name}")
                        with open(results_files[0], 'r') as f:
                            file_data = json.load(f)

                            # Extract from file data
                            if "beliefs" in file_data:
                                simulation_data["beliefs"] = file_data["beliefs"]
                                logger.debug(f"Extracted {len(file_data['beliefs'])} belief states")
                            if "actions" in file_data:
                                simulation_data["actions"] = file_data["actions"]
                                logger.debug(f"Extracted {len(file_data['actions'])} actions")
                            if "observations" in file_data:
                                simulation_data["observations"] = file_data["observations"]
                                logger.debug(f"Extracted {len(file_data['observations'])} observations")

                            logger.info(f"Enhanced PyMDP data from {results_files[0].name}")
                    except Exception as e:
                        logger.warning(f"Failed to read simulation_results.json: {e}")

            # Count visualizations
            viz_dir = impl_path / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.svg"))
                if viz_files:
                    simulation_data["visualization_count"] = len(viz_files)
                    simulation_data["visualization_files"] = [str(f.name) for f in viz_files]

        except Exception as e:
            logger.debug(f"Error reading PyMDP files: {e}")

    return {
        "traces": simulation_data.get("traces", []),
        "free_energy": simulation_data.get("free_energy", []),
        "states": simulation_data.get("states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),
        "policy": simulation_data.get("policy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "visualization_count": simulation_data.get("visualization_count", 0),
        "visualization_files": simulation_data.get("visualization_files", [])
    }


def extract_rxinfer_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract RxInfer.jl-specific data from execution result.
    Enhanced to read from collected files if available.
    """
    simulation_data = execution_result.get("simulation_data", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)
            results_file = impl_path / "simulation_results.json"
            if results_file.exists():
                logger.info(f"Reading RxInfer simulation data from {results_file.name}")
                with open(results_file, 'r') as f:
                    file_data = json.load(f)
                    if "beliefs" in file_data:
                        simulation_data["beliefs"] = file_data["beliefs"]
                    if "true_states" in file_data:
                        simulation_data["true_states"] = file_data["true_states"]
                    if "observations" in file_data:
                        simulation_data["observations"] = file_data["observations"]
        except Exception as e:
            logger.debug(f"Error reading RxInfer files: {e}")

    return {
        "beliefs": simulation_data.get("beliefs", []),
        "true_states": simulation_data.get("true_states", []),
        "observations": simulation_data.get("observations", []),
        "posterior": simulation_data.get("posterior", []),
        "inference_data": simulation_data.get("inference_data", [])
    }


def extract_activeinference_jl_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ActiveInference.jl-specific data from execution result.
    Enhanced to read from simulation_results.csv if available.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data with full Active Inference fields
    """
    simulation_data = execution_result.get("simulation_data", {})
    model_parameters = simulation_data.get("model_parameters", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)

            # Look for ActiveInference.jl output directories (timestamped)
            # or directly in the implementation directory (if flattened)
            possible_dirs = [impl_path] + list(impl_path.glob("activeinference_outputs_*"))

            csv_found = False
            for search_dir in possible_dirs:
                results_file = search_dir / "simulation_results.csv"
                if results_file.exists():
                    logger.info(f"Reading ActiveInference.jl simulation data from {results_file.name}")
                    import csv

                    traces = []
                    observations = []
                    actions = []
                    beliefs = []

                    with open(results_file, 'r') as f:
                        # Skip comments
                        lines = [line for line in f if not line.startswith('#')]

                        if lines:
                            reader = csv.reader(lines)
                            for row in reader:
                                if len(row) >= 3:
                                    # step, observation, action, belief...
                                    try:
                                        # Parse basic data
                                        observations.append(float(row[1]))
                                        actions.append(float(row[2]))

                                        # Beliefs are the rest of the columns
                                        if len(row) > 3:
                                            belief = [float(x) for x in row[3:]]
                                            beliefs.append(belief)

                                        # Add to generic traces for step counting
                                        traces.append({
                                            "step": int(float(row[0])),
                                            "observation": float(row[1]),
                                            "action": float(row[2])
                                        })
                                    except ValueError:
                                        continue

                    if traces:
                        simulation_data["traces"] = traces
                        simulation_data["observations"] = observations
                        simulation_data["actions"] = actions
                        simulation_data["beliefs"] = beliefs
                        simulation_data["num_timesteps"] = len(traces)
                        csv_found = True
                        logger.info(f"Extracted {len(traces)} steps from ActiveInference.jl results")
                        break

            if not csv_found:
                logger.debug("No simulation_results.csv found for ActiveInference.jl")

        except Exception as e:
            logger.warning(f"Error reading ActiveInference.jl files: {e}")

    # Extract all Active Inference-relevant fields
    extracted = {
        # Core state/observation/action traces
        "traces": simulation_data.get("traces", []),
        "free_energy": simulation_data.get("free_energy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "states": simulation_data.get("states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),

        # Expected free energy components
        "expected_free_energy": simulation_data.get("expected_free_energy", []),
        "expected_energy": simulation_data.get("expected_energy", []),
        "expected_entropy": simulation_data.get("expected_entropy", []),

        # Model parameters (A, B, C, D matrices)
        "num_states": model_parameters.get("num_states", 0),
        "num_observations": model_parameters.get("num_observations", 0),
        "num_actions": model_parameters.get("num_actions", 0),
        "A_matrix": model_parameters.get("A", []),
        "B_matrix": model_parameters.get("B", []),
        "C_vector": model_parameters.get("C", []),
        "D_vector": model_parameters.get("D", []),

        # Precision parameters
        "action_precision": simulation_data.get("action_precision", 1.0),
        "parameter_precision": simulation_data.get("parameter_precision", 1.0),
        "policy_precision": simulation_data.get("policy_precision", 1.0),

        # Inference metadata
        "num_timesteps": simulation_data.get("num_timesteps", 0),
        "inference_iterations": simulation_data.get("inference_iterations", 0),
        "convergence_threshold": simulation_data.get("convergence_threshold", 0.0),

        # Performance metrics
        "execution_time": simulation_data.get("execution_time", 0.0),
        "memory_usage": simulation_data.get("memory_usage", 0.0),

        # Additional Active Inference specific metrics
        "variational_free_energy": simulation_data.get("variational_free_energy", []),
        "information_gain": simulation_data.get("information_gain", []),
        "pragmatic_value": simulation_data.get("pragmatic_value", [])
    }

    return extracted


def extract_jax_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract JAX-specific data from execution result.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data
    """
    # Similar to PyMDP
    return extract_pymdp_data(execution_result)


def extract_discopy_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DisCoPy-specific data from execution result.
    Enhanced to count executions/diagrams as 'steps' if needed.

    Args:
        execution_result: Execution result dictionary

    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})

    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)

            # Read execution report
            possible_reports = [
                impl_path / "discopy_execution_report.json",
                impl_path / "discopy_results" / "discopy_execution_report.json",
                impl_path / "execution_results" / "discopy_results" / "discopy_execution_report.json"
            ]

            for report_file in possible_reports:
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)

                        # Use analysis summary to populate data
                        summary = report_data.get("analysis_summary", {})
                        executions = report_data.get("executions", [])

                        total_processed = summary.get("total_files_processed", 0)
                        jax_analyzed = summary.get("jax_outputs_analyzed", 0)

                        # Populate diagrams list
                        diagrams = []
                        for exec_rec in executions:
                            if exec_rec.get("type") == "diagram_validation" and exec_rec.get("status") == "SUCCESS":
                                diagrams.append(exec_rec.get("file_path"))

                        simulation_data["diagrams"] = diagrams
                        simulation_data["visualization_count"] = len(diagrams)

                        if jax_analyzed > 0:
                             simulation_data["traces"] = [{"step": 1, "type": "jax_eval"} for _ in range(jax_analyzed)]
                        elif total_processed > 0:
                             simulation_data["traces"] = [{"step": 1, "type": "diagram"} for _ in range(total_processed)]

                        logger.info(f"Extracted DisCoPy data: {len(diagrams)} diagrams, {jax_analyzed} JAX evals")
                        break
        except Exception as e:
            logger.warning(f"Error reading DisCoPy files: {e}")

    return {
        "diagrams": simulation_data.get("diagrams", []),
        "circuits": simulation_data.get("circuits", []),
        "traces": simulation_data.get("traces", []),
        "visualization_count": simulation_data.get("visualization_count", 0)
    }
