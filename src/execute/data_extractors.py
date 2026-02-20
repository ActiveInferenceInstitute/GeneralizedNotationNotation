#!/usr/bin/env python3
"""
Data extraction functions for simulation output parsing.

Extracts simulation data from both file-based outputs and stdout/stderr
for all supported frameworks: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from shutil import copy2


# ---------------------------------------------------------------------------
# Path normalization and output collection
# ---------------------------------------------------------------------------

def normalize_and_deduplicate_paths(found_files: List[Path], logger) -> List[Path]:
    """
    Normalize paths and remove duplicates/nested paths.

    Args:
        found_files: List of file paths to normalize
        logger: Logger instance for logging

    Returns:
        Deduplicated list of normalized paths
    """
    if not found_files:
        return []

    normalized = {}
    for file_path in found_files:
        try:
            abs_path = file_path.resolve()
            if abs_path not in normalized:
                normalized[abs_path] = file_path
        except (OSError, RuntimeError) as e:
            logger.debug(f"Skipping invalid path {file_path}: {e}")
            continue

    sorted_paths = sorted(normalized.values(), key=lambda p: len(p.parts))
    deduplicated = []
    seen_names = set()

    for file_path in sorted_paths:
        file_name = file_path.name
        file_parent = file_path.parent

        is_nested_duplicate = False
        for seen_path in deduplicated:
            seen_parent = seen_path.parent
            try:
                if file_parent.is_relative_to(seen_parent) and file_name == seen_path.name:
                    is_nested_duplicate = True
                    logger.debug(f"Skipping nested duplicate: {file_path} (already have {seen_path})")
                    break
            except (ValueError, AttributeError):
                try:
                    if str(file_parent).startswith(str(seen_parent)) and file_name == seen_path.name:
                        is_nested_duplicate = True
                        logger.debug(f"Skipping nested duplicate: {file_path} (already have {seen_path})")
                        break
                except Exception as e:
                    logger.debug(f"Error comparing paths {file_path} and {seen_path}: {e}")

        if not is_nested_duplicate:
            deduplicated.append(file_path)
            seen_names.add(file_name)

    if len(found_files) != len(deduplicated):
        logger.info(f"Deduplicated paths: {len(found_files)} -> {len(deduplicated)} files")

    return deduplicated


def collect_execution_outputs(
    script_path: Path,
    output_dir: Path,
    framework: str,
    logger
) -> Dict[str, List[str]]:
    """
    Collect all outputs from executed script and copy to execute output directory.

    Args:
        script_path: Path to the executed script
        output_dir: Execute output directory for this model/framework
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with lists of copied file paths by category
    """
    collected = {
        "visualizations": [],
        "simulation_data": [],
        "traces": [],
        "other": []
    }

    try:
        script_dir = script_path.parent

        found_files = []

        if framework == "pymdp":
            pymdp_output = script_dir / "output" / "pymdp_simulations"
            if pymdp_output.exists():
                found_files.extend(pymdp_output.rglob("*.png"))
                found_files.extend(pymdp_output.rglob("*.svg"))
                found_files.extend(pymdp_output.rglob("*.json"))
                found_files.extend(pymdp_output.rglob("*.pkl"))
        elif framework == "discopy":
            discopy_dir = script_dir / "discopy_diagrams"
            if discopy_dir.exists():
                found_files.extend(discopy_dir.rglob("*.png"))
                found_files.extend(discopy_dir.rglob("*.svg"))
                found_files.extend(discopy_dir.rglob("*.json"))
        elif framework == "activeinference_jl":
            for out_dir in script_dir.glob("activeinference_outputs_*"):
                if out_dir.is_dir():
                    viz_dir = out_dir / "visualizations"
                    if viz_dir.exists() and viz_dir.is_dir():
                        found_files.extend(viz_dir.glob("*.png"))
                        found_files.extend(viz_dir.glob("*.svg"))
                    sim_data_dir = out_dir / "simulation_data"
                    if sim_data_dir.exists() and sim_data_dir.is_dir():
                        found_files.extend(sim_data_dir.glob("*.json"))
                        found_files.extend(sim_data_dir.glob("*.csv"))
                    traces_dir = out_dir / "free_energy_traces"
                    if traces_dir.exists() and traces_dir.is_dir():
                        found_files.extend(traces_dir.glob("*.json"))
                        found_files.extend(traces_dir.glob("*.csv"))
        elif framework == "rxinfer":
            rxinfer_dir = script_dir / "rxinfer_outputs"
            if rxinfer_dir.exists():
                found_files.extend(rxinfer_dir.rglob("*.png"))
                found_files.extend(rxinfer_dir.rglob("*.json"))
                found_files.extend(rxinfer_dir.rglob("*.csv"))
        elif framework == "jax":
            jax_dir = script_dir / "jax_outputs"
            if jax_dir.exists():
                found_files.extend(jax_dir.rglob("*.png"))
                found_files.extend(jax_dir.rglob("*.json"))

        if not found_files:
            found_files.extend(script_dir.rglob("*.png"))
            found_files.extend(script_dir.rglob("*.svg"))
            found_files.extend(script_dir.rglob("*.json"))
            found_files.extend(script_dir.rglob("*.pkl"))
            found_files.extend(script_dir.rglob("*.csv"))

        found_files = [f for f in found_files if f != script_path and f.exists() and f.is_file()]
        found_files = normalize_and_deduplicate_paths(found_files, logger)

        if not found_files:
            logger.debug(f"No output files found for {framework} script {script_path.name}")
            return collected

        logger.info(f"Found {len(found_files)} output files to collect for {framework}")

        for source_file in found_files:
            try:
                ext = source_file.suffix.lower()

                if ext in ['.png', '.svg', '.jpg', '.jpeg']:
                    logger.debug(f"Skipping visualization {source_file.name} (will be collected by analysis step)")
                    continue

                if ext in ['.json', '.pkl', '.csv']:
                    if 'trace' in source_file.name.lower() or 'posterior' in source_file.name.lower():
                        dest_dir = output_dir / "traces"
                        category = "traces"
                    else:
                        dest_dir = output_dir / "simulation_data"
                        category = "simulation_data"
                else:
                    dest_dir = output_dir / "other"
                    category = "other"

                dest_dir.mkdir(parents=True, exist_ok=True)

                dest_file = dest_dir / source_file.name

                if dest_file.exists():
                    try:
                        source_stat = source_file.stat()
                        dest_stat = dest_file.stat()
                        if (source_stat.st_size == dest_stat.st_size and
                            abs(source_stat.st_mtime - dest_stat.st_mtime) < 1.0):
                            logger.debug(f"Skipping duplicate: {dest_file.name} already exists")
                            collected[category].append(str(dest_file))
                            continue
                    except OSError:
                        pass

                    parent_name = source_file.parent.name
                    if not source_file.name.startswith(f"{parent_name}_"):
                        dest_file = dest_dir / f"{parent_name}_{source_file.name}"
                    else:
                        counter = 1
                        base_name = source_file.stem
                        ext = source_file.suffix
                        while dest_file.exists():
                            dest_file = dest_dir / f"{base_name}_{counter}{ext}"
                            counter += 1

                copy2(source_file, dest_file)

                collected[category].append(str(dest_file))
                logger.info(f"Copied {source_file.name} -> {dest_file.relative_to(output_dir)}")

            except Exception as e:
                logger.warning(f"Failed to copy {source_file}: {e}")

        total_copied = sum(len(files) for files in collected.values())
        if total_copied > 0:
            logger.info(f"Collected {total_copied} output files: "
                       f"{len(collected['visualizations'])} visualizations, "
                       f"{len(collected['simulation_data'])} data files, "
                       f"{len(collected['traces'])} traces")

    except Exception as e:
        logger.error(f"Error collecting execution outputs: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return collected


# ---------------------------------------------------------------------------
# File-based extraction (reads saved simulation artifacts)
# ---------------------------------------------------------------------------

def extract_simulation_data_from_files(
    output_dir: Path,
    framework: str,
    logger
) -> Dict[str, Any]:
    """
    Extract simulation data from collected files (not just stdout/stderr).

    Args:
        output_dir: Directory containing collected output files
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    enhanced_data = {}

    try:
        if framework == "pymdp":
            enhanced_data = extract_pymdp_data_from_files(output_dir, logger)
        elif framework == "rxinfer":
            enhanced_data = extract_rxinfer_data_from_files(output_dir, logger)
        elif framework == "activeinference_jl":
            enhanced_data = extract_activeinference_jl_data_from_files(output_dir, logger)
        elif framework == "discopy":
            enhanced_data = extract_discopy_data_from_files(output_dir, logger)
        elif framework == "jax":
            enhanced_data = extract_jax_data_from_files(output_dir, logger)

    except Exception as e:
        logger.warning(f"Failed to extract simulation data from files for {framework}: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return enhanced_data


def extract_pymdp_data_from_files(output_dir: Path, logger) -> Dict[str, Any]:
    """Extract PyMDP simulation data from saved files."""
    data = {}

    try:
        # Look for simulation_results.json
        sim_data_dir = output_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                results_file = results_files[0]
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                    # Extract beliefs, actions, observations
                    if "beliefs" in results:
                        data["beliefs"] = results["beliefs"]
                    if "actions" in results:
                        data["actions"] = results["actions"]
                    if "observations" in results:
                        data["observations"] = results["observations"]
                    if "num_timesteps" in results:
                        data["num_timesteps"] = results["num_timesteps"]

                    logger.info(f"Extracted PyMDP data from {results_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to parse {results_file}: {e}")

        # Count visualizations
        viz_dir = output_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.svg"))
            if viz_files:
                data["visualization_count"] = len(viz_files)
                data["visualization_files"] = [str(f.name) for f in viz_files]

    except Exception as e:
        logger.warning(f"Error extracting PyMDP data from files: {e}")

    return data


def extract_rxinfer_data_from_files(output_dir: Path, logger) -> Dict[str, Any]:
    """Extract RxInfer.jl simulation data from saved files."""
    data = {}

    try:
        # Look for inference data JSON files
        data_dir = output_dir / "inference_data"
        if data_dir.exists():
            json_files = list(data_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        inference_data = json.load(f)
                        if "posterior" in inference_data:
                            data["posterior"] = inference_data["posterior"]
                except Exception as e:
                    logger.debug(f"Error reading inference data from {json_file}: {e}")

        # Look for trace files
        trace_dir = output_dir / "posterior_traces"
        if trace_dir.exists():
            trace_files = list(trace_dir.glob("*.csv"))
            if trace_files:
                data["trace_files"] = [str(f.name) for f in trace_files]

    except Exception as e:
        logger.warning(f"Error extracting RxInfer data from files: {e}")

    return data


def extract_activeinference_jl_data_from_files(output_dir: Path, logger) -> Dict[str, Any]:
    """Extract ActiveInference.jl simulation data from saved files."""
    data = {}

    try:
        # Look for activeinference_outputs_* directories (timestamped output dirs)
        output_dirs = list(output_dir.glob("activeinference_outputs_*"))
        if not output_dirs:
            # Also check parent directories
            output_dirs = list(output_dir.parent.glob("**/activeinference_outputs_*"))

        # Get most recent output directory
        if output_dirs:
            output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_output = output_dirs[0]
            logger.debug(f"Found ActiveInference.jl output directory: {latest_output}")

            # Parse model_parameters.json
            params_file = latest_output / "model_parameters.json"
            if params_file.exists():
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                        data["model_name"] = params.get("model_name")
                        data["n_states"] = params.get("n_states")
                        data["n_observations"] = params.get("n_observations")
                        data["n_actions"] = params.get("n_actions")
                        data["timestamp"] = params.get("timestamp")
                        logger.debug(f"Loaded model parameters from {params_file}")
                except Exception as e:
                    logger.warning(f"Error reading model_parameters.json: {e}")

            # Parse simulation_results.csv
            results_csv = latest_output / "simulation_results.csv"
            if results_csv.exists():
                try:
                    import csv
                    with open(results_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            data["timesteps"] = len(rows)
                            data["observations"] = [int(r.get("observation", 0)) for r in rows]
                            data["actions"] = [int(r.get("action", 0)) for r in rows]
                            # Extract beliefs if available
                            belief_keys = [k for k in rows[0].keys() if k.startswith("belief")]
                            if belief_keys:
                                data["beliefs"] = [[float(r.get(k, 0)) for k in belief_keys] for r in rows]
                            logger.debug(f"Loaded {len(rows)} timesteps from simulation_results.csv")
                except Exception as e:
                    logger.warning(f"Error reading simulation_results.csv: {e}")

            # Parse summary.txt for validation status
            summary_file = latest_output / "summary.txt"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary_text = f.read()
                        data["validation_passed"] = "PASSED" in summary_text
                        logger.debug(f"Read summary from {summary_file}")
                except Exception as e:
                    logger.warning(f"Error reading summary.txt: {e}")

            # Count visualizations
            viz_files = list(latest_output.glob("*.png"))
            if viz_files:
                data["visualization_count"] = len(viz_files)
                data["visualization_files"] = [f.name for f in viz_files]

        # Also check traditional locations for backwards compatibility
        data_dir = output_dir / "simulation_data"
        if data_dir.exists():
            json_files = list(data_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        sim_data = json.load(f)
                        if "free_energy" in sim_data and "free_energy" not in data:
                            data["free_energy"] = sim_data["free_energy"]
                        if "beliefs" in sim_data and "beliefs" not in data:
                            data["beliefs"] = sim_data["beliefs"]
                except Exception as e:
                    logger.debug(f"Error reading simulation data from {json_file}: {e}")

        # Look for free energy traces
        fe_dir = output_dir / "free_energy_traces"
        if fe_dir.exists():
            trace_files = list(fe_dir.glob("*.csv"))
            if trace_files:
                data["free_energy_trace_files"] = [str(f.name) for f in trace_files]

    except Exception as e:
        logger.warning(f"Error extracting ActiveInference.jl data from files: {e}")

    return data


def extract_discopy_data_from_files(output_dir: Path, logger) -> Dict[str, Any]:
    """Extract DisCoPy simulation data from saved files."""
    data = {}

    try:
        # Look for circuit analysis JSON in multiple possible locations
        search_dirs = [
            output_dir / "simulation_data",
            output_dir / "discopy_diagrams",
            output_dir / "analysis",
            output_dir
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                json_files = list(search_dir.glob("*circuit*.json")) + list(search_dir.glob("*analysis*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            circuit_data = json.load(f)
                            if "circuit" in circuit_data:
                                data["circuit"] = circuit_data["circuit"]
                            if "components" in circuit_data:
                                data["components"] = circuit_data["components"]
                            if "analysis" in circuit_data:
                                data["analysis"] = circuit_data["analysis"]
                            if "parameters" in circuit_data:
                                data["parameters"] = circuit_data["parameters"]
                            logger.debug(f"Loaded DisCoPy data from {json_file}")
                    except Exception as e:
                        logger.debug(f"Error reading DisCoPy data from {json_file}: {e}")

        # Count diagram outputs in multiple possible locations
        diagram_dirs = [
            output_dir / "discopy_diagrams",
            output_dir / "diagram_outputs",
            output_dir / "simulation_data"
        ]

        for diagram_dir in diagram_dirs:
            if diagram_dir.exists():
                diagram_files = list(diagram_dir.glob("*.png"))
                if diagram_files:
                    data["diagram_count"] = len(diagram_files)
                    data["diagram_files"] = [str(f.name) for f in diagram_files]
                    logger.debug(f"Found {len(diagram_files)} DisCoPy diagrams in {diagram_dir}")
                    break

    except Exception as e:
        logger.warning(f"Error extracting DisCoPy data from files: {e}")

    return data


def extract_jax_data_from_files(output_dir: Path, logger) -> Dict[str, Any]:
    """Extract JAX simulation data from saved files."""
    # Similar to PyMDP
    return extract_pymdp_data_from_files(output_dir, logger)


# ---------------------------------------------------------------------------
# Stdout/stderr-based extraction (parses execution output text)
# ---------------------------------------------------------------------------

def extract_simulation_data(stdout: str, stderr: str, framework: str, logger) -> Dict[str, Any]:
    """
    Extract simulation data from execution output.

    Args:
        stdout: Standard output from script execution
        stderr: Standard error from script execution
        framework: Framework name
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    simulation_data = {
        "traces": [],
        "free_energy": [],
        "states": [],
        "observations": [],
        "actions": [],
        "policy": [],
        "raw_output": stdout[:10000] if stdout else "",  # Limit size
        "raw_error": stderr[:10000] if stderr else ""
    }

    try:
        # Framework-specific extraction
        if framework == "pymdp":
            simulation_data.update(extract_pymdp_data(stdout, stderr))
        elif framework == "rxinfer":
            simulation_data.update(extract_rxinfer_data(stdout, stderr))
        elif framework == "activeinference_jl":
            simulation_data.update(extract_activeinference_jl_data(stdout, stderr))
        elif framework == "jax":
            simulation_data.update(extract_jax_data(stdout, stderr))
        elif framework == "discopy":
            simulation_data.update(extract_discopy_data(stdout, stderr))
        else:
            # Generic extraction - try to find common patterns
            simulation_data.update(extract_generic_data(stdout, stderr))

    except Exception as e:
        logger.warning(f"Failed to extract simulation data for {framework}: {e}")

    return simulation_data


def extract_pymdp_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract PyMDP-specific simulation data from stdout/stderr."""
    data = {}

    # Combine stdout and stderr for parsing
    combined_output = stdout + "\n" + stderr

    # Try to find observations from log lines like "Step 0: obs=2, belief=[...], action=0.0"
    obs_pattern = r'Step\s+\d+:\s+obs=(\d+)'
    obs_matches = re.findall(obs_pattern, combined_output, re.IGNORECASE)
    if obs_matches:
        data["observations"] = [int(obs) for obs in obs_matches]

    # Try to find actions from log lines
    action_pattern = r'Step\s+\d+:\s+obs=\d+,\s+belief=[^\]]+,\s+action=([\d.]+)'
    action_matches = re.findall(action_pattern, combined_output, re.IGNORECASE)
    if action_matches:
        data["actions"] = [int(float(act)) for act in action_matches]

    # Try to find beliefs from log lines
    belief_pattern = r'belief=\[([^\]]+)\]'
    belief_matches = re.findall(belief_pattern, combined_output, re.IGNORECASE)
    if belief_matches:
        try:
            beliefs = []
            for match in belief_matches:
                # Parse array string like "0.05 0.05 0.9" or "0.05, 0.05, 0.9"
                values = re.findall(r'[\d.]+', match)
                if values:
                    beliefs.append([float(v) for v in values])
            if beliefs:
                data["beliefs"] = beliefs
        except Exception as e:
            logging.getLogger(__name__).debug(f"Error parsing belief data: {e}")

    # Try to find state trajectories (legacy pattern)
    state_pattern = r'state[:\s]+\[([^\]]+)\]|states[:\s]+\[([^\]]+)\]'
    state_matches = re.findall(state_pattern, combined_output, re.IGNORECASE)
    if state_matches and "states" not in data:
        data["states"] = [match[0] or match[1] for match in state_matches]

    # Try to find observations (legacy pattern)
    if "observations" not in data:
        obs_pattern_legacy = r'observation[:\s]+(\d+)|obs[:\s]+(\d+)'
        obs_matches_legacy = re.findall(obs_pattern_legacy, combined_output, re.IGNORECASE)
        if obs_matches_legacy:
            data["observations"] = [int(match[0] or match[1]) for match in obs_matches_legacy]

    # Try to find actions (legacy pattern)
    if "actions" not in data:
        action_pattern_legacy = r'action[:\s]+(\d+)|action_taken[:\s]+(\d+)'
        action_matches_legacy = re.findall(action_pattern_legacy, combined_output, re.IGNORECASE)
        if action_matches_legacy:
            data["actions"] = [int(match[0] or match[1]) for match in action_matches_legacy]

    # Try to find free energy
    fe_pattern = r'free[_\s]?energy[:\s]+([\d.]+)|FE[:\s]+([\d.]+)'
    fe_matches = re.findall(fe_pattern, combined_output, re.IGNORECASE)
    if fe_matches:
        data["free_energy"] = [float(match[0] or match[1]) for match in fe_matches]

    return data


def extract_rxinfer_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract RxInfer.jl-specific simulation data."""
    data = {}

    # Try to find posterior distributions
    posterior_pattern = r'posterior[:\s]+\[([^\]]+)\]'
    posterior_matches = re.findall(posterior_pattern, stdout, re.IGNORECASE)
    if posterior_matches:
        data["posterior"] = posterior_matches

    return data


def extract_activeinference_jl_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract ActiveInference.jl-specific simulation data."""
    data = {}

    # Try to find free energy traces
    fe_pattern = r'free[_\s]?energy[:\s]+([\d.]+)|FE[:\s]+([\d.]+)'
    fe_matches = re.findall(fe_pattern, stdout, re.IGNORECASE)
    if fe_matches:
        data["free_energy"] = [float(match[0] or match[1]) for match in fe_matches]

    # Try to find state beliefs
    belief_pattern = r'belief[:\s]+\[([^\]]+)\]|q\(s\)[:\s]+\[([^\]]+)\]'
    belief_matches = re.findall(belief_pattern, stdout, re.IGNORECASE)
    if belief_matches:
        data["beliefs"] = [match[0] or match[1] for match in belief_matches]

    return data


def extract_jax_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract JAX-specific simulation data."""
    # Similar to PyMDP but may have different output format
    return extract_pymdp_data(stdout, stderr)


def extract_discopy_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract DisCoPy-specific simulation data."""
    data = {}

    # Try to find diagram information
    diagram_pattern = r'diagram[:\s]+(\w+)|circuit[:\s]+(\w+)'
    diagram_matches = re.findall(diagram_pattern, stdout, re.IGNORECASE)
    if diagram_matches:
        data["diagrams"] = [match[0] or match[1] for match in diagram_matches]

    return data


def extract_generic_data(stdout: str, stderr: str) -> Dict[str, Any]:
    """Generic extraction for unknown frameworks."""
    data = {}

    # Try to find any numeric arrays or lists
    array_pattern = r'\[([\d.,\s]+)\]'
    array_matches = re.findall(array_pattern, stdout)
    if array_matches:
        data["arrays"] = array_matches[:10]  # Limit to first 10

    return data
