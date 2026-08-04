"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.

This module is exercised for the pomdp_gridworld exemplar but is hardened to be
generic across ALL exemplars whose ``simulation_results.json`` conform to the
``rxinfer_simulation_v1`` schema. Different exemplars emit slightly different
mixtures of keys:

* ``beliefs`` / ``beliefs_by_factor`` (dict-shaped: ``{"joint_state": [...]}``)
* ``observations`` / ``observations_by_modality``
* ``true_states`` / ``hidden_states_by_factor``
* ``actions`` / ``actions_by_control_factor``
* ``expected_free_energy`` / ``variational_free_energy`` / ``efe_per_action``
* ``policy_posterior`` (optional)
* ``metrics``, ``validation`` (optional)

The analyzer normalises these into flat arrays and renders a consistent,
comprehensive visualization set for any result that carries the core belief /
observation arrays, skipping optional plots gracefully when their data is
absent rather than raising.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import shared visualization utilities (centralized matplotlib setup)
from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt

# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------
_CORE_PLOT_TYPES: List[str] = [
    "belief_evolution",
    "obs_vs_true",
    "belief_heatmap",
    "belief_entropy",
    "accuracy",
    "action_frequencies",
    "belief_convergence",
    "belief_trace",
    "free_energy",
    "observations",
]


def _first_dict_value(value: Any) -> Any:
    """If value is a non-empty dict, return the first entry's value.

    RxInfer results often store arrays inside a ``{factor_name: [...]}`` dict
    (e.g. ``beliefs_by_factor`` -> ``{"joint_state": [...]}``). Helpers pull out
    the first such array so downstream code sees a plain list.
    """
    if isinstance(value, dict) and value:
        first_key = next(iter(value))
        return value[first_key]
    return value


def _as_flat_list(value: Any) -> List[Any]:
    """Best-effort conversion of a raw JSON value into a flat Python list.

    Handles lists, dicts whose single value is a list, and scalar values
    (wrapped in a single-element list).
    """
    if value is None:
        return []
    value = _first_dict_value(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    # numpy / scalar
    if np is not None and isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, (int, float)):
        return [value]
    return []


def _as_2d_list(value: Any) -> List[List[float]]:
    """Best-effort conversion into a list of sequence rows (2D-ish)."""
    value = _first_dict_value(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    rows: List[List[float]] = []
    for row in value:
        if isinstance(row, (list, tuple)):
            rows.append([float(x) for x in row])
        elif np is not None and isinstance(row, np.ndarray):
            rows.append([float(x) for x in row.tolist()])
        elif isinstance(row, (int, float)):
            rows.append([float(row)])
    return rows


def _normalise_beliefs(data: Dict[str, Any]) -> List[List[float]]:
    """Return beliefs as a list of rows, from either key form."""
    beliefs = data.get("beliefs")
    if beliefs is None:
        beliefs = data.get("beliefs_by_factor")
    rows = _as_2d_list(beliefs)
    if rows and all(np is not None and len(r) != len(rows[0]) for r in rows):
        # ragged rows cannot be plotted together; fall back to empty
        return []
    return rows


def _normalise_obs(data: Dict[str, Any]) -> List[float]:
    obs = data.get("observations")
    if obs is None:
        obs = data.get("observations_by_modality")
    return [float(x) for x in _as_flat_list(obs)]


def _normalise_true_states(data: Dict[str, Any]) -> List[float]:
    states = data.get("true_states")
    if states is None:
        states = data.get("hidden_states_by_factor")
    return [float(x) for x in _as_flat_list(states)]


def _normalise_actions(data: Dict[str, Any]) -> List[float]:
    actions = data.get("actions")
    if actions is None:
        actions = data.get("actions_by_control_factor")
    return _as_flat_list(actions)


def _normalise_free_energy(data: Dict[str, Any]) -> List[float]:
    """Best available expected / variational free energy trace."""
    efe = data.get("expected_free_energy")
    if efe is None:
        efe = data.get("variational_free_energy")
    efe_list = _as_flat_list(efe)
    if efe_list:
        return [float(x) for x in efe_list]
    # Fall back to per-action EFE mean across actions per step.
    efe_per_action = _as_2d_list(data.get("efe_per_action"))
    if efe_per_action and np is not None:
        arr = np.asarray(efe_per_action, dtype=float)
        if arr.ndim == 2:
            return [float(v) for v in np.mean(arr, axis=1)]
    return []


def _normalise_policy_posterior(data: Dict[str, Any]) -> List[List[float]]:
    pp = data.get("policy_posterior")
    if pp is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            pp = metrics.get("policy_posterior")
    return _as_2d_list(pp)


def _normalise_efe_per_action(data: Dict[str, Any]) -> List[List[float]]:
    eaa = data.get("efe_per_action")
    if eaa is None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            eaa = metrics.get("efe_per_action")
    return _as_2d_list(eaa)


# ---------------------------------------------------------------------------
# Current-model helpers
# ---------------------------------------------------------------------------
def _current_rxinfer_models(execution_dir: Path) -> set[str] | None:
    """Return model names for RxInfer entries in the current Step 12 summary."""
    summary_file = execution_dir / "summaries" / "execution_summary.json"
    if not summary_file.exists():
        summary_file = execution_dir / "execution_summary.json"
    if not summary_file.exists():
        return None
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    details = data.get("execution_details")
    if not isinstance(details, list):
        details = data.get("execution_results", [])
    if not isinstance(details, list):
        return None
    models = {
        str(detail.get("model_name"))
        for detail in details
        if isinstance(detail, dict)
        and str(detail.get("framework", "")).lower() == "rxinfer"
        and detail.get("model_name")
    }
    return models or None


def _latest_current_results_file(sim_data_dir: Path) -> Path | None:
    """Select one current RxInfer result file from a simulation_data directory."""
    candidates = sorted(
        sim_data_dir.glob("*simulation_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("schema_version") == "rxinfer_simulation_v1":
            return candidate
    return candidates[0] if candidates else None


def generate_analysis_from_logs(
    execution_dir: Path, output_dir: Path, verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from RxInfer execution logs.

    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging

    Returns:
        List of generated visualization file paths
    """
    visualizations: list[Any] = []

    try:
        # Find RxInfer execution results
        current_models = _current_rxinfer_models(execution_dir)
        rxinfer_dirs = list(execution_dir.glob("*/rxinfer"))

        for rxinfer_dir in rxinfer_dirs:
            model_name = rxinfer_dir.parent.name
            if current_models and model_name not in current_models:
                continue
            sim_data_dir = rxinfer_dir / "simulation_data"
            if sim_data_dir.exists():
                # Load simulation results
                results_file = _latest_current_results_file(sim_data_dir)
                if results_file is not None:
                    try:
                        with open(results_file, "r") as f:
                            data = json.load(f)

                        viz_files = create_rxinfer_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)

                    except Exception as e:
                        logger.warning(f"Failed to process {results_file}: {e}")

    except Exception as e:
        logger.error(f"RxInfer analysis failed: {e}")

    return visualizations


def create_rxinfer_visualizations(
    data: Dict[str, Any], output_dir: Path, model_name: str, verbose: bool = False
) -> List[str]:
    """
    Create visualizations from RxInfer simulation data.

    Produces a comprehensive, consistent visualization set for any
    ``rxinfer_simulation_v1`` result. Core plots (belief evolution,
    observation-vs-true, belief heatmap, belief entropy) are emitted whenever
    the required arrays exist; optional plots (accuracy, action frequencies,
    belief convergence, belief trace, free energy, observations) are skipped
    gracefully when their source data is missing — never raising.

    Args:
        data: Simulation results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE or plt is None or np is None:
        logger.warning("Matplotlib unavailable, skipping RxInfer visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Normalise data into flat arrays (tolerates dict-shaped / missing keys)
    beliefs = _normalise_beliefs(data)
    observations = _normalise_obs(data)
    true_states = _normalise_true_states(data)
    actions = _normalise_actions(data)
    free_energy = _normalise_free_energy(data)

    beliefs_arr = np.asarray(beliefs, dtype=float) if beliefs else np.zeros((0, 0))
    have_beliefs = beliefs_arr.ndim >= 1
    have_2d_beliefs = beliefs_arr.ndim == 2 and beliefs_arr.shape[0] > 0

    def _record(plot_type: str) -> Optional[str]:
        viz_file = output_dir / f"{model_name}_rxinfer_{plot_type}.png"
        if viz_file.exists() and viz_file.stat().st_size > 0:
            visualizations.append(str(viz_file))
            logger.info(f"Generated {plot_type}: {viz_file.name}")
            return str(viz_file)
        return None

    # 1. Belief Evolution Plot
    if have_beliefs:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            if have_2d_beliefs:
                for i in range(beliefs_arr.shape[1]):
                    ax.plot(beliefs_arr[:, i], label=f"State {i + 1}", linewidth=2)
            else:
                ax.plot(beliefs_arr, label="Belief", linewidth=2)

            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Belief Probability", fontweight="bold")
            ax.set_title(f"RxInfer Belief Evolution - {model_name}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_evolution.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief evolution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 2. Observation vs True State Plot
    if observations and true_states:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(len(observations))
            ax.scatter(x, observations, label="Observations", alpha=0.7, s=50)
            ax.scatter(x, true_states, label="True States", alpha=0.7, s=50, marker="x")
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("State/Observation", fontweight="bold")
            ax.set_title(
                f"RxInfer Observations vs True States - {model_name}", fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_obs_vs_true.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated obs vs true: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create obs plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 3. Belief Heatmap (2D visualization of beliefs over time)
    if have_2d_beliefs and beliefs_arr.shape[0] > 1:
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(
                beliefs_arr.T,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("State", fontweight="bold")
            ax.set_title(f"RxInfer Belief Heatmap - {model_name}", fontweight="bold")
            ax.set_yticks(range(beliefs_arr.shape[1]))
            ax.set_yticklabels([f"State {i + 1}" for i in range(beliefs_arr.shape[1])])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Belief Probability", fontweight="bold")

            viz_file = output_dir / f"{model_name}_rxinfer_belief_heatmap.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief heatmap: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 4. Belief Entropy (uncertainty tracking over time)
    if have_2d_beliefs:
        try:
            # Calculate entropy for each timestep: H = -sum(p * log(p))
            epsilon = 1e-10
            beliefs_clipped = np.clip(beliefs_arr, epsilon, 1.0)
            entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped), axis=1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(entropy, "purple", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(entropy)), entropy, alpha=0.3, color="purple")
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Belief Entropy (bits)", fontweight="bold")
            ax.set_title(
                f"RxInfer Belief Uncertainty - {model_name}", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            max_entropy = np.log2(beliefs_arr.shape[1])
            ax.axhline(
                y=max_entropy,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Max Entropy ({max_entropy:.2f})",
            )
            ax.legend()

            viz_file = output_dir / f"{model_name}_rxinfer_belief_entropy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief entropy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create entropy plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 5. Inference Accuracy (if we have true states + 2D beliefs)
    if have_2d_beliefs and true_states:
        try:
            inferred_states = np.argmax(beliefs_arr, axis=1) + 1  # 1-indexed
            true_arr = np.asarray(true_states[: len(inferred_states)], dtype=float)
            true_arr = true_arr + 1  # to match 1-indexed inferred states
            n = min(len(inferred_states), len(true_arr))

            matches = (inferred_states[:n] == true_arr[:n]).astype(int)
            cumulative_accuracy = np.cumsum(matches) / (np.arange(n) + 1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(cumulative_accuracy * 100, "green", linewidth=2)
            ax.fill_between(
                range(len(cumulative_accuracy)),
                cumulative_accuracy * 100,
                alpha=0.3,
                color="green",
            )
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Cumulative Accuracy (%)", fontweight="bold")
            ax.set_title(
                f"RxInfer Inference Accuracy - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

            if len(cumulative_accuracy) > 0:
                final_acc = cumulative_accuracy[-1] * 100
                ax.axhline(y=final_acc, color="navy", linestyle="--", alpha=0.5)
                ax.text(
                    len(cumulative_accuracy) - 1,
                    final_acc + 3,
                    f"Final: {final_acc:.1f}%",
                    ha="right",
                    fontweight="bold",
                )

            viz_file = output_dir / f"{model_name}_rxinfer_accuracy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated inference accuracy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create accuracy plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 6. Action Frequencies (bar chart, if actions exist)
    if actions:
        try:
            actions_arr = np.asarray(actions, dtype=float)
            unique, counts = np.unique(actions_arr, return_counts=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(unique, counts, color="skyblue", edgecolor="navy")
            ax.set_xlabel("Action", fontweight="bold")
            ax.set_ylabel("Frequency", fontweight="bold")
            ax.set_title(
                f"RxInfer Action Frequencies - {model_name}", fontweight="bold"
            )
            ax.set_xticks(unique)
            ax.grid(True, alpha=0.3, axis="y")

            viz_file = output_dir / f"{model_name}_rxinfer_action_frequencies.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action frequencies: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action frequencies plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 7. Belief Convergence (max belief probability over time)
    if have_2d_beliefs:
        try:
            max_probabilities = np.max(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(max_probabilities, color="brown", linewidth=2)
            ax.fill_between(
                range(len(max_probabilities)),
                max_probabilities,
                alpha=0.3,
                color="brown",
            )
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Max State Probability", fontweight="bold")
            ax.set_title(
                f"RxInfer Belief Convergence - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_convergence.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief convergence: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief convergence plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 8. Belief Trace (inferred most-likely state over time vs true state)
    if have_2d_beliefs:
        try:
            inferred_states = np.argmax(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(inferred_states, "o-", label="Inferred State", color="darkorange")
            if true_states:
                ax.plot(
                    np.asarray(true_states[: len(inferred_states)], dtype=float),
                    "s--",
                    label="True State",
                    color="steelblue",
                )
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("State Index", fontweight="bold")
            ax.set_title(f"RxInfer Belief Trace - {model_name}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_trace.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief trace: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief trace plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 9. Free Energy (expected / variational free energy over time)
    if free_energy:
        try:
            fe_arr = np.asarray(free_energy, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(fe_arr, color="crimson", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(fe_arr)), fe_arr, alpha=0.3, color="crimson")
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Free Energy", fontweight="bold")
            ax.set_title(f"RxInfer Free Energy - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_free_energy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated free energy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create free energy plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    # 10. Observations (raw observation trace over time)
    if observations:
        try:
            obs_arr = np.asarray(observations, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(obs_arr, "o-", color="teal", linewidth=2, markersize=4)
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Observation", fontweight="bold")
            ax.set_title(f"RxInfer Observations - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_observations.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated observations: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create observations plot: {e}")
            if plt is not None:
                try:
                    plt.close()
                except (OSError, ValueError):
                    pass

    return visualizations


def extract_simulation_data(
    execution_dir: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract RxInfer simulation data from execution outputs.

    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data: dict[str, Any] = {
        "beliefs": [],
        "observations": [],
        "true_states": [],
        "time_steps": 0,
        "model_name": "",
        "framework": "rxinfer",
    }

    try:
        sim_data_dir = execution_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                with open(results_files[0], "r") as f:
                    results = json.load(f)
                data.update(results)

    except Exception as e:
        logger.warning(f"Failed to extract RxInfer data: {e}")

    return data


__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "extract_simulation_data",
]
