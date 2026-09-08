#!/usr/bin/env python3
"""
Belief and GridWorld animation builders for GNN Step 16 analysis visualizations.

Extracted from ``analysis.visualizations``.
"""

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter,
)

from .viz_base import (
    np,
    plt,
    safe_savefig,
)
from .viz_schema import (
    _current_schema_visualization_data,
    _grid_side_for_states,
    _gridworld_state_sequence,
    _is_gridworld_payload,
    _series_from_payload,
    _state_count_from_payload,
)

logger = logging.getLogger(__name__)


def plot_belief_evolution(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief Evolution",
    true_states: Optional[List[int]] = None,
) -> str:
    """
    Plot belief evolution over time.
    """
    plt.figure(figsize=(10, 6))
    belief_array = np.array(beliefs)
    time_steps = range(len(beliefs))

    for i in range(belief_array.shape[1]):
        plt.plot(time_steps, belief_array[:, i], label=f"State {i + 1}")

    if true_states:
        # Normalize true states if they are 1-indexed
        _min_state = min(true_states)
        for t, s in enumerate(true_states):
            plt.scatter(t, 1.05, marker="*", color="black", alpha=0.5 if t > 0 else 0)
            plt.text(t, 1.1, f"S{s}", ha="center", fontsize=8)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)

    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def animate_belief_evolution(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief Evolution Animation",
) -> str:
    """
    Create a GIF animation of belief evolution.
    """
    belief_array = np.array(beliefs)
    n_steps, n_states = belief_array.shape

    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], label=f"State {i + 1}")[0] for i in range(n_states)]

    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def init() -> Any:
        """Provide init behavior."""
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame: int) -> Any:
        """Update operation."""
        for i in range(n_states):
            lines[i].set_data(range(frame + 1), belief_array[: frame + 1, i])
        return lines

    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=5)
    ani.save(output_path, writer=writer)
    plt.close()
    return str(output_path)


def animate_gridworld_trajectory(
    states: list[int],
    output_path: Path,
    title: str,
    state_count: int = 9,
    fps: int = 4,
) -> str:
    """Create a GIF showing a single GridWorld state trajectory."""
    if not states:
        raise ValueError("No states provided for GridWorld animation")

    side = _grid_side_for_states(state_count)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    grid = np.zeros((side, side), dtype=float)
    image = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
    (path_line,) = ax.plot([], [], color="#F39C12", linewidth=2, alpha=0.8)
    (marker,) = ax.plot([], [], "o", color="#E74C3C", markersize=12)
    ax.plot([side - 1], [side - 1], "*", color="#27AE60", markersize=16)

    ax.set_xticks(range(side))
    ax.set_yticks(range(side))
    ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, side - 0.5)
    ax.set_ylim(side - 0.5, -0.5)

    def _coords(sequence: list[int]) -> tuple[list[int], list[int]]:
        """Handle coords for internal callers."""
        cols = [int(state) % side for state in sequence]
        rows = [int(state) // side for state in sequence]
        return cols, rows

    def update(frame: int) -> list[Any]:
        """Update operation."""
        current_states = states[: frame + 1]
        current_state = max(0, min(state_count - 1, current_states[-1]))
        grid.fill(0.0)
        grid[current_state // side, current_state % side] = 1.0
        image.set_data(grid)
        cols, rows = _coords(current_states)
        path_line.set_data(cols, rows)
        marker.set_data([cols[-1]], [rows[-1]])
        ax.set_title(f"{title}\nStep {frame + 1}: state {current_state}")
        return [image, path_line, marker]

    animation = FuncAnimation(fig, update, frames=len(states), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(output_path)


def _gridworld_animation_items(
    framework_data: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Handle gridworld animation items for internal callers."""
    items: list[Dict[str, Any]] = []
    for data in framework_data.values():
        payload = data.get("raw_simulation_data") or data.get("simulation_data", {})
        if not isinstance(payload, dict) or not _is_gridworld_payload(payload):
            continue

        current_data = data.get("simulation_data", {})
        if not isinstance(current_data, dict):
            current_data = _current_schema_visualization_data(payload)
        state_count = _state_count_from_payload(payload)
        states = _gridworld_state_sequence(payload, current_data)
        beliefs = _series_from_payload(
            payload, current_data, "beliefs", "beliefs_by_factor", "joint_state"
        )
        framework = str(data.get("framework", "unknown"))
        model_name = str(data.get("model_name", "unknown"))
        items.append(
            {
                "framework": framework,
                "model_name": model_name,
                "schema_version": payload.get("schema_version"),
                "state_count": state_count,
                "states": states,
                "beliefs": beliefs,
                "source_file": data.get("source_file"),
                "matrix_provenance": payload.get("matrix_provenance", {}),
            }
        )

    framework_order = {"pymdp": 0, "rxinfer": 1, "activeinference_jl": 2}
    return sorted(
        items,
        key=lambda item: (
            framework_order.get(str(item.get("framework")), 99),
            str(item.get("model_name")),
        ),
    )


def animate_cross_framework_gridworld_trajectories(
    items: list[Dict[str, Any]],
    output_path: Path,
    title: str = "GridWorld Cross-Framework Trajectories",
    fps: int = 4,
) -> str:
    """Create one GIF comparing GridWorld trajectories across frameworks."""
    usable_items = [item for item in items if item.get("states")]
    if len(usable_items) < 2:
        raise ValueError("Need at least two framework trajectories")

    state_count = int(usable_items[0].get("state_count") or 9)
    side = _grid_side_for_states(state_count)
    frame_count = max(len(item["states"]) for item in usable_items)

    fig, axes = plt.subplots(
        1,
        len(usable_items),
        figsize=(4.2 * len(usable_items), 4.8),
        squeeze=False,
    )
    flat_axes = list(axes[0])
    artists: list[dict[str, Any]] = []
    for ax, item in zip(flat_axes, usable_items):
        grid = np.zeros((side, side), dtype=float)
        image = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
        (path_line,) = ax.plot([], [], color="#F39C12", linewidth=2, alpha=0.8)
        (marker,) = ax.plot([], [], "o", color="#E74C3C", markersize=12)
        ax.plot([side - 1], [side - 1], "*", color="#27AE60", markersize=16)
        ax.set_title(str(item.get("framework", "unknown")))
        ax.set_xticks(range(side))
        ax.set_yticks(range(side))
        ax.set_xticks(np.arange(-0.5, side, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, side, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, side - 0.5)
        ax.set_ylim(side - 0.5, -0.5)
        artists.append(
            {
                "item": item,
                "grid": grid,
                "image": image,
                "path": path_line,
                "marker": marker,
            }
        )

    fig.suptitle(title)

    def _coords(sequence: list[int]) -> tuple[list[int], list[int]]:
        """Handle coords for internal callers."""
        cols = [int(state) % side for state in sequence]
        rows = [int(state) // side for state in sequence]
        return cols, rows

    def update(frame: int) -> list[Any]:
        """Update operation."""
        changed: list[Any] = []
        for entry in artists:
            states = entry["item"]["states"]
            frame_index = min(frame, len(states) - 1)
            current_states = states[: frame_index + 1]
            current_state = max(0, min(state_count - 1, int(current_states[-1])))
            entry["grid"].fill(0.0)
            entry["grid"][current_state // side, current_state % side] = 1.0
            entry["image"].set_data(entry["grid"])
            cols, rows = _coords(current_states)
            entry["path"].set_data(cols, rows)
            entry["marker"].set_data([cols[-1]], [rows[-1]])
            changed.extend([entry["image"], entry["path"], entry["marker"]])
        fig.suptitle(f"{title} - Step {frame + 1}")
        return changed

    animation = FuncAnimation(fig, update, frames=frame_count, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(output_path)


def generate_gridworld_animation_suite(
    framework_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    logger_instance: Optional[logging.Logger] = None,
) -> List[str]:
    """Generate per-framework and cross-framework GridWorld GIFs."""
    log = logger_instance or logger
    generated_files: List[str] = []
    items = _gridworld_animation_items(framework_data)
    if not items:
        log.debug("No current GridWorld schemas found for animation")
        return generated_files

    animation_dir = output_dir / "gridworld_animations"
    animation_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        framework = str(item["framework"])
        model_name = str(item["model_name"])
        state_count = int(item["state_count"] or 9)
        beliefs = item.get("beliefs", [])
        states = item.get("states", [])

        if beliefs:
            belief_file = (
                animation_dir / f"{model_name}_{framework}_belief_evolution.gif"
            )
            try:
                generated_files.append(
                    animate_belief_evolution(
                        beliefs,
                        belief_file,
                        title=f"Belief Evolution - {model_name} ({framework})",
                    )
                )
                log.info(f"Generated GridWorld belief GIF: {belief_file.name}")
            except Exception as e:
                log.warning(f"Failed to generate belief GIF for {framework}: {e}")

        if states:
            trajectory_file = (
                animation_dir / f"{model_name}_{framework}_state_trajectory.gif"
            )
            try:
                generated_files.append(
                    animate_gridworld_trajectory(
                        states,
                        trajectory_file,
                        title=f"State Trajectory - {model_name} ({framework})",
                        state_count=state_count,
                    )
                )
                log.info(f"Generated GridWorld trajectory GIF: {trajectory_file.name}")
            except Exception as e:
                log.warning(f"Failed to generate trajectory GIF for {framework}: {e}")

    try:
        cross_file = animation_dir / "gridworld_cross_framework_trajectory.gif"
        generated_files.append(
            animate_cross_framework_gridworld_trajectories(items, cross_file)
        )
        log.info(f"Generated GridWorld cross-framework GIF: {cross_file.name}")
    except Exception as e:
        log.warning(f"Failed to generate cross-framework GridWorld GIF: {e}")

    return generated_files
