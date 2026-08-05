#!/usr/bin/env python3
"""GIF animation generator for RxInfer simulation results.

Produces animated GIF files using matplotlib FuncAnimation showing
belief evolution, state tracking, actions, and VFE convergence over time.

Requires matplotlib with a working backend (Agg for headless).
"""

import colorsys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .animator import (
    _normalize_actions,
    _normalize_beliefs,
    _normalize_observations,
    _normalize_true_states,
    _normalize_vfe,
)

logger = logging.getLogger(__name__)


def generate_gif_animation(
    data: Dict[str, Any],
    output_path: Path,
    model_name: str = "model",
    fps: int = 4,
    dpi: int = 100,
) -> str:
    """Generate an animated GIF from RxInfer simulation results.

    The GIF shows a 2x2 panel:
    - Top-left: Belief bar chart (colors per state, heights = probability)
    - Top-right: True state vs argmax(belief) heatmap
    - Bottom-left: Actions timeline
    - Bottom-right: VFE convergence line

    Args:
        data: Simulation results dict (rxinfer_simulation_v1)
        output_path: Where to write the GIF file
        model_name: Model name for labeling
        fps: Frames per second (default 4)
        dpi: DPI for the figure (default 100)

    Returns:
        Path to the generated GIF file, or "" if no beliefs
    """
    beliefs = _normalize_beliefs(data)
    if not beliefs:
        logger.warning("No beliefs found — cannot generate GIF")
        return ""

    observations = _normalize_observations(data)
    true_states = _normalize_true_states(data)
    actions = _normalize_actions(data)
    vfe = _normalize_vfe(data)

    n_steps = len(beliefs)
    n_states = len(beliefs[0]) if beliefs else 0
    n_actions = max(actions) + 1 if actions else 0

    beliefs_arr = np.array(beliefs)

    # Color palette — convert HSL to RGB for matplotlib
    state_colors = []
    for i in range(n_states):
        hue = i / max(n_states, 1)
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.7)
        state_colors.append((r, g, b))

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"RxInfer Animation — {model_name}", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    # --- Pre-compute VFE range ---
    vfe_arr = np.array(vfe) if vfe else np.array([0.0])
    vfe_min = float(vfe_arr.min()) if len(vfe_arr) > 0 else 0.0
    vfe_max = float(vfe_arr.max()) if len(vfe_arr) > 0 else 1.0
    vfe_range = vfe_max - vfe_min if vfe_max != vfe_min else 1.0

    def animate(frame):
        """Update function for each animation frame."""
        for ax in axes.flat:
            ax.clear()

        step = min(frame, n_steps - 1)

        # --- Top-left: Belief bar chart ---
        ax1 = axes[0, 0]
        ax1.set_facecolor("#16213e")
        belief = beliefs_arr[step]
        x_pos = np.arange(n_states)
        bars = ax1.bar(
            x_pos, belief, color=state_colors, edgecolor="#0f3460", linewidth=0.5
        )
        ax1.set_ylim(0, 1.0)
        ax1.set_xlim(-0.5, n_states - 0.5)
        ax1.set_title("Belief Evolution", color="#00d4ff", fontsize=10)
        ax1.set_xlabel("State", color="#888", fontsize=8)
        ax1.set_ylabel("Probability", color="#888", fontsize=8)
        ax1.tick_params(colors="#888", labelsize=7)
        for j, b in enumerate(bars):
            ax1.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{belief[j]:.2f}",
                ha="center",
                va="bottom",
                color="#ccc",
                fontsize=6,
            )

        # --- Top-right: State heatmap ---
        ax2 = axes[0, 1]
        ax2.set_facecolor("#16213e")
        # Show argmax(belief), true state, and observation up to current step
        if step >= 0:
            argmax_beliefs = np.argmax(beliefs_arr[: step + 1], axis=1)
            ts = (
                np.array(true_states[: step + 1])
                if true_states
                else np.zeros(step + 1, dtype=int)
            )
            obs = (
                np.array(observations[: step + 1])
                if observations
                else np.zeros(step + 1, dtype=int)
            )

            # Build a 3-row heatmap: argmax, true, obs
            heatmap_data = np.zeros((3, step + 1))
            for t in range(step + 1):
                heatmap_data[0, t] = argmax_beliefs[t] / max(n_states - 1, 1)
                heatmap_data[1, t] = ts[t] / max(n_states - 1, 1)
                heatmap_data[2, t] = (
                    obs[t] / max(n_states - 1, 1) if n_states > 1 else 0.5
                )

            ax2.imshow(
                heatmap_data,
                aspect="auto",
                cmap="hsv",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax2.set_yticks([0, 1, 2])
            ax2.set_yticklabels(["Belief", "True", "Obs"], color="#888", fontsize=7)
            ax2.set_title("State Tracking", color="#00d4ff", fontsize=10)
            ax2.set_xlabel("Timestep", color="#888", fontsize=8)
            ax2.tick_params(colors="#888", labelsize=7)

        # --- Bottom-left: Actions timeline ---
        ax3 = axes[1, 0]
        ax3.set_facecolor("#16213e")
        if actions and step < len(actions):
            action_seq = actions[: step + 1]
            for t, a in enumerate(action_seq):
                hue = a / max(n_actions, 1)
                ar, ag, ab = colorsys.hls_to_rgb(hue, 0.45, 0.6)
                color = (ar, ag, ab)
                ax3.bar(t, 1, color=color, edgecolor="#0f3460", linewidth=0.3)
            ax3.set_xlim(-0.5, n_steps - 0.5)
            ax3.set_ylim(0, 1)
            ax3.set_title("Actions", color="#00d4ff", fontsize=10)
            ax3.set_xlabel("Timestep", color="#888", fontsize=8)
            ax3.tick_params(colors="#888", labelsize=7)
            ax3.set_yticks([])

        # --- Bottom-right: VFE convergence ---
        ax4 = axes[1, 1]
        ax4.set_facecolor("#16213e")
        if vfe and len(vfe) > 0:
            # Show VFE up to current step (map step to VFE iteration)
            vfe_step = min(int(step * len(vfe) / max(n_steps, 1)), len(vfe) - 1)
            vfe_x = np.arange(vfe_step + 1)
            vfe_y = vfe_arr[: vfe_step + 1]
            ax4.fill_between(vfe_x, vfe_y, vfe_min, alpha=0.3, color="#ff6b6b")
            ax4.plot(vfe_x, vfe_y, color="#ff6b6b", linewidth=2)
            ax4.scatter(
                [vfe_step],
                [vfe_y[-1] if len(vfe_y) > 0 else 0],
                color="#ff6b6b",
                s=30,
                zorder=5,
            )
            ax4.set_ylim(vfe_min - 0.1 * vfe_range, vfe_max + 0.1 * vfe_range)
        ax4.set_title("VFE Convergence", color="#00d4ff", fontsize=10)
        ax4.set_xlabel("Iteration", color="#888", fontsize=8)
        ax4.tick_params(colors="#888", labelsize=7)

        # Step indicator
        fig.text(
            0.5,
            0.02,
            f"Step {step + 1}/{n_steps}",
            ha="center",
            color="#00d4ff",
            fontsize=10,
            fontweight="bold",
        )

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_steps, interval=1000 // fps, repeat=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)  # type: ignore[arg-type]
    anim.save(str(output_path), writer=writer)  # type: ignore[arg-type]
    plt.close(fig)

    logger.info(f"Generated GIF animation: {output_path}")
    return str(output_path)


__all__: list[Any] = ["generate_gif_animation"]
