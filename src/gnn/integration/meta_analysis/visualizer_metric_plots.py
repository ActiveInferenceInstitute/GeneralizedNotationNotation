#!/usr/bin/env python3
"""
Accuracy/entropy/correlation plot mixins for GNN meta-analysis sweep visualizations.

Extracted from ``integration.meta_analysis.visualizer``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from .collector import SweepRecord
from .visualizer_style import (
    _STYLE,
    _add_watermark,
    _get_color,
    np,
    plt,
)


class SweepMetricPlotMixin:
    """Verbatim plot methods moved from ``SweepVisualizer``."""

    if TYPE_CHECKING:
        output_dir: Path
        logger: logging.Logger

        @staticmethod
        def _safe_log_scale(
            ax: Any, *, x: bool = False, y: bool = False, base: int = 10
        ) -> None: ...
    # ─── Accuracy comparison ───────────────────────────────────────────────

    def _plot_accuracy_comparison(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Final accuracy comparison across sweep cells."""
        acc_records = [
            r
            for r in records
            if r.final_accuracy is not None and r.num_states is not None
        ]
        if not acc_records:
            return None

        fig, ax = plt.subplots(figsize=(16, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        import re as _re

        for fw in frameworks:
            fw_recs = [r for r in acc_records if r.framework == fw]
            if not fw_recs:
                continue
            fw_recs.sort(key=lambda r: (r.num_states or 0, r.num_timesteps or 0))
            # Create concise labels: "N2/T10"
            labels: list[Any] = []
            for r in fw_recs:
                m = _re.search(r"N(\d+).*T(\d+)", r.model_name)
                labels.append(
                    f"N{m.group(1)}/T{m.group(2)}" if m else r.sweep_label[:12]
                )
            accs = [r.final_accuracy for r in fw_recs]
            ax.plot(
                range(len(accs)),
                accs,
                "o-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                alpha=0.8,
                markeredgecolor="white",
            )

        # Mean accuracy line
        all_accs = [
            r.final_accuracy for r in acc_records if r.final_accuracy is not None
        ]
        if all_accs:
            mean_acc = np.mean(all_accs)
            ax.axhline(
                mean_acc,
                color="#457B9D",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                label=f"Mean Accuracy ({mean_acc:.3f})",
            )
            # Stats box
            stats_text = f"μ={mean_acc:.3f}  σ={np.std(all_accs):.3f}  range=[{min(all_accs):.2f}, {max(all_accs):.2f}]"
            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=_STYLE["tick_size"] - 2,
                color="#495057",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#F8F9FA",
                    edgecolor="#DEE2E6",
                    alpha=0.9,
                ),
            )

        # Use abbreviated tick labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            labels,
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"] - 3,
            rotation=55,
            ha="right",
        )
        ax.set_ylabel(
            "Observation Accuracy",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Simulation Accuracy Across Parameter Sweep",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"] - 2)
        ax.set_ylim(0.55, 1.02)
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        _add_watermark(ax)

        if frameworks:
            ax.legend(
                fontsize=_STYLE["legend_size"],
                facecolor="white",
                edgecolor=_STYLE["grid_color"],
                labelcolor="black",
            )

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "accuracy_comparison.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "accuracy_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "Model", "Accuracy"])
            for fw in frameworks:
                fw_recs = [r for r in acc_records if r.framework == fw]
                fw_recs.sort(key=lambda r: (r.num_states or 0, r.num_timesteps or 0))
                for r in fw_recs:
                    writer.writerow([fw, r.sweep_label, f"{r.final_accuracy:.4f}"])

        self.logger.info(
            f"Generated accuracy comparison: {path.name} and {csv_path.name}"
        )
        return path

    # ─── Entropy comparison ────────────────────────────────────────────────

    def _plot_entropy_comparison(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Mean belief entropy comparison."""
        ent_records = [
            r
            for r in records
            if r.mean_belief_entropy is not None and r.num_states is not None
        ]
        if not ent_records:
            return None

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            fw_recs = [r for r in ent_records if r.framework == fw]
            if not fw_recs:
                continue

            ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
            avg_ents: list[Any] = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                avg_ents.append(
                    sum(r.mean_belief_entropy for r in subset) / len(subset)
                )

            ax.plot(
                ns,
                avg_ents,
                "s-",
                color=_get_color(fw),
                label=fw,
                linewidth=_STYLE["line_width"],
                markersize=_STYLE["marker_size"],
                alpha=0.8,
                markeredgecolor="white",
            )

        ax.set_xlabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Mean Belief Entropy (nats)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Belief Entropy vs State Space Size",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")

        if frameworks:
            ax.legend(
                fontsize=_STYLE["legend_size"],
                facecolor="white",
                edgecolor=_STYLE["grid_color"],
                labelcolor="black",
            )

        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "entropy_vs_states.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "entropy_vs_states.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "N", "Mean Belief Entropy (nats)"])
            for fw in frameworks:
                fw_recs = [r for r in ent_records if r.framework == fw]
                ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
                for n in ns:
                    subset = [r for r in fw_recs if r.num_states == n]
                    avg = sum(r.mean_belief_entropy for r in subset) / len(subset)
                    writer.writerow([fw, n, f"{avg:.4f}"])

        self.logger.info(
            f"Generated entropy comparison: {path.name} and {csv_path.name}"
        )
        return path

    # ─── Accuracy Heatmap ──────────────────────────────────────────────────

    def _plot_accuracy_heatmap(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate an N×T accuracy heatmap."""
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework
            and r.num_states is not None
            and r.num_timesteps is not None
        }

        grid = np.full((len(all_n_values), len(all_t_values)), np.nan)
        for (n, t), r in fw_records.items():
            if n in all_n_values and t in all_t_values and r.final_accuracy is not None:
                i = all_n_values.index(n)
                j = all_t_values.index(t)
                grid[i, j] = r.final_accuracy

        valid_count = np.count_nonzero(~np.isnan(grid))
        if valid_count == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        im = ax.imshow(grid, cmap="YlGn", aspect="auto", origin="lower", vmin=0, vmax=1)

        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                if np.isnan(grid[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="#F1F3F5",
                            edgecolor="#DEE2E6",
                            hatch="xxx",
                            linewidth=0.5,
                        )
                    )
                else:
                    val = grid[i, j]
                    text_color = "white" if val > 0.7 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=_STYLE["annotation_size"],
                        fontweight="bold",
                        color=text_color,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Observation Accuracy",
            size=_STYLE["label_size"],
            color=_STYLE["text_color"],
        )
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels(
            [f"{t:,}" for t in all_t_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels(
            [str(n) for n in all_n_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )
        ax.set_xlabel(
            "Timesteps (T)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            f"Simulation Accuracy: {framework}",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            pad=20,
        )

        _add_watermark(ax)

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "heatmaps"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"accuracy_heatmap_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = fw_dir / f"accuracy_heatmap_{framework}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N \\ T"] + [str(t) for t in all_t_values])
            for i, n in enumerate(all_n_values):
                row = [str(n)] + [
                    f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A"
                    for j in range(len(all_t_values))
                ]
                writer.writerow(row)

        self.logger.info(f"Generated accuracy heatmap: {path.name} and {csv_path.name}")
        return path

    # ─── Entropy Heatmap ───────────────────────────────────────────────────

    def _plot_entropy_heatmap(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate an N×T entropy heatmap."""
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework
            and r.num_states is not None
            and r.num_timesteps is not None
        }

        grid = np.full((len(all_n_values), len(all_t_values)), np.nan)
        for (n, t), r in fw_records.items():
            if (
                n in all_n_values
                and t in all_t_values
                and r.mean_belief_entropy is not None
            ):
                i = all_n_values.index(n)
                j = all_t_values.index(t)
                grid[i, j] = r.mean_belief_entropy

        valid_count = np.count_nonzero(~np.isnan(grid))
        if valid_count == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        im = ax.imshow(grid, cmap="Purples", aspect="auto", origin="lower")

        vmax = np.nanmax(grid)

        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                if np.isnan(grid[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="#F1F3F5",
                            edgecolor="#DEE2E6",
                            hatch="xxx",
                            linewidth=0.5,
                        )
                    )
                else:
                    val = grid[i, j]
                    text_color = "white" if val > vmax * 0.7 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=_STYLE["annotation_size"],
                        fontweight="bold",
                        color=text_color,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Belief Entropy (nats)",
            size=_STYLE["label_size"],
            color=_STYLE["text_color"],
        )
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels(
            [f"{t:,}" for t in all_t_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels(
            [str(n) for n in all_n_values],
            color=_STYLE["text_color"],
            fontsize=_STYLE["tick_size"],
        )
        ax.set_xlabel(
            "Timesteps (T)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "State Space Size (N)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            f"Belief Entropy (Certainty): {framework}",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            pad=20,
        )

        _add_watermark(ax)

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "heatmaps"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"entropy_heatmap_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = fw_dir / f"entropy_heatmap_{framework}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N \\ T"] + [str(t) for t in all_t_values])
            for i, n in enumerate(all_n_values):
                row = [str(n)] + [
                    f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A"
                    for j in range(len(all_t_values))
                ]
                writer.writerow(row)

        self.logger.info(f"Generated entropy heatmap: {path.name} and {csv_path.name}")
        return path

    def _plot_accuracy_entropy_correlation(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Scatter plot showing correlation between accuracy and belief entropy."""
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            subset = [
                r
                for r in records
                if r.framework == fw
                and r.final_accuracy is not None
                and r.mean_belief_entropy is not None
            ]
            if not subset:
                continue

            accs = np.array([r.final_accuracy for r in subset])
            ents = np.array([r.mean_belief_entropy for r in subset])
            sizes = [40 + (r.num_states or 2) * 4 for r in subset]

            color = _get_color(fw)
            ax.scatter(
                ents,
                accs,
                s=sizes,
                color=color,
                alpha=0.6,
                label=fw,
                edgecolor="black",
                linewidth=0.5,
            )

            # Annotate each point with N-value
            for r_idx, r in enumerate(subset):
                ax.annotate(
                    f"N={r.num_states}",
                    (ents[r_idx], accs[r_idx]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=_STYLE["tick_size"] - 5,
                    color="#495057",
                    alpha=0.8,
                )

            # Add regression line
            try:
                if len(ents) >= 2:
                    coeffs = np.polyfit(ents, accs, 1)
                    p = np.poly1d(coeffs)
                    x_range = np.linspace(min(ents), max(ents), 100)
                    ax.plot(
                        x_range, p(x_range), "--", color=color, alpha=0.4, linewidth=1.5
                    )

                    # Correlation coefficient
                    r_corr = np.corrcoef(ents, accs)[0, 1]
                    ax.text(
                        max(ents),
                        p(max(ents)),
                        f" r={r_corr:.2f}",
                        color=color,
                        fontsize=_STYLE["tick_size"] - 2,
                        fontweight="bold",
                    )
            except Exception as e:
                self.logger.debug("Skipped entropy/accuracy fit for %s: %s", fw, e)

        ax.set_xlabel(
            "Mean Belief Entropy (nats)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Final Accuracy",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Accuracy vs Belief Certainty Correlation",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.set_ylim(0.55, 1.02)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(
            facecolor="white",
            edgecolor=_STYLE["grid_color"],
            labelcolor="black",
            fontsize=_STYLE["legend_size"],
        )
        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "accuracy_entropy_correlation.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Accuracy vs Timesteps ─────────────────────────────────────────────

    def _plot_accuracy_vs_timesteps(
        self, records: List[SweepRecord], frameworks: List[str]
    ) -> Optional[Path]:
        """Accuracy as a function of timesteps, grouped by N — shows convergence behavior."""
        valid = [
            r
            for r in records
            if r.final_accuracy is not None and r.num_states and r.num_timesteps
        ]
        if len(valid) < 4:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            ns = sorted({r.num_states for r in fw_recs if r.num_states is not None})
            cmap = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(ns)))
            for idx, n in enumerate(ns):
                subset = sorted(
                    [r for r in fw_recs if r.num_states == n],
                    key=lambda r: r.num_timesteps,
                )
                if len(subset) < 2:
                    continue
                ts = [r.num_timesteps for r in subset]
                accs = [r.final_accuracy for r in subset]
                ax.plot(
                    ts,
                    accs,
                    "o-",
                    color=cmap[idx],
                    alpha=0.85,
                    linewidth=2.5,
                    markersize=8,
                    markeredgecolor="white",
                    label=f"N={n}",
                )
        self._safe_log_scale(ax, x=True)
        ax.set_xlabel(
            "Timesteps (T)",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_ylabel(
            "Final Accuracy",
            color=_STYLE["text_color"],
            fontsize=_STYLE["label_size"],
            fontweight="bold",
        )
        ax.set_title(
            "Accuracy Convergence vs Timesteps",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.set_ylim(0.55, 1.02)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=_STYLE["legend_size"] - 2,
                facecolor="white",
                edgecolor=_STYLE["grid_color"],
                labelcolor="black",
                ncol=2,
            )
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "accuracy_vs_timesteps.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated accuracy vs timesteps: {path.name}")
        return path
