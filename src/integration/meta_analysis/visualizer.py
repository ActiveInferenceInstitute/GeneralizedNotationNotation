"""
Sweep Visualizer — matplotlib-based plotting for parameter sweep meta-analysis.

Generates publication-quality figures:
- Runtime heatmaps (N × T grids per framework) with proper NaN handling
- Runtime scaling curves (log-log plots)
- Cross-framework comparison bar charts
- Per-step timing comparison
- Simulation metric summaries (accuracy, entropy, EFE convergence)
- Plaintext CSV data export

Fixes and improvements over v1:
- Heatmap grids use the FULL N×T space (from all records), not just the framework's data
- NaN cells shown as hatched gray instead of silently dropped
- Color scale uses log normalization for large dynamic ranges
- CSV plaintext export alongside every visualization
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .collector import SweepRecord

# Deferred matplotlib import to avoid import-time side effects
_MPL_AVAILABLE = False
try:
    import os

    from utils.matplotlib_setup import apply_env_backend_if_set

    apply_env_backend_if_set()
    import matplotlib

    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    _MPL_AVAILABLE = True
except ImportError:
    pass


# Style Constants — Curated for ultra-bold high-contrast scientific reports
_STYLE = {
    "bg_color": "white",
    "axis_bg": "#FFFFFF",
    "text_color": "#212529",
    "grid_color": "#DEE2E6",
    "title_size": 22,
    "label_size": 18,
    "tick_size": 14,
    "line_width": 3.0,
    "marker_size": 10,
    "legend_size": 14,
    "annotation_size": 13,
    "watermark_size": 10,
}

# Color palette — Adjusted for white background visibility
_FRAMEWORK_COLORS = {
    "pymdp": "#E63946",      # Vivid Red
    "jax": "#457B9D",        # Deep Blue-Gray
    "numpyro": "#1D3557",    # Navy
    "rxinfer": "#2A9D8F",    # Teal/Green
    "activeinference_jl": "#B8860B", # Dark Goldenrod (visible on white)
    "discopy": "#9B59B6",    # Amethyst
    "bnlearn": "#D35400",    # Pumpkin
    "pytorch": "#EE4C2C",    # PyTorch Red
}


def _get_color(framework: str) -> str:
    return _FRAMEWORK_COLORS.get(framework, "#AAAAAA")


def _fmt_time(val: float) -> str:
    """Format a runtime value as human-readable string."""
    if val < 0.001:
        return "—"
    if val < 1.0:
        return f"{val*1000:.0f}ms"
    if val < 60:
        return f"{val:.1f}s"
    if val < 3600:
        return f"{val/60:.1f}m"
    return f"{val/3600:.1f}h"


def _add_watermark(ax: plt.Axes):
    """Add a small watermark to the plot for traceability."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    watermark = f"GNN Scaling Analysis | {timestamp} | v1.7.0"
    
    # Use figure-level text to avoid 2D/3D coordinate issues
    ax.get_figure().text(
        0.99, 0.01, watermark,
        color="#CED4DA",
        fontsize=_STYLE["watermark_size"],
        ha="right", va="bottom",
        alpha=0.6
    )


class SweepVisualizer:
    """Generate visualizations from sweep records."""

    def __init__(
        self,
        records: List[SweepRecord],
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
        gnn_format_statistics: Optional[Dict[str, Any]] = None,
    ):
        self.records = records
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self._gnn_format_statistics = gnn_format_statistics

    def generate_all(self) -> List[str]:
        """Generate all available plots. Returns list of generated file paths."""
        if not _MPL_AVAILABLE:
            self.logger.warning("matplotlib not available — skipping visualizations")
            return []

        generated: List[str] = []

        # Filter to records with valid sweep parameters
        sweep_records = [r for r in self.records if r.num_states is not None and r.num_timesteps is not None]

        if not sweep_records:
            self.logger.warning("No sweep-parameterized records found")
            return generated

        # Global N and T values across ALL frameworks
        all_n_values = sorted({r.num_states for r in sweep_records if r.num_states is not None})
        all_t_values = sorted({r.num_timesteps for r in sweep_records if r.num_timesteps is not None})

        # Extract unique frameworks with actual runtime data
        runtime_frameworks = sorted({
            r.framework for r in sweep_records
            if r.execution_time > 0 and r.success
        })

        # 0. Export plaintext CSV of all data
        path = self._export_csv(sweep_records)
        if path:
            generated.append(str(path))

        # 1. Runtime heatmaps per framework (using global grid)
        for fw in runtime_frameworks:
            path = self._plot_runtime_heatmap(sweep_records, fw, all_n_values, all_t_values)
            if path:
                generated.append(str(path))

        # 2. Runtime scaling curves
        path = self._plot_runtime_scaling(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 3. Cross-framework bar chart
        path = self._plot_framework_comparison(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 4. Time-per-step comparison
        path = self._plot_time_per_step(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 5. Accuracy comparison
        path = self._plot_accuracy_comparison(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 6. Belief entropy comparison
        path = self._plot_entropy_comparison(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 7. Accuracy Heatmap
        for fw in runtime_frameworks:
            path = self._plot_accuracy_heatmap(sweep_records, fw, all_n_values, all_t_values)
            if path:
                generated.append(str(path))

        # 8. Entropy Heatmap
        for fw in runtime_frameworks:
            path = self._plot_entropy_heatmap(sweep_records, fw, all_n_values, all_t_values)
            if path:
                generated.append(str(path))

        # 9. 3D Runtime Surface
        for fw in runtime_frameworks:
            path = self._plot_3d_runtime_surface(sweep_records, fw, all_n_values, all_t_values)
            if path:
                generated.append(str(path))

        # 10. Compute Efficiency
        path = self._plot_compute_efficiency(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 11. Resource Scaling (LOC)
        path = self._plot_resource_scaling(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 12. Accuracy-Entropy Correlation
        path = self._plot_accuracy_entropy_correlation(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 13. Inference Throughput vs State Space
        path = self._plot_throughput_vs_n(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 14. Runtime Distribution (violin/box plot per N)
        path = self._plot_runtime_distribution(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 15. Scaling Exponent Summary Bar Chart
        path = self._plot_scaling_exponent_summary(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 16. Code Efficiency (LOC per N² vs N)
        path = self._plot_code_efficiency(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 17. Comprehensive Dashboard (multi-panel summary)
        path = self._plot_comprehensive_dashboard(sweep_records, runtime_frameworks, all_n_values, all_t_values)
        if path:
            generated.append(str(path))

        # 18. Accuracy vs Timesteps (convergence)
        path = self._plot_accuracy_vs_timesteps(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        # 19. Step 3 serialization footprint (when format_statistics.json provided)
        path = self._plot_gnn_serialization_footprint()
        if path:
            generated.append(str(path))

        # 20. Benchmark uncertainty bars (when σ > 0)
        path = self._plot_runtime_uncertainty(sweep_records, runtime_frameworks)
        if path:
            generated.append(str(path))

        return generated

    # ─── Plaintext data export ─────────────────────────────────────────────

    def _export_csv(self, records: List[SweepRecord]) -> Optional[Path]:
        """Export all sweep data as a CSV file for external analysis."""
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        path = data_dir / "sweep_data.csv"
        fieldnames = [
            "model_name", "framework", "num_states", "num_timesteps",
            "execution_time_s", "execution_time_std_s", "execution_benchmark_repeats",
            "time_per_step_ms", "success", "timed_out",
            "lines_of_code", "total_lines",
            "final_accuracy", "mean_belief_entropy",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(records, key=lambda x: (x.framework, x.num_states or 0, x.num_timesteps or 0)):
                writer.writerow({
                    "model_name": r.model_name,
                    "framework": r.framework,
                    "num_states": r.num_states,
                    "num_timesteps": r.num_timesteps,
                    "execution_time_s": f"{r.execution_time:.3f}" if r.execution_time > 0 else "",
                    "execution_time_std_s": (
                        f"{r.execution_time_std:.3f}"
                        if r.execution_time_std is not None and r.execution_time_std > 0
                        else ""
                    ),
                    "execution_benchmark_repeats": r.execution_benchmark_repeats,
                    "time_per_step_ms": f"{r.time_per_step:.4f}" if r.time_per_step else "",
                    "success": r.success,
                    "timed_out": r.timed_out,
                    "lines_of_code": r.lines_of_code or "",
                    "total_lines": r.total_lines or "",
                    "final_accuracy": f"{r.final_accuracy:.4f}" if r.final_accuracy is not None else "",
                    "mean_belief_entropy": f"{r.mean_belief_entropy:.6f}" if r.mean_belief_entropy is not None else "",
                })

        # Also write a human-readable TSV summary
        txt_path = data_dir / "sweep_data.txt"
        with open(txt_path, "w") as f:
            f.write(f"{'Model':<35} {'Framework':<20} {'N':>4} {'T':>8} {'Runtime':>12} {'ms/step':>10} {'Accuracy':>10} {'Entropy':>10}\n")
            f.write("=" * 115 + "\n")
            for r in sorted(records, key=lambda x: (x.framework, x.num_states or 0, x.num_timesteps or 0)):
                rt = _fmt_time(r.execution_time) if r.execution_time > 0 else "—"
                tps = f"{r.time_per_step:.2f}" if r.time_per_step else "—"
                acc = f"{r.final_accuracy:.3f}" if r.final_accuracy is not None else "—"
                ent = f"{r.mean_belief_entropy:.4f}" if r.mean_belief_entropy is not None else "—"
                f.write(f"{r.model_name:<35} {r.framework:<20} {r.num_states or 0:>4} {r.num_timesteps or 0:>8} {rt:>12} {tps:>10} {acc:>10} {ent:>10}\n")

        self.logger.info(f"Exported plaintext data: {path.name}, {txt_path.name}")
        return path

    # ─── Heatmap ───────────────────────────────────────────────────────────

    def _plot_runtime_heatmap(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate an N×T runtime heatmap for a single framework.

        Uses the GLOBAL N and T grid so all heatmaps share the same dimensions.
        NaN cells (missing data) are shown as dark gray with a hatch pattern.
        """
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        # Build grid from all sweep records for this framework
        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework and r.num_states is not None and r.num_timesteps is not None
        }

        grid = np.full((len(all_n_values), len(all_t_values)), np.nan)
        for (n, t), r in fw_records.items():
            if n in all_n_values and t in all_t_values and r.execution_time > 0 and r.success:
                i = all_n_values.index(n)
                j = all_t_values.index(t)
                grid[i, j] = r.execution_time

        # Skip if no data at all
        valid_count = np.count_nonzero(~np.isnan(grid))
        if valid_count == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        # Use log normalization if the dynamic range exceeds 10x
        vmin = np.nanmin(grid)
        vmax = np.nanmax(grid)
        if vmax / max(vmin, 0.001) > 10:
            norm = mcolors.LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
        else:
            norm = None

        # Draw the heatmap — NaN cells will be transparent
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower", norm=norm)

        # Fill NaN cells with a light gray hatched pattern
        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                if np.isnan(grid[i, j]):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#F1F3F5", edgecolor="#DEE2E6",
                        hatch="xxx", linewidth=0.5,
                    ))
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=_STYLE["tick_size"]-2, color="#ADB5BD", fontstyle="italic")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Runtime (s)", size=_STYLE["label_size"], color=_STYLE["text_color"])
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        # Tick labels: use actual N and T values
        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels([f"{t:,}" for t in all_t_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels([str(n) for n in all_n_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])

        ax.set_xlabel("Timesteps (T)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")

        # Enhanced Title with Stats
        avg_runtime = np.nanmean(grid)
        ax.set_title(f"Wall-Clock Runtime: {framework}", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold", pad=30)
        subtitle = f"N ∈ [{min(all_n_values)}, {max(all_n_values)}] · T ∈ [{min(all_t_values):,}, {max(all_t_values):,}] · Mean: {_fmt_time(avg_runtime)} · Range: {_fmt_time(np.nanmin(grid))}–{_fmt_time(np.nanmax(grid))}"
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=_STYLE["tick_size"]-1, color="#495057")
        _add_watermark(ax)
        ax.tick_params(colors=_STYLE["text_color"])

        # Annotate cells with values
        for i in range(len(all_n_values)):
            for j in range(len(all_t_values)):
                val = grid[i, j]
                if not np.isnan(val):
                    label = _fmt_time(val)
                    # Use contrast color based on intensity
                    text_color = "white" if val > vmax * 0.7 else "black"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=_STYLE["annotation_size"], fontweight="bold", color=text_color)

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "heatmaps"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"runtime_heatmap_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = fw_dir / f"runtime_heatmap_{framework}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N \\ T"] + [str(t) for t in all_t_values])
            for i, n in enumerate(all_n_values):
                row = [str(n)] + [f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A" for j in range(len(all_t_values))]
                writer.writerow(row)

        self.logger.info(f"Generated runtime heatmap: {path.name} and {csv_path.name}")
        return path

    # ─── Scaling curves ────────────────────────────────────────────────────

    def _plot_runtime_scaling(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Log-log scaling curves: total runtime vs N for each T tier with power-law fits."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])

        t_values = sorted({r.num_timesteps for r in records if r.num_timesteps is not None})
        n_values = sorted({r.num_states for r in records if r.num_states is not None})

        for ax_idx, (x_param, x_label, group_param, group_label, group_values) in enumerate([
            ("num_states", "State Space Size (N)", "num_timesteps", "T", t_values),
            ("num_timesteps", "Timesteps (T)", "num_states", "N", n_values),
        ]):
            ax = axes[ax_idx]
            ax.set_facecolor(_STYLE["axis_bg"])

            all_exponents = []
            for fw in frameworks:
                fw_records = [r for r in records if r.framework == fw and r.execution_time > 0 and r.success]
                if not fw_records:
                    continue

                for gval in group_values:
                    subset = [r for r in fw_records if getattr(r, group_param) == gval]
                    if len(subset) < 2:
                        continue

                    subset.sort(key=lambda r: getattr(r, x_param) or 0)
                    xs = np.array([getattr(r, x_param) for r in subset], dtype=float)
                    ys = np.array([r.execution_time for r in subset], dtype=float)
                    yerr = np.array(
                        [(getattr(r, "execution_time_std", None) or 0.0) for r in subset],
                        dtype=float,
                    )
                    show_err = bool(np.any(yerr > 0))

                    color = _get_color(fw)

                    if show_err:
                        ax.errorbar(
                            xs,
                            ys,
                            yerr=yerr,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.2,
                            capsize=4,
                            alpha=0.55,
                            zorder=2,
                        )

                    ax.plot(
                        xs,
                        ys,
                        "o",
                        color=color,
                        alpha=0.7,
                        markersize=_STYLE["marker_size"],
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        label=f"{fw} ({group_label}={gval})",
                        zorder=3,
                    )

                    # Fit power law: y = a * x^b => log(y) = log(a) + b*log(x)
                    try:
                        valid = (xs > 0) & (ys > 0)
                        if np.sum(valid) >= 2:
                            log_x = np.log(xs[valid])
                            log_y = np.log(ys[valid])
                            
                            # Log-linear fit
                            coeffs, residuals, rank, singular_values, rcond = np.polyfit(log_x, log_y, 1, full=True)
                            exponent = coeffs[0]
                            all_exponents.append(exponent)
                            
                            # R^2 calculation
                            y_mean = np.mean(log_y)
                            ss_tot = np.sum((log_y - y_mean)**2)
                            ss_res = residuals[0] if len(residuals) > 0 else 0
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
                            
                            # RMSE calculation
                            y_pred = coeffs[0] * log_x + coeffs[1]
                            rmse = np.sqrt(np.mean((log_y - y_pred)**2))
                            
                            # Label with exponents and metrics
                            label = f"{fw} ({group_label}={gval}): α={exponent:.2f} (R²={r_squared:.3f}, RMSE={rmse:.3f})"
                            
                            fit_fn = np.exp(coeffs[1]) * (xs**exponent)
                            ax.plot(xs, fit_fn, "-", color=color, alpha=0.6, linewidth=_STYLE["line_width"], label=label)
                            
                            # Annotate exponent near the end of the line
                            ax.text(xs[-1], ys[-1], f" α={exponent:.2f}", 
                                    color=color, fontsize=_STYLE["tick_size"]-4, fontweight="bold", va="center")
                    except Exception:
                        pass

            ax.set_xlabel(x_label, color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
            ax.set_ylabel("Wall-clock Runtime (s)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
            ax.set_title(f"Runtime Scaling vs {x_label.split('(')[0].strip()}", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
            ax.grid(True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
            _add_watermark(ax)

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize=_STYLE["legend_size"],
                          facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black",
                          loc="best", framealpha=0.9, title="Framework (Params)")
        
        # Global Title
        avg_exp = np.mean(all_exponents) if all_exponents else 0
        fig.suptitle(f"Empirical Scaling Analysis (Avg Exponent α={avg_exp:.2f})", 
                     color=_STYLE["text_color"], fontsize=_STYLE["title_size"]+2, fontweight="bold", y=1.02)
        fig.tight_layout()
        
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_scaling_curves.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "runtime_scaling_curves.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "N", "T", "Runtime (s)", "Runtime_std (s)"])
            for fw in frameworks:
                fw_records = [r for r in records if r.framework == fw and r.execution_time > 0 and r.success]
                for r in sorted(fw_records, key=lambda x: (x.num_states or 0, x.num_timesteps or 0)):
                    std_val = ""
                    if r.execution_time_std is not None and r.execution_time_std > 0:
                        std_val = f"{r.execution_time_std:.4f}"
                    writer.writerow([fw, r.num_states, r.num_timesteps, f"{r.execution_time:.4f}", std_val])

        self.logger.info(f"Generated scaling curves: {path.name} and {csv_path.name}")
        return path

    # ─── Framework comparison ──────────────────────────────────────────────

    def _plot_framework_comparison(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Grouped bar chart: runtime per model, grouped by framework."""
        models = sorted({r.model_name for r in records if r.num_states is not None})
        if not models or not frameworks:
            return None

        fig, ax = plt.subplots(figsize=(max(14, len(models) * 1.2), 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        x = np.arange(len(models))
        bar_width = 0.8 / max(len(frameworks), 1)

        all_times = []
        for i, fw in enumerate(frameworks):
            times = []
            for model in models:
                match = [r for r in records if r.model_name == model and r.framework == fw and r.success and r.execution_time > 0]
                times.append(match[0].execution_time if match else 0)
            
            all_times.extend([t for t in times if t > 0])
            offset = (i - len(frameworks) / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, times, bar_width * 0.9,
                          color=_get_color(fw), alpha=0.8, label=fw, edgecolor="black", linewidth=0.5)

            # Annotate non-zero bars
            for bar, t in zip(bars, times):
                if t > 0:
                    label = _fmt_time(t)
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                            label, ha="center", va="bottom", fontsize=_STYLE["tick_size"]-4, 
                            color=_STYLE["text_color"], fontweight="bold", rotation=45)

        # Global Median Line
        if all_times:
            median_val = np.median(all_times)
            ax.axhline(median_val, color="#E63946", linestyle="--", alpha=0.6, linewidth=1.5, label=f"Global Median ({_fmt_time(median_val)})")

        # Clean up model labels
        import re as _re
        display_labels = []
        for m in models:
            match = _re.search(r"N(\d+).*T(\d+)", m)
            if match:
                display_labels.append(f"N={match.group(1)}\nT={match.group(2)}")
            else:
                display_labels.append(m[:20])

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, color=_STYLE["text_color"], fontsize=_STYLE["tick_size"]-2)
        ax.set_ylabel("Runtime (s)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Cross-Framework Runtime Comparison", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        
        ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black",
                  loc="upper left", ncol=min(len(frameworks)+1, 4), framealpha=0.9)

        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "framework_runtime_comparison.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "framework_runtime_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + frameworks)
            for model in models:
                row = [model]
                for fw in frameworks:
                    match = [r for r in records if r.model_name == model and r.framework == fw and r.success and r.execution_time > 0]
                    row.append(f"{match[0].execution_time:.4f}" if match else "N/A")
                writer.writerow(row)

        self.logger.info(f"Generated framework comparison: {path.name} and {csv_path.name}")
        return path

    # ─── Time per step ─────────────────────────────────────────────────────

    def _plot_time_per_step(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Per-step timing comparison grouped by framework and N."""
        # Build grouped data: for each (framework, N), compute median ms/step
        groups = []
        for fw in frameworks:
            fw_records = [r for r in records if r.framework == fw and r.success and r.time_per_step is not None and r.time_per_step > 0]
            if not fw_records:
                continue
            n_values = sorted({r.num_states for r in fw_records if r.num_states is not None})
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                times = [r.time_per_step for r in subset if r.time_per_step is not None]
                if times:
                    median_t = sorted(times)[len(times) // 2]
                    groups.append((fw, n, median_t))

        if not groups:
            return None

        fig, ax = plt.subplots(figsize=(max(12, len(groups) * 0.8), 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        labels = [f"N={n}\n{fw}" for fw, n, _ in groups]
        values = [t for _, _, t in groups]
        colors = [_get_color(fw) for fw, _, _ in groups]

        bars = ax.bar(range(len(groups)), values, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)

        # Annotate bars — place inside bar for tall values to avoid title collision
        for bar, val in zip(bars, values):
            if val > 300:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.85,
                        f"{val:.1f}", ha="center", va="top", fontsize=_STYLE["tick_size"]-3, 
                        color="white", fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=_STYLE["tick_size"]-3, 
                        color=_STYLE["text_color"], fontweight="bold")

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, color=_STYLE["text_color"], fontsize=_STYLE["tick_size"]-3, rotation=45, ha="right")
        ax.set_ylabel("Time per Timestep (ms)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Per-Timestep Execution Cost", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold", pad=15)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"]-2)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")

        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "time_per_step.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        csv_path = cf_dir / "time_per_step.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Framework", "N", "Median Time per Step (ms)"])
            for fw, n, t in groups:
                writer.writerow([fw, n, f"{t:.4f}"])

        self.logger.info(f"Generated time-per-step: {path.name} and {csv_path.name}")
        return path

    # ─── Accuracy comparison ───────────────────────────────────────────────

    def _plot_accuracy_comparison(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Final accuracy comparison across sweep cells."""
        acc_records = [r for r in records if r.final_accuracy is not None and r.num_states is not None]
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
            labels = []
            for r in fw_recs:
                m = _re.search(r"N(\d+).*T(\d+)", r.model_name)
                labels.append(f"N{m.group(1)}/T{m.group(2)}" if m else r.sweep_label[:12])
            accs = [r.final_accuracy for r in fw_recs]
            ax.plot(range(len(accs)), accs, "o-", color=_get_color(fw), label=fw, 
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], 
                    alpha=0.8, markeredgecolor="white")

        # Mean accuracy line
        all_accs = [r.final_accuracy for r in acc_records if r.final_accuracy is not None]
        if all_accs:
            mean_acc = np.mean(all_accs)
            ax.axhline(mean_acc, color="#457B9D", linestyle="--", alpha=0.6, linewidth=1.5,
                        label=f"Mean Accuracy ({mean_acc:.3f})")
            # Stats box
            stats_text = f"μ={mean_acc:.3f}  σ={np.std(all_accs):.3f}  range=[{min(all_accs):.2f}, {max(all_accs):.2f}]"
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=_STYLE["tick_size"]-2,
                    color="#495057", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DEE2E6", alpha=0.9))

        # Use abbreviated tick labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, color=_STYLE["text_color"], fontsize=_STYLE["tick_size"]-3, rotation=55, ha="right")
        ax.set_ylabel("Observation Accuracy", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Simulation Accuracy Across Parameter Sweep", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"]-2)
        ax.set_ylim(0.55, 1.02)
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        _add_watermark(ax)
        
        if frameworks:
            ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black")

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

        self.logger.info(f"Generated accuracy comparison: {path.name} and {csv_path.name}")
        return path

    # ─── Entropy comparison ────────────────────────────────────────────────

    def _plot_entropy_comparison(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Mean belief entropy comparison."""
        ent_records = [r for r in records if r.mean_belief_entropy is not None and r.num_states is not None]
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
            avg_ents = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                avg_ents.append(sum(r.mean_belief_entropy for r in subset) / len(subset))

            ax.plot(ns, avg_ents, "s-", color=_get_color(fw), label=fw, 
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], 
                    alpha=0.8, markeredgecolor="white")

        ax.set_xlabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Mean Belief Entropy (nats)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Belief Entropy vs State Space Size", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        
        if frameworks:
            ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black")

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

        self.logger.info(f"Generated entropy comparison: {path.name} and {csv_path.name}")
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
            if r.framework == framework and r.num_states is not None and r.num_timesteps is not None
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
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#F1F3F5", edgecolor="#DEE2E6",
                        hatch="xxx", linewidth=0.5,
                    ))
                else:
                    val = grid[i, j]
                    text_color = "white" if val > 0.7 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=_STYLE["annotation_size"], fontweight="bold", color=text_color)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Observation Accuracy", size=_STYLE["label_size"], color=_STYLE["text_color"])
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels([f"{t:,}" for t in all_t_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels([str(n) for n in all_n_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])
        ax.set_xlabel("Timesteps (T)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title(f"Simulation Accuracy: {framework}", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold", pad=20)
        
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
                row = [str(n)] + [f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A" for j in range(len(all_t_values))]
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
            if r.framework == framework and r.num_states is not None and r.num_timesteps is not None
        }

        grid = np.full((len(all_n_values), len(all_t_values)), np.nan)
        for (n, t), r in fw_records.items():
            if n in all_n_values and t in all_t_values and r.mean_belief_entropy is not None:
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
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=True, facecolor="#F1F3F5", edgecolor="#DEE2E6",
                        hatch="xxx", linewidth=0.5,
                    ))
                else:
                    val = grid[i, j]
                    text_color = "white" if val > vmax * 0.7 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=_STYLE["annotation_size"], fontweight="bold", color=text_color)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Belief Entropy (nats)", size=_STYLE["label_size"], color=_STYLE["text_color"])
        cbar.ax.tick_params(labelsize=_STYLE["tick_size"], colors=_STYLE["text_color"])

        ax.set_xticks(range(len(all_t_values)))
        ax.set_xticklabels([f"{t:,}" for t in all_t_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])
        ax.set_yticks(range(len(all_n_values)))
        ax.set_yticklabels([str(n) for n in all_n_values], color=_STYLE["text_color"], fontsize=_STYLE["tick_size"])
        ax.set_xlabel("Timesteps (T)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title(f"Belief Entropy (Certainty): {framework}", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold", pad=20)
        
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
                row = [str(n)] + [f"{grid[i, j]:.4f}" if not np.isnan(grid[i, j]) else "N/A" for j in range(len(all_t_values))]
                writer.writerow(row)

        self.logger.info(f"Generated entropy heatmap: {path.name} and {csv_path.name}")
        return path

    # ─── New Visualizations ────────────────────────────────────────────────

    def _plot_3d_runtime_surface(
        self,
        records: List[SweepRecord],
        framework: str,
        all_n_values: List[int],
        all_t_values: List[int],
    ) -> Optional[Path]:
        """Generate a 3D surface plot of runtime vs N and T."""
        if len(all_n_values) < 2 or len(all_t_values) < 2:
            return None

        # Build grid
        fw_records = {
            (r.num_states, r.num_timesteps): r
            for r in records
            if r.framework == framework and r.num_states is not None and r.num_timesteps is not None
        }

        X, Y = np.meshgrid(np.log10(all_t_values), np.log10(all_n_values))
        Z = np.full(X.shape, np.nan)
        
        for i, n in enumerate(all_n_values):
            for j, t in enumerate(all_t_values):
                r = fw_records.get((n, t))
                if r and r.execution_time > 0 and r.success:
                    Z[i, j] = np.log10(r.execution_time)

        if np.all(np.isnan(Z)):
            return None

        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(_STYLE["bg_color"])
        
        # Mask NaN for surface plot
        Zm = np.ma.masked_invalid(Z)
        
        surf = ax.plot_surface(X, Y, Zm, cmap="viridis", edgecolor='black', linewidth=0.1, alpha=0.8)
        
        ax.set_xlabel("log10(Timesteps T)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], labelpad=10)
        ax.set_ylabel("log10(State Size N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], labelpad=10)
        ax.set_zlabel("log10(Runtime [s])", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], labelpad=10)
        ax.set_title(f"3D Runtime Response Surface: {framework}", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold", pad=20)
        
        _add_watermark(ax)

        # Style the panes for white background
        ax.xaxis.pane.set_facecolor("#F8F9FA")
        ax.yaxis.pane.set_facecolor("#F8F9FA")
        ax.zaxis.pane.set_facecolor("#F8F9FA")
        ax.xaxis.pane.set_edgecolor("#DEE2E6")
        ax.yaxis.pane.set_edgecolor("#DEE2E6")
        ax.zaxis.pane.set_edgecolor("#DEE2E6")
        
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.label.set_color(_STYLE["text_color"])
            axis.set_tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"]-2)

        fig.tight_layout()
        fw_dir = self.output_dir / framework / "surfaces"
        fw_dir.mkdir(parents=True, exist_ok=True)
        path = fw_dir / f"runtime_surface_3d_{framework}.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_compute_efficiency(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Plot 'Compute Efficiency' (N^3 * T / runtime) to show scaling bottlenecks."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            fw_records = [r for r in records if r.framework == fw and r.success and r.execution_time > 0]
            if not fw_records:
                continue
            
            # Group by N, take average across T
            n_values = sorted({r.num_states for r in fw_records if r.num_states is not None})
            efficiencies = []
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                # Approximation of ops: N^3 * T
                scores = [( (r.num_states**3) * (r.num_timesteps or 1) ) / r.execution_time for r in subset]
                efficiencies.append(np.mean(scores))
            
            # Normalize to max efficiency across all frameworks
            ax.plot(n_values, efficiencies, "o-", color=_get_color(fw), label=fw, 
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], markeredgecolor="white")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Efficiency Score (Ops/sec proxy)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Compute Efficiency Benchmark", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        
        _add_watermark(ax)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black", fontsize=_STYLE["legend_size"])

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "compute_efficiency.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_resource_scaling(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Plot lines of code (LOC) vs N."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            fw_records = [r for r in records if r.framework == fw and r.lines_of_code is not None]
            if not fw_records:
                continue
            
            # Group by N (T doesn't affect LOC usually)
            n_values = sorted({r.num_states for r in fw_records})
            loc_values = []
            for n in n_values:
                subset = [r for r in fw_records if r.num_states == n]
                loc_values.append(np.mean([r.lines_of_code for r in subset]))
            
            ax.plot(n_values, loc_values, "D-", color=_get_color(fw), label=fw, 
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], alpha=0.8, markeredgecolor="white")

            # Power-law fit for LOC scaling
            try:
                if len(n_values) >= 3:
                    log_n = np.log(np.array(n_values, dtype=float))
                    log_loc = np.log(np.array(loc_values, dtype=float))
                    coeffs = np.polyfit(log_n, log_loc, 1)
                    exponent = coeffs[0]
                    fit_vals = np.exp(coeffs[1]) * np.array(n_values, dtype=float)**exponent
                    ax.plot(n_values, fit_vals, "--", color=_get_color(fw), alpha=0.5, linewidth=1.5,
                            label=f"{fw} fit: O(N^{exponent:.2f})")
            except Exception:
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Lines of Code (Rendered)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Generated Code Complexity Scaling", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black", fontsize=_STYLE["legend_size"])
        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "resource_scaling_loc.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_accuracy_entropy_correlation(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Scatter plot showing correlation between accuracy and belief entropy."""
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])

        for fw in frameworks:
            subset = [r for r in records if r.framework == fw and r.final_accuracy is not None and r.mean_belief_entropy is not None]
            if not subset:
                continue
            
            accs = np.array([r.final_accuracy for r in subset])
            ents = np.array([r.mean_belief_entropy for r in subset])
            sizes = [40 + (r.num_states or 2) * 4 for r in subset]
            
            color = _get_color(fw)
            ax.scatter(ents, accs, s=sizes, color=color, alpha=0.6, label=fw, edgecolor="black", linewidth=0.5)

            # Annotate each point with N-value
            for r_idx, r in enumerate(subset):
                ax.annotate(f"N={r.num_states}", (ents[r_idx], accs[r_idx]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=_STYLE["tick_size"]-5, color="#495057", alpha=0.8)

            # Add regression line
            try:
                if len(ents) >= 2:
                    coeffs = np.polyfit(ents, accs, 1)
                    p = np.poly1d(coeffs)
                    x_range = np.linspace(min(ents), max(ents), 100)
                    ax.plot(x_range, p(x_range), "--", color=color, alpha=0.4, linewidth=1.5)
                    
                    # Correlation coefficient
                    r_corr = np.corrcoef(ents, accs)[0, 1]
                    ax.text(max(ents), p(max(ents)), f" r={r_corr:.2f}", color=color, 
                            fontsize=_STYLE["tick_size"]-2, fontweight="bold")
            except Exception:
                pass

        ax.set_xlabel("Mean Belief Entropy (nats)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Final Accuracy", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Accuracy vs Belief Certainty Correlation", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.set_ylim(0.55, 1.02)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black", fontsize=_STYLE["legend_size"])
        _add_watermark(ax)

        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "accuracy_entropy_correlation.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Throughput vs N ───────────────────────────────────────────────────

    def _plot_throughput_vs_n(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Inference throughput (timesteps/second) vs state space size."""
        valid = [r for r in records if r.execution_time > 0 and r.success and r.num_states and r.num_timesteps]
        if not valid:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"]); ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            if not fw_recs: continue
            ns = sorted({r.num_states for r in fw_recs})
            throughputs = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                throughputs.append(np.mean([r.num_timesteps / r.execution_time for r in subset]))
            ax.plot(ns, throughputs, "o-", color=_get_color(fw), label=fw,
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], alpha=0.8, markeredgecolor="white")
            for n, tp in zip(ns, throughputs):
                ax.annotate(f"{tp:.0f}", (n, tp), textcoords="offset points", xytext=(0, 10),
                           ha="center", fontsize=_STYLE["tick_size"]-3, color=_STYLE["text_color"], fontweight="bold")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Throughput (timesteps/s)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Inference Throughput vs Model Complexity", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, which="both", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black")
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "throughput_vs_n.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        csv_path = cf_dir / "throughput_vs_n.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Framework", "N", "Avg_Throughput_timesteps_per_s"])
            for fw in frameworks:
                fw_recs = [r for r in valid if r.framework == fw]
                for n in sorted({r.num_states for r in fw_recs}):
                    subset = [r for r in fw_recs if r.num_states == n]
                    w.writerow([fw, n, f"{np.mean([r.num_timesteps / r.execution_time for r in subset]):.2f}"])
        self.logger.info(f"Generated throughput plot: {path.name}")
        return path

    # ─── Runtime Distribution ──────────────────────────────────────────────

    def _plot_runtime_distribution(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Box plot of runtime distribution for each N value across all T."""
        valid = [r for r in records if r.execution_time > 0 and r.success and r.num_states]
        if len(valid) < 4:
            return None
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"]); ax.set_facecolor(_STYLE["axis_bg"])
        ns = sorted({r.num_states for r in valid})
        data_by_n = []; labels = []
        for n in ns:
            runtimes = [r.execution_time for r in valid if r.num_states == n]
            if runtimes:
                data_by_n.append(runtimes); labels.append(f"N={n}")
        if not data_by_n:
            plt.close(fig); return None
        bp = ax.boxplot(data_by_n, patch_artist=True, labels=labels, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        base_color = _get_color(frameworks[0]) if frameworks else "#E63946"
        for patch in bp["boxes"]:
            patch.set_facecolor(base_color); patch.set_alpha(0.7)
        for i, d in enumerate(data_by_n):
            x = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x, d, alpha=0.5, s=30, color="black", zorder=3, edgecolors="white", linewidths=0.5)
        ax.set_ylabel("Wall-clock Runtime (s)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_xlabel("State Space Size", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Runtime Distribution by State Space Size", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        medians = [np.median(d) for d in data_by_n]
        stats = f"Median range: {_fmt_time(min(medians))}–{_fmt_time(max(medians))}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=_STYLE["tick_size"]-2, va="top",
                color="#495057", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#DEE2E6", alpha=0.9))
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_distribution.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated runtime distribution: {path.name}")
        return path

    # ─── Scaling Exponent Summary ──────────────────────────────────────────

    def _plot_scaling_exponent_summary(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Bar chart of empirical scaling exponents by parameter grouping."""
        valid = [r for r in records if r.execution_time > 0 and r.success and r.num_states and r.num_timesteps]
        if len(valid) < 6:
            return None
        t_values = sorted({r.num_timesteps for r in valid})
        n_values = sorted({r.num_states for r in valid})
        exponents = []  # (label, exponent, r_squared)
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            for t in t_values:
                subset = sorted([r for r in fw_recs if r.num_timesteps == t], key=lambda r: r.num_states)
                if len(subset) < 3: continue
                xs, ys = np.array([r.num_states for r in subset], dtype=float), np.array([r.execution_time for r in subset], dtype=float)
                try:
                    lx, ly = np.log(xs), np.log(ys)
                    c = np.polyfit(lx, ly, 1)
                    ss_res = np.sum((ly - (c[0]*lx + c[1]))**2)
                    ss_tot = np.sum((ly - np.mean(ly))**2)
                    exponents.append((f"N-scaling\n(T={t})", c[0], 1 - ss_res/ss_tot if ss_tot > 0 else 0))
                except Exception: pass
            for n in n_values:
                subset = sorted([r for r in fw_recs if r.num_states == n], key=lambda r: r.num_timesteps)
                if len(subset) < 3: continue
                xs, ys = np.array([r.num_timesteps for r in subset], dtype=float), np.array([r.execution_time for r in subset], dtype=float)
                try:
                    lx, ly = np.log(xs), np.log(ys)
                    c = np.polyfit(lx, ly, 1)
                    ss_res = np.sum((ly - (c[0]*lx + c[1]))**2)
                    ss_tot = np.sum((ly - np.mean(ly))**2)
                    exponents.append((f"T-scaling\n(N={n})", c[0], 1 - ss_res/ss_tot if ss_tot > 0 else 0))
                except Exception: pass
        if not exponents:
            return None
        fig, ax = plt.subplots(figsize=(max(14, len(exponents) * 1.2), 8))
        fig.patch.set_facecolor(_STYLE["bg_color"]); ax.set_facecolor(_STYLE["axis_bg"])
        labels = [e[0] for e in exponents]; vals = [e[1] for e in exponents]; r2s = [e[2] for e in exponents]
        colors = ["#E63946" if "N-scaling" in l else "#457B9D" for l in labels]
        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        for bar, v, r2 in zip(bars, vals, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"α={v:.2f}\nR²={r2:.2f}", ha="center", va="bottom", fontsize=_STYLE["tick_size"]-3, fontweight="bold", color=_STYLE["text_color"])
        ax.axhline(1.0, color="#ADB5BD", linestyle="--", linewidth=1, alpha=0.7, label="Linear scaling (α=1)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=_STYLE["tick_size"]-3, color=_STYLE["text_color"])
        ax.set_ylabel("Scaling Exponent (α)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Empirical Scaling Exponents by Parameter Group", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(axis="y", alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black")
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "scaling_exponent_summary.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated scaling exponent summary: {path.name}")
        return path

    # ─── Code Efficiency ───────────────────────────────────────────────────

    def _plot_code_efficiency(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """LOC per state-squared (LOC/N²) vs N — measures code generation efficiency."""
        valid = [r for r in records if r.lines_of_code and r.num_states and r.lines_of_code > 0]
        if len(valid) < 3:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"]); ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            if not fw_recs: continue
            ns = sorted({r.num_states for r in fw_recs})
            efficiencies = []
            for n in ns:
                subset = [r for r in fw_recs if r.num_states == n]
                avg_loc = np.mean([r.lines_of_code for r in subset])
                efficiencies.append(avg_loc / (n * n))
            ax.plot(ns, efficiencies, "s-", color=_get_color(fw), label=fw,
                    linewidth=_STYLE["line_width"], markersize=_STYLE["marker_size"], alpha=0.8, markeredgecolor="white")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("State Space Size (N)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("LOC / N²", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Code Generation Efficiency (LOC per State²)", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        ax.legend(fontsize=_STYLE["legend_size"], facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black")
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "scaling"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "code_efficiency.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated code efficiency: {path.name}")
        return path

    # ─── Comprehensive Dashboard ───────────────────────────────────────────

    def _plot_comprehensive_dashboard(self, records: List[SweepRecord], frameworks: List[str],
                                       all_n: List[int], all_t: List[int]) -> Optional[Path]:
        """Multi-panel summary dashboard combining key metrics."""
        valid = [r for r in records if r.execution_time > 0 and r.success]
        if len(valid) < 4:
            return None
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        fig.suptitle("PyMDP Scaling Analysis — Executive Dashboard", fontsize=_STYLE["title_size"]+4,
                     fontweight="bold", color=_STYLE["text_color"], y=1.01)

        # Panel 1: Runtime vs N (median across T)
        ax = axes[0, 0]; ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            ns = sorted({r.num_states for r in fw_recs if r.num_states})
            meds = [np.median([r.execution_time for r in fw_recs if r.num_states == n]) for n in ns]
            ax.plot(ns, meds, "o-", color=_get_color(fw), label=fw, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title("Median Runtime vs N", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"])
        ax.set_xlabel("N"); ax.set_ylabel("Runtime (s)")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 2: Throughput vs N
        ax = axes[0, 1]; ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw and r.num_states and r.num_timesteps]
            ns = sorted({r.num_states for r in fw_recs})
            tps = [np.mean([r.num_timesteps/r.execution_time for r in fw_recs if r.num_states == n]) for n in ns]
            ax.plot(ns, tps, "o-", color=_get_color(fw), label=fw, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title("Throughput vs N", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"])
        ax.set_xlabel("N"); ax.set_ylabel("Timesteps/s")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 3: Accuracy vs N
        ax = axes[0, 2]; ax.set_facecolor(_STYLE["axis_bg"])
        acc_recs = [r for r in valid if r.final_accuracy is not None and r.num_states]
        for fw in frameworks:
            fw_a = [r for r in acc_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_a})
            accs = [np.mean([r.final_accuracy for r in fw_a if r.num_states == n]) for n in ns]
            ax.plot(ns, accs, "o-", color=_get_color(fw), label=fw, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.set_xscale("log", base=2); ax.set_ylim(0.55, 1.02)
        ax.set_title("Accuracy vs N", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"])
        ax.set_xlabel("N"); ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 4: LOC vs N
        ax = axes[1, 0]; ax.set_facecolor(_STYLE["axis_bg"])
        loc_recs = [r for r in records if r.lines_of_code and r.num_states]
        for fw in frameworks:
            fw_l = [r for r in loc_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_l})
            locs = [np.mean([r.lines_of_code for r in fw_l if r.num_states == n]) for n in ns]
            ax.plot(ns, locs, "D-", color=_get_color(fw), label=fw, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title("Generated LOC vs N", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"])
        ax.set_xlabel("N"); ax.set_ylabel("Lines of Code")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 5: Entropy vs N
        ax = axes[1, 1]; ax.set_facecolor(_STYLE["axis_bg"])
        ent_recs = [r for r in valid if r.mean_belief_entropy is not None and r.num_states]
        for fw in frameworks:
            fw_e = [r for r in ent_recs if r.framework == fw]
            ns = sorted({r.num_states for r in fw_e})
            ents = [np.mean([r.mean_belief_entropy for r in fw_e if r.num_states == n]) for n in ns]
            ax.plot(ns, ents, "s-", color=_get_color(fw), label=fw, linewidth=2.5, markersize=8, markeredgecolor="white")
        ax.set_xscale("log", base=2)
        ax.set_title("Belief Entropy vs N", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"])
        ax.set_xlabel("N"); ax.set_ylabel("Entropy (nats)")
        ax.grid(True, alpha=0.3, color=_STYLE["grid_color"], linestyle="--")

        # Panel 6: Summary stats table
        ax = axes[1, 2]; ax.set_facecolor(_STYLE["axis_bg"]); ax.axis("off")
        total_models = len(valid)
        total_runtime = sum(r.execution_time for r in valid)
        avg_acc = np.mean([r.final_accuracy for r in valid if r.final_accuracy is not None]) if any(r.final_accuracy for r in valid) else 0
        n_range = f"[{min(all_n)}, {max(all_n)}]"
        t_range = f"[{min(all_t):,}, {max(all_t):,}]"
        table_data = [
            ["Total Models", str(total_models)],
            ["N Range", n_range],
            ["T Range", t_range],
            ["Total Runtime", _fmt_time(total_runtime)],
            ["Mean Accuracy", f"{avg_acc:.3f}"],
            ["Frameworks", ", ".join(frameworks)],
        ]
        table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center",
                         cellLoc="left", colWidths=[0.5, 0.5])
        table.auto_set_font_size(False); table.set_fontsize(_STYLE["tick_size"])
        table.scale(1, 2)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor(_STYLE["grid_color"])
            if key[0] == 0:
                cell.set_facecolor("#E63946"); cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F8F9FA" if key[0] % 2 else "white")
        ax.set_title("Summary Statistics", fontsize=_STYLE["label_size"], fontweight="bold", color=_STYLE["text_color"], pad=20)

        _add_watermark(axes[0, 0])
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "comprehensive_dashboard.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated comprehensive dashboard: {path.name}")
        return path

    # ─── Accuracy vs Timesteps ─────────────────────────────────────────────

    def _plot_accuracy_vs_timesteps(self, records: List[SweepRecord], frameworks: List[str]) -> Optional[Path]:
        """Accuracy as a function of timesteps, grouped by N — shows convergence behavior."""
        valid = [r for r in records if r.final_accuracy is not None and r.num_states and r.num_timesteps]
        if len(valid) < 4:
            return None
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(_STYLE["bg_color"]); ax.set_facecolor(_STYLE["axis_bg"])
        for fw in frameworks:
            fw_recs = [r for r in valid if r.framework == fw]
            ns = sorted({r.num_states for r in fw_recs})
            cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ns)))
            for idx, n in enumerate(ns):
                subset = sorted([r for r in fw_recs if r.num_states == n], key=lambda r: r.num_timesteps)
                if len(subset) < 2: continue
                ts = [r.num_timesteps for r in subset]
                accs = [r.final_accuracy for r in subset]
                ax.plot(ts, accs, "o-", color=cmap[idx], alpha=0.85, linewidth=2.5,
                        markersize=8, markeredgecolor="white", label=f"N={n}")
        ax.set_xscale("log")
        ax.set_xlabel("Timesteps (T)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_ylabel("Final Accuracy", color=_STYLE["text_color"], fontsize=_STYLE["label_size"], fontweight="bold")
        ax.set_title("Accuracy Convergence vs Timesteps", color=_STYLE["text_color"], fontsize=_STYLE["title_size"], fontweight="bold")
        ax.set_ylim(0.55, 1.02)
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, alpha=0.4, color=_STYLE["grid_color"], linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=_STYLE["legend_size"]-2,
                      facecolor="white", edgecolor=_STYLE["grid_color"], labelcolor="black", ncol=2)
        _add_watermark(ax)
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "accuracy_vs_timesteps.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated accuracy vs timesteps: {path.name}")
        return path

    def _plot_gnn_serialization_footprint(self) -> Optional[Path]:
        """Bar chart of total bytes per emitted format from Step 3 format_statistics.json."""
        if not _MPL_AVAILABLE:
            return None
        stats = self._gnn_format_statistics
        if not isinstance(stats, dict):
            return None
        labels: List[str] = []
        sizes_mb: List[float] = []
        for key in ("markdown", "python", "json"):
            block = stats.get(key)
            if not isinstance(block, dict):
                continue
            sz = block.get("total_size")
            if sz is None:
                continue
            labels.append(key)
            sizes_mb.append(float(sz) / 1e6)
        if not labels:
            return None
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(_STYLE["bg_color"])
        ax.set_facecolor(_STYLE["axis_bg"])
        palette = ["#457B9D", "#E63946", "#2A9D8F"]
        ax.bar(labels, sizes_mb, color=palette[: len(labels)], edgecolor=_STYLE["grid_color"])
        ax.set_ylabel("Total size (MB)", color=_STYLE["text_color"], fontsize=_STYLE["label_size"])
        ax.set_title(
            "Step 3 multi-format serialization footprint",
            color=_STYLE["text_color"],
            fontsize=_STYLE["title_size"],
            fontweight="bold",
        )
        ax.tick_params(colors=_STYLE["text_color"], labelsize=_STYLE["tick_size"])
        ax.grid(True, axis="y", alpha=0.35, color=_STYLE["grid_color"], linestyle="--")
        _add_watermark(ax)
        fig.tight_layout()
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "gnn_serialization_footprint.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated GNN serialization footprint: {path.name}")
        return path

    def _plot_runtime_uncertainty(
        self,
        records: List[SweepRecord],
        frameworks: List[str],
    ) -> Optional[Path]:
        """Bar chart of median runtime with ±σ when benchmark repeats recorded σ > 0."""
        if not _MPL_AVAILABLE:
            return None
        uncertain = [
            r
            for r in records
            if r.framework in frameworks
            and r.execution_time_std is not None
            and r.execution_time_std > 0
            and r.success
            and r.num_states is not None
            and r.num_timesteps is not None
        ]
        if not uncertain:
            return None
        by_t: Dict[int, List[SweepRecord]] = {}
        for r in uncertain:
            by_t.setdefault(int(r.num_timesteps or 0), []).append(r)

        n_panels = len(by_t)
        fig, axes_arr = plt.subplots(n_panels, 1, figsize=(12, 4 * max(1, n_panels)))
        if n_panels == 1:
            axes_list: List[Any] = [axes_arr]
        else:
            axes_list = list(axes_arr)

        for ax, (t_val, group) in zip(axes_list, sorted(by_t.items())):
            group.sort(key=lambda r: (r.num_states or 0, r.framework))
            xs = np.arange(len(group))
            heights = [r.execution_time for r in group]
            yerr = [float(r.execution_time_std or 0) for r in group]
            colors_b = [_get_color(r.framework) for r in group]
            ax.bar(xs, heights, yerr=yerr, color=colors_b, capsize=4, edgecolor=_STYLE["grid_color"])
            ax.set_xticks(xs)
            ax.set_xticklabels([f"N={r.num_states}\n({r.framework})" for r in group], fontsize=10)
            ax.set_ylabel("Time (s)", color=_STYLE["text_color"])
            ax.set_title(f"Runtime ± σ (T={t_val})", color=_STYLE["text_color"], fontweight="bold")
            ax.tick_params(colors=_STYLE["text_color"])
            ax.grid(True, axis="y", alpha=0.35, color=_STYLE["grid_color"], linestyle="--")
            _add_watermark(ax)
        fig.suptitle(
            "Benchmark repeat uncertainty",
            fontsize=_STYLE["title_size"],
            fontweight="bold",
            color=_STYLE["text_color"],
        )
        fig.tight_layout()
        cf_dir = self.output_dir / "cross_framework" / "comparisons"
        cf_dir.mkdir(parents=True, exist_ok=True)
        path = cf_dir / "runtime_uncertainty.png"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Generated runtime uncertainty plot: {path.name}")
        return path
