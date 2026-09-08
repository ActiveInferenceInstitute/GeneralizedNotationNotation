#!/usr/bin/env python3
"""
Plaintext CSV export mixin for GNN meta-analysis sweep visualizations.

Extracted from ``integration.meta_analysis.visualizer``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from .collector import SweepRecord
from .visualizer_style import (
    _fmt_time,
)


class SweepExportMixin:
    """Verbatim plot methods moved from ``SweepVisualizer``."""

    if TYPE_CHECKING:
        output_dir: Path
        logger: logging.Logger
    # ─── Plaintext data export ─────────────────────────────────────────────

    def _export_csv(self, records: List[SweepRecord]) -> Optional[Path]:
        """Export all sweep data as a CSV file for external analysis."""
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        path = data_dir / "sweep_data.csv"
        fieldnames: list[Any] = [
            "model_name",
            "framework",
            "num_states",
            "num_timesteps",
            "execution_time_s",
            "execution_time_std_s",
            "execution_benchmark_repeats",
            "time_per_step_ms",
            "success",
            "timed_out",
            "lines_of_code",
            "total_lines",
            "final_accuracy",
            "mean_belief_entropy",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(
                records,
                key=lambda x: (x.framework, x.num_states or 0, x.num_timesteps or 0),
            ):
                writer.writerow(
                    {
                        "model_name": r.model_name,
                        "framework": r.framework,
                        "num_states": r.num_states,
                        "num_timesteps": r.num_timesteps,
                        "execution_time_s": f"{r.execution_time:.3f}"
                        if r.execution_time > 0
                        else "",
                        "execution_time_std_s": (
                            f"{r.execution_time_std:.3f}"
                            if r.execution_time_std is not None
                            and r.execution_time_std > 0
                            else ""
                        ),
                        "execution_benchmark_repeats": r.execution_benchmark_repeats,
                        "time_per_step_ms": f"{r.time_per_step:.4f}"
                        if r.time_per_step
                        else "",
                        "success": r.success,
                        "timed_out": r.timed_out,
                        "lines_of_code": r.lines_of_code or "",
                        "total_lines": r.total_lines or "",
                        "final_accuracy": f"{r.final_accuracy:.4f}"
                        if r.final_accuracy is not None
                        else "",
                        "mean_belief_entropy": f"{r.mean_belief_entropy:.6f}"
                        if r.mean_belief_entropy is not None
                        else "",
                    }
                )

        # Also write a human-readable TSV summary
        txt_path = data_dir / "sweep_data.txt"
        with open(txt_path, "w") as f:
            f.write(
                f"{'Model':<35} {'Framework':<20} {'N':>4} {'T':>8} {'Runtime':>12} {'ms/step':>10} {'Accuracy':>10} {'Entropy':>10}\n"
            )
            f.write("=" * 115 + "\n")
            for r in sorted(
                records,
                key=lambda x: (x.framework, x.num_states or 0, x.num_timesteps or 0),
            ):
                rt = _fmt_time(r.execution_time) if r.execution_time > 0 else "—"
                tps = f"{r.time_per_step:.2f}" if r.time_per_step else "—"
                acc = f"{r.final_accuracy:.3f}" if r.final_accuracy is not None else "—"
                ent = (
                    f"{r.mean_belief_entropy:.4f}"
                    if r.mean_belief_entropy is not None
                    else "—"
                )
                f.write(
                    f"{r.model_name:<35} {r.framework:<20} {r.num_states or 0:>4} {r.num_timesteps or 0:>8} {rt:>12} {tps:>10} {acc:>10} {ent:>10}\n"
                )

        self.logger.info(f"Exported plaintext data: {path.name}, {txt_path.name}")
        return path
