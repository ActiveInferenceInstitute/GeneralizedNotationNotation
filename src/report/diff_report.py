#!/usr/bin/env python3
"""
Diff-Aware Pipeline Reporting — Compare runs to detect regressions.

Provides:
  - DiffReport: dataclass with timing deltas, new failures, status changes
  - compare_runs(): loads two pipeline summaries and computes diffs
  - archive_run(): copies current summary to .history/ with timestamp
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepDiff:
    """Diff for a single pipeline step between two runs."""
    step_name: str
    prev_status: str
    curr_status: str
    prev_duration: float
    curr_duration: float
    duration_delta_pct: float  # positive = slower
    is_regression: bool  # new failure or >20% slower


@dataclass
class DiffReport:
    """Comparison between two pipeline runs."""
    current_timestamp: str
    previous_timestamp: str
    step_diffs: List[StepDiff] = field(default_factory=list)
    new_failures: List[str] = field(default_factory=list)
    fixed_steps: List[str] = field(default_factory=list)
    timing_regressions: List[str] = field(default_factory=list)
    overall_badge: str = "🟢"  # 🟢 / 🟡 / 🔴

    def to_markdown(self) -> str:
        """Render diff as Markdown section for PIPELINE_REPORT.md."""
        lines = [f"## Run Comparison ({self.overall_badge})"]
        lines.append("")
        lines.append(f"Comparing **{self.current_timestamp}** vs **{self.previous_timestamp}**")
        lines.append("")

        if self.new_failures:
            lines.append(f"### 🔴 New Failures ({len(self.new_failures)})")
            for step in self.new_failures:
                lines.append(f"- ❌ `{step}` — newly failing")
            lines.append("")

        if self.fixed_steps:
            lines.append(f"### 🟢 Fixed ({len(self.fixed_steps)})")
            for step in self.fixed_steps:
                lines.append(f"- ✅ `{step}` — now passing")
            lines.append("")

        if self.timing_regressions:
            lines.append(f"### ⏱️ Timing Regressions ({len(self.timing_regressions)})")
            for step in self.timing_regressions:
                lines.append(f"- ⚠️ `{step}`")
            lines.append("")

        if self.step_diffs:
            lines.append("### Step-by-Step Deltas")
            lines.append("")
            lines.append("| Step | Prev | Curr | Δ Duration |")
            lines.append("|------|------|------|-----------|")
            for sd in self.step_diffs:
                sign = "+" if sd.duration_delta_pct > 0 else ""
                emoji = "🔴" if sd.is_regression else "🟢"
                lines.append(
                    f"| {sd.step_name} | {sd.prev_status} ({sd.prev_duration:.1f}s) "
                    f"| {sd.curr_status} ({sd.curr_duration:.1f}s) "
                    f"| {emoji} {sign}{sd.duration_delta_pct:.0f}% |"
                )

        return "\n".join(lines)


def compare_runs(current: Path, previous: Path) -> DiffReport:
    """
    Compare two pipeline_execution_summary.json files.

    Args:
        current: Path to current run's summary.
        previous: Path to previous run's summary.

    Returns:
        DiffReport with step-level comparisons.
    """
    try:
        with open(current) as f:
            curr_data = json.load(f)
        with open(previous) as f:
            prev_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load summaries for diff: {e}")
        return DiffReport(
            current_timestamp="unknown",
            previous_timestamp="unknown",
        )

    curr_steps = {s["name"]: s for s in curr_data.get("steps", []) if isinstance(s, dict)}
    prev_steps = {s["name"]: s for s in prev_data.get("steps", []) if isinstance(s, dict)}

    report = DiffReport(
        current_timestamp=curr_data.get("timestamp", "unknown"),
        previous_timestamp=prev_data.get("timestamp", "unknown"),
    )

    all_steps = set(curr_steps.keys()) | set(prev_steps.keys())
    for name in sorted(all_steps):
        cs = curr_steps.get(name, {})
        ps = prev_steps.get(name, {})

        curr_status = cs.get("status", "missing")
        prev_status = ps.get("status", "missing")
        curr_dur = cs.get("duration_seconds", 0)
        prev_dur = ps.get("duration_seconds", 0)

        delta_pct = ((curr_dur - prev_dur) / prev_dur * 100) if prev_dur > 0 else 0

        is_new_failure = (
            curr_status.lower() in ("failed", "error")
            and prev_status.lower() not in ("failed", "error")
        )
        is_fixed = (
            prev_status.lower() in ("failed", "error")
            and curr_status.lower() not in ("failed", "error")
        )
        is_regression = is_new_failure or delta_pct > 20

        sd = StepDiff(
            step_name=name,
            prev_status=prev_status,
            curr_status=curr_status,
            prev_duration=prev_dur,
            curr_duration=curr_dur,
            duration_delta_pct=delta_pct,
            is_regression=is_regression,
        )
        report.step_diffs.append(sd)

        if is_new_failure:
            report.new_failures.append(name)
        if is_fixed:
            report.fixed_steps.append(name)
        if delta_pct > 20 and not is_new_failure:
            report.timing_regressions.append(f"{name}: {delta_pct:+.0f}%")

    # Determine badge
    if report.new_failures:
        report.overall_badge = "🔴"
    elif report.timing_regressions:
        report.overall_badge = "🟡"
    else:
        report.overall_badge = "🟢"

    return report


def archive_run(
    summary_path: Path,
    history_dir: Optional[Path] = None,
    max_archives: int = 10,
) -> Optional[Path]:
    """
    Archive current pipeline summary to .history/ directory.

    Args:
        summary_path: Path to pipeline_execution_summary.json.
        history_dir: Archive directory. Defaults to summary_path.parent / ".history".
        max_archives: Maximum number of archived runs to keep.

    Returns:
        Path to archived file, or None on failure.
    """
    if not summary_path.exists():
        return None

    history_dir = history_dir or summary_path.parent / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = history_dir / f"{timestamp}.json"

    try:
        shutil.copy2(summary_path, archive_path)
        logger.info(f"📁 Archived run to: {archive_path}")
    except OSError as e:
        logger.warning(f"Could not archive run: {e}")
        return None

    # Prune old archives
    archives = sorted(history_dir.glob("*.json"))
    while len(archives) > max_archives:
        oldest = archives.pop(0)
        oldest.unlink()
        logger.debug(f"🗑️ Pruned old archive: {oldest.name}")

    return archive_path


def get_previous_run(history_dir: Path) -> Optional[Path]:
    """Get the most recent archived run, if any."""
    if not history_dir.exists():
        return None
    archives = sorted(history_dir.glob("*.json"))
    return archives[-1] if archives else None
