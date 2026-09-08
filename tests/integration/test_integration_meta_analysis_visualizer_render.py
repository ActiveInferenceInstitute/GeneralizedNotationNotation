#!/usr/bin/env python3
"""Render-smoke tests for the SweepVisualizer mixin split (MAJ-04 5/6).

The 20 plot methods moved into ``visualizer_*`` sibling mixins are
best-effort: ``generate_all`` swallows any per-method exception into a
non-fatal warning. These tests run a full synthetic sweep grid through
``generate_all`` and assert that *every* family renders without a
failure warning, locking the mixin wiring (MRO dispatch, style imports,
attribute stubs) against regressions that import probes cannot see.
"""

import logging
from pathlib import Path

import pytest

from gnn.integration.meta_analysis.collector import SweepRecord
from gnn.integration.meta_analysis.visualizer import SweepVisualizer


def _sweep_record(framework: str, n: int, t: int) -> SweepRecord:
    """One fully-populated sweep cell exercising every plot family."""
    return SweepRecord(
        model_name=f"{framework}_scaling_N{n}_T{t}",
        framework=framework,
        num_states=n,
        num_timesteps=t,
        execution_time=0.5 + 0.1 * n + 0.01 * t,
        execution_time_std=0.05,
        execution_time_mean=0.55 + 0.1 * n,
        execution_benchmark_repeats=3,
        execution_time_samples=[0.4, 0.5, 0.6],
        success=True,
        lines_of_code=100 * n,
        total_lines=120 * n,
        final_accuracy=0.6 + 0.01 * t,
        mean_belief_entropy=max(0.1, 1.5 - 0.01 * t),
        efe_trace=[-0.1 * k for k in range(t)],
        vfe_trace=[-0.2 * k for k in range(t)],
        model_params={"num_states": n, "num_timesteps": t},
    )


FORMAT_STATISTICS = {
    "markdown": {"total_size": 1200},
    "python": {"total_size": 3400},
    "json": {"total_size": 900},
}


def _full_grid() -> list[SweepRecord]:
    return [
        _sweep_record(framework=framework, n=n, t=t)
        for framework in ("pymdp", "rxinfer")
        for n in (3, 9)
        for t in (10, 50)
    ]


def test_generate_all_renders_every_family_without_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    visualizer = SweepVisualizer(
        _full_grid(),
        tmp_path / "visualizations",
        gnn_format_statistics=FORMAT_STATISTICS,
    )

    generated = visualizer.generate_all()

    failures = [
        record.getMessage()
        for record in caplog.records
        if "failed (non-fatal)" in record.getMessage()
    ]
    assert failures == [], f"plot families failed: {failures}"
    # export, so a fully healthy grid yields >= 19 artifacts.
    assert len(generated) >= 19, (
        f"expected >= 19 artifacts from a full sweep grid, got {len(generated)}"
    )
    for artifact in generated:
        assert Path(artifact).exists(), f"reported artifact missing: {artifact}"

    # Key families actually produced their files.
    names = {Path(p).name for p in generated}
    assert "sweep_data.csv" in names
    assert any("runtime_heatmap" in n for n in names)
    assert any("dashboard" in n for n in names)


def test_generate_all_is_repeatable(tmp_path: Path) -> None:
    """A second run over the same grid regenerates cleanly (no state leaks)."""
    records = _full_grid()
    first = SweepVisualizer(
        records, tmp_path / "a", gnn_format_statistics=FORMAT_STATISTICS
    )
    second = SweepVisualizer(
        records, tmp_path / "b", gnn_format_statistics=FORMAT_STATISTICS
    )

    first_generated = first.generate_all()
    second_generated = second.generate_all()

    assert len(first_generated) == len(second_generated) > 0


import pytest  # noqa: E402  — kept last so the module docstring stays first
