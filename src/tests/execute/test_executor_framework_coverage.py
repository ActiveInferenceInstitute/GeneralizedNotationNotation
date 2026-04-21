"""Smoke tests for ``execute.executor.execute_rendered_simulators`` framework coverage.

Verifies that the NumPyro and PyTorch branches are wired in alongside PyMDP,
RxInfer, DisCoPy, ActiveInference.jl, and JAX — and that all framework result
buckets are populated even when the render input is empty (everything should
be either SKIPPED or run-and-report-no-scripts, not missing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execute.executor import execute_rendered_simulators  # noqa: E402


EXPECTED_FRAMEWORK_KEYS = {
    "pymdp_executions",
    "rxinfer_executions",
    "discopy_executions",
    "activeinference_executions",
    "jax_executions",
    "numpyro_executions",
    "pytorch_executions",
}


def test_executor_covers_all_frameworks(tmp_path: Path) -> None:
    import logging

    empty_render_dir = tmp_path / "11_render_output"
    empty_render_dir.mkdir()
    output_dir = tmp_path / "out"

    logger = logging.getLogger("test_executor_framework_coverage")

    result = execute_rendered_simulators(
        target_dir=empty_render_dir,
        output_dir=output_dir,
        logger=logger,
        recursive=False,
        verbose=False,
    )

    # With an empty render dir every framework should either skip or succeed
    # without exploding. The function returns True on all-success / all-skip.
    assert isinstance(result, bool)

    summary_file = (
        output_dir / "12_execute_output" / "summaries" / "execution_summary.json"
    )
    assert summary_file.exists(), (
        f"Expected execution summary at {summary_file}, got tree:"
        f" {[p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file()]}"
    )

    summary = json.loads(summary_file.read_text())
    assert EXPECTED_FRAMEWORK_KEYS.issubset(summary.keys()), (
        f"Missing framework buckets: {EXPECTED_FRAMEWORK_KEYS - summary.keys()}"
    )

    framework_dirs = summary.get("framework_execution_dirs", {})
    assert {"numpyro", "pytorch"}.issubset(framework_dirs.keys()), (
        "numpyro/pytorch output directories not declared in framework_execution_dirs"
    )
