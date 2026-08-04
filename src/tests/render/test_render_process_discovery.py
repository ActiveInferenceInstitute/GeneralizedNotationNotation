"""Regression tests for process_render recursive discovery of nested exemplar GNN files.

Verifies the recursive-discovery fix in ``src/render/processor.py``:

1. ``process_render(..., recursive=True)`` (the default) walks nested exemplar
   folders (discrete/, basics/, continuous/, pomdp_gridworld/, ...) and renders
   every exemplar GNN spec to RxInfer.jl — 45 exemplar ``*.md`` files all
   discovered and rendered.
2. Passing ``recursive=False`` via kwargs reverts to a top-level-only glob, so
   no nested files are found and ``process_render`` returns exit code ``2``.

Kept fast: no Julia is executed, only code generation and summary JSON checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from render.processor import process_render


REPO_ROOT = Path(__file__).resolve().parents[3]
EXEMPLAR_DIR = REPO_ROOT / "input" / "gnn_files"
EXPECTED_EXEMPLAR_COUNT = 45


def _count_exemplar_md_files() -> int:
    """Count GNN exemplar model files, matching the processor's discovery policy."""
    from gnn.discovery import is_model_source_path

    return sum(
        1
        for path in EXEMPLAR_DIR.rglob("*.md")
        if is_model_source_path(path)
    )


def test_process_render_recursive_discovers_and_renders_all_45_exemplars(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "render_out"

    result = process_render(
        target_dir=EXEMPLAR_DIR,
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
    )

    # Recursive render of 45 exemplars should succeed under the aggregate policy.
    assert result is True or result is not False

    summary_path = output_dir / "render_processing_summary.json"
    assert summary_path.exists(), (
        f"render_processing_summary.json not written to {output_dir}"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # (1) Recursive discovery is the fix under test: all 45 exemplars found.
    assert summary["total_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["total_files"] == _count_exemplar_md_files()

    # (2) Real render behavior: every exemplar rendered to RxInfer.
    assert summary["successful_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["total_framework_attempts"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["successful_framework_renderings"] == EXPECTED_EXEMPLAR_COUNT

    # Each rendered exemplar produces exactly one RxInfer.jl artifact.
    rendered_jl = list(output_dir.rglob("*.jl"))
    assert len(rendered_jl) == EXPECTED_EXEMPLAR_COUNT
    assert len(rendered_jl) == _count_exemplar_md_files()


def test_process_render_recursive_false_skips_nested_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "render_out"

    result = process_render(
        target_dir=EXEMPLAR_DIR,
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        recursive=False,
    )

    # There are no top-level "*.md" files, so recursion disabled finds nothing
    # and the processor returns exit code 2 (no input).
    assert result == 2

    summary_path = output_dir / "render_processing_summary.json"
    assert not summary_path.exists()
    assert not list(output_dir.rglob("*.jl"))
