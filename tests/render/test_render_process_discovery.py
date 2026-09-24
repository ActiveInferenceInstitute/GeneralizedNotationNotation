"""Regression tests for process_render recursive discovery of nested exemplar GNN files.

Verifies the recursive-discovery fix in ``src/render/processor.py``:

1. ``process_render(..., recursive=True)`` (the default) walks nested exemplar
   folders (discrete/, basics/, continuous/, pomdp_gridworld/, ...) and renders
   every exemplar GNN spec to RxInfer.jl — 36 exemplar ``*.md`` files are all
   discovered; the 31 plain ones render, and the five receipted exemplars are
   never rendered: the composed trio (continuous × multi-agent, hybrid,
   factored-continuous LGSSM) plus the two non-stationary discrete specs
   (``unsupported-nonstationary``).
2. Passing ``recursive=False`` via kwargs reverts to a top-level-only glob, so
   no nested files are found and ``process_render`` returns exit code ``2``.

Kept fast: no Julia is executed, only code generation and summary JSON checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.render.processor import process_render

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEMPLAR_DIR = REPO_ROOT / "input" / "gnn_files"
EXPECTED_EXEMPLAR_COUNT = 36
# Plain exemplars that still render to RxInfer.jl; the five receipted
# exemplars (composed trio + the two non-stationary discrete specs) are
# never rendered.
EXPECTED_RENDERED_COUNT = 31


def _count_exemplar_md_files() -> int:
    """Count GNN exemplar model files, matching the processor's discovery policy."""
    from gnn.processing.discovery import is_model_source_path

    return sum(1 for path in EXEMPLAR_DIR.rglob("*.md") if is_model_source_path(path))


def test_process_render_recursive_discovers_and_renders_all_exemplars(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "render_out"

    result = process_render(
        target_dir=EXEMPLAR_DIR,
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
    )

    # Recursive render of all exemplars should succeed under the aggregate policy.
    assert result is True or result is not False

    summary_path = output_dir / "render_processing_summary.json"
    assert summary_path.exists(), (
        f"render_processing_summary.json not written to {output_dir}"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # (1) Recursive discovery is the fix under test: all exemplars found.
    assert summary["total_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["total_files"] == _count_exemplar_md_files()

    # (2) Real render behavior: every plain exemplar rendered to RxInfer.
    assert summary["successful_files"] == EXPECTED_EXEMPLAR_COUNT
    assert summary["total_framework_attempts"] == EXPECTED_RENDERED_COUNT
    assert summary["successful_framework_renderings"] == EXPECTED_RENDERED_COUNT

    # (3) The five receipted exemplars are receipted on RxInfer — never
    # rendered as one family or rendered flat.
    expected_receipts = {
        "multi_agent_lgssm": "unsupported-composition",
        "hybrid_discrete_continuous": "unsupported-composition",
        "factored_continuous_lgssm": "unsupported-factored-continuous",
        "time_varying_dynamics": "unsupported-nonstationary",
        "regime_switched_dynamics": "unsupported-nonstationary",
    }
    for stem, prefix in expected_receipts.items():
        entries = [
            entry
            for entry in summary["unsupported_framework_renderings"]
            if stem in entry["file"]
        ]
        assert len(entries) == 1, stem
        assert entries[0]["framework"] == "rxinfer"
        assert prefix in entries[0]["message"], stem

    # Each plain rendered exemplar produces exactly one RxInfer.jl artifact.
    rendered_jl = list(output_dir.rglob("*.jl"))
    assert len(rendered_jl) == EXPECTED_RENDERED_COUNT
    assert not any(
        stem in path.name for stem in expected_receipts for path in rendered_jl
    )


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


def test_process_render_aggregates_summary_across_invocations(tmp_path: Path) -> None:
    """Sequential per-folder invocations must accumulate file_results.

    The pipeline invokes ``process_render`` once per top-level input folder,
    each writing the same ``render_processing_summary.json``. Without
    aggregation only the last folder's ``file_results`` survive, and Step 12's
    manifest-based discovery executes just that folder.
    """
    output_dir = tmp_path / "render_out"

    first = process_render(
        target_dir=EXEMPLAR_DIR / "basics",
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        run_id="folder-aggregation-test",
    )
    assert first is not False

    summary = json.loads(
        (output_dir / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    first_total = summary["total_files"]
    first_keys = set(summary["file_results"])
    assert first_total == len(first_keys) == 2

    second = process_render(
        target_dir=EXEMPLAR_DIR / "discrete",
        output_dir=output_dir,
        frameworks=["rxinfer"],
        verbose=False,
        run_id="folder-aggregation-test",
    )
    assert second is not False

    summary = json.loads(
        (output_dir / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    merged_keys = set(summary["file_results"])

    # The second invocation carried forward the first folder's results and
    # added its own: nothing was dropped, and the aggregate count matches.
    assert first_keys <= merged_keys
    assert len(merged_keys) > first_total
    assert summary["total_files"] == len(merged_keys)
    assert any("basics/" in key for key in merged_keys)
    assert any("discrete/" in key for key in merged_keys)
