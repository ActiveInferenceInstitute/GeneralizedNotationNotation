"""Current render receipts must not count retries or earlier runs twice."""

import json
import shutil
from pathlib import Path
from typing import Any

from render.processor import process_render

EXAMPLES = Path(__file__).resolve().parents[3] / "input/gnn_files/basics"


def render(source: Path, output: Path, run_id: str) -> dict[str, Any]:
    process_render(source, output, frameworks=["rxinfer"], run_id=run_id)
    payload: dict[str, Any] = json.loads(
        (output / "render_processing_summary.json").read_text()
    )
    return payload


def test_render_retry_recomputes_counts_and_tracks_identity(tmp_path: Path) -> None:
    source = tmp_path / "input"
    shutil.copytree(EXAMPLES, source)
    output = tmp_path / "output"
    first = render(source, output, "run-a")
    second = render(source, output, "run-a")
    assert second["total_files"] == first["total_files"] == len(second["file_results"])
    assert second["total_framework_attempts"] == first["total_framework_attempts"]
    assert second["receipt_identity"]["run_id"] == "run-a"
    assert second["receipt_identity"]["config_sha256"]
    for record in second["file_results"].values():
        assert record["source_identity"]["sha256"]
    assert list((output / "history").glob("render-*.json"))


def test_render_new_run_and_removed_inputs_do_not_retain_old_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    shutil.copytree(EXAMPLES, source)
    output = tmp_path / "output"
    first = render(source, output, "run-a")
    removed = Path(next(iter(first["file_results"])))
    removed.unlink()
    second = render(source, output, "run-a")
    assert str(removed) not in second["file_results"]
    assert second["total_files"] == first["total_files"] - 1
    other = tmp_path / "other"
    shutil.copytree(EXAMPLES, other)
    third = render(other, output, "run-b")
    assert all(Path(path).is_relative_to(other) for path in third["file_results"])


def test_render_empty_retry_clears_current_scope(tmp_path: Path) -> None:
    source = tmp_path / "input"
    shutil.copytree(EXAMPLES, source)
    output = tmp_path / "output"
    prior = render(source, output, "run-a")
    for path in prior["file_results"]:
        Path(path).unlink()
    assert process_render(source, output, frameworks=["rxinfer"], run_id="run-a") == 2
    current = json.loads((output / "render_processing_summary.json").read_text())
    assert current["file_results"] == {}
    assert current["total_files"] == 0


def test_bnlearn_capability_is_render_only() -> None:
    from render.framework_registry import FRAMEWORK_REGISTRY, get_available_renderers

    assert FRAMEWORK_REGISTRY["bnlearn"]["supports_execution"] is False
    assert get_available_renderers()["bnlearn"]["supports_execution"] is False
    assert all(
        spec["supports_execution"]
        for name, spec in FRAMEWORK_REGISTRY.items()
        if name != "bnlearn"
    )
