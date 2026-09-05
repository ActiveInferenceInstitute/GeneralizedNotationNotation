"""Run manifest verification must bind inventory and top-level provenance."""

import json
from pathlib import Path

import pytest

from pipeline.run_manifest import emit_run_manifests, verify_run_manifests

from .test_run_manifest import _build_fake_run


@pytest.mark.parametrize(
    "change",
    [
        "omit",
        "duplicate",
        "extra",
        "run_id",
        "run_hash",
        "overall_status",
        "count",
        "source",
    ],
)
def test_manifest_detects_inventory_and_provenance_changes(
    tmp_path: Path, change: str
) -> None:
    run = _build_fake_run(tmp_path)
    out = Path(emit_run_manifests(run)["manifest_dir"])
    assert verify_run_manifests(out, run) == []
    path = out / "index.json"
    data = json.loads(path.read_text())
    if change == "omit":
        omitted = data["manifests"].pop()
        (run / omitted["source"]).write_text("changed omitted evidence")
    elif change == "duplicate":
        data["manifests"].append(data["manifests"][0])
    elif change == "count":
        data["stream_count"] += 1
    elif change == "source":
        data["manifests"][0]["source"] = "different.json"
    elif change == "extra":
        (run / "3_gnn_output" / "new.json").write_text("{}")
    else:
        summary_path = run / "00_pipeline_summary" / "pipeline_execution_summary.json"
        summary = json.loads(summary_path.read_text())
        summary[change] = "different"
        summary_path.write_text(json.dumps(summary))
    path.write_text(json.dumps(data))
    assert verify_run_manifests(out, run)


@pytest.mark.parametrize(
    "file,root",
    [
        ("index.json", []),
        ("index.json", {"manifests": [None]}),
        ("execution_trace.json", []),
        ("summary", []),
    ],
)
def test_malformed_manifest_inputs_return_diagnostics(
    tmp_path: Path, file: str, root: object
) -> None:
    run = _build_fake_run(tmp_path)
    out = Path(emit_run_manifests(run)["manifest_dir"])
    path = (
        out / file
        if file != "summary"
        else run / "00_pipeline_summary" / "pipeline_execution_summary.json"
    )
    path.write_text(json.dumps(root))
    assert verify_run_manifests(out, run)
