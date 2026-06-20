#!/usr/bin/env python3
"""
Tests for pipeline/run_manifest.py — durable run manifest emission.

No mocks: every test builds a REAL fake run output directory under ``tmp_path``
with real ``N_*_output/`` step subdirectories, real JSON artifact files, and a
real ``pipeline_execution_summary.json``. Assertions use real serialization,
real sha256 checksums (via durable_streams), and a NEGATIVE control that proves
artifact tampering is detected.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.durable_streams import read_trace, replay_trace
from pipeline.run_manifest import emit_run_manifests, verify_run_manifests


def _build_fake_run(root: Path) -> Path:
    """Create a realistic completed-run output directory under ``root``.

    Returns the run output directory path. It contains two ``N_*_output/`` step
    dirs with JSON artifacts and a ``00_pipeline_summary`` with a real summary.
    """
    run_dir = root / "output"
    run_dir.mkdir(parents=True)

    gnn = run_dir / "3_gnn_output"
    gnn.mkdir()
    (gnn / "gnn_processing_summary.json").write_text(
        json.dumps({"models": 2, "status": "ok"}, indent=2),
        encoding="utf-8",
    )
    (gnn / "format_statistics.json").write_text(
        json.dumps({"markdown": 5}, indent=2),
        encoding="utf-8",
    )

    setup = run_dir / "1_setup_output"
    setup.mkdir()
    (setup / "environment_setup_summary.json").write_text(
        json.dumps({"python": "3.11"}, indent=2),
        encoding="utf-8",
    )

    summary: Dict[str, Any] = {
        "run_hash": "deadbeef",
        "overall_status": "SUCCESS",
        "steps": [
            {"step_number": 1, "script_name": "1_setup.py", "status": "SUCCESS"},
            {"step_number": 3, "script_name": "3_gnn.py", "status": "SUCCESS"},
        ],
    }
    summary_dir = run_dir / "00_pipeline_summary"
    summary_dir.mkdir()
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_emit_produces_manifests_and_sound_trace(tmp_path: Path) -> None:
    """emit_run_manifests yields >=1 manifest and a trace_integrity_ok trace."""
    run_dir = _build_fake_run(tmp_path)

    result = emit_run_manifests(run_dir)

    # Three JSON artifacts across the two step dirs.
    assert result["stream_count"] == 3
    # Two step records in the summary -> two trace events.
    assert result["trace_event_count"] == 2
    assert result["trace_integrity_ok"] is True

    manifest_dir = Path(result["manifest_dir"])
    assert manifest_dir.is_dir()
    assert (manifest_dir / "index.json").is_file()
    assert (manifest_dir / "execution_trace.json").is_file()

    # verify returns no problems on a freshly emitted, untampered run.
    assert verify_run_manifests(manifest_dir, run_dir) == []


def test_round_trip_validates_clean(tmp_path: Path) -> None:
    """Emitted manifests re-validate clean against the run dir (round-trip)."""
    run_dir = _build_fake_run(tmp_path)
    emit_run_manifests(run_dir, manifest_out=tmp_path / "emitted")

    problems = verify_run_manifests(tmp_path / "emitted", run_dir)
    assert problems == [], f"expected clean round-trip, got: {problems}"


def test_negative_tampered_artifact_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: modifying an artifact after emission fires a checksum mismatch."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    # Sanity: clean before tampering.
    assert verify_run_manifests(manifest_dir, run_dir) == []

    # Tamper with one artifact's bytes AFTER emission.
    target = run_dir / "3_gnn_output" / "gnn_processing_summary.json"
    target.write_text(
        json.dumps({"models": 999, "status": "TAMPERED"}, indent=2),
        encoding="utf-8",
    )

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems, "expected verification to report the tampered artifact"
    assert any("checksum mismatch" in p for p in problems), problems


def test_negative_missing_artifact_is_detected(tmp_path: Path) -> None:
    """NEGATIVE: deleting an artifact after emission fires a missing-source error."""
    run_dir = _build_fake_run(tmp_path)
    result = emit_run_manifests(run_dir)
    manifest_dir = Path(result["manifest_dir"])

    (run_dir / "1_setup_output" / "environment_setup_summary.json").unlink()

    problems = verify_run_manifests(manifest_dir, run_dir)
    assert problems
    assert any("does not exist" in p for p in problems), problems


def test_determinism_identical_stream_count_and_trace_digest(tmp_path: Path) -> None:
    """Emitting twice over the same run dir is deterministic (count + trace digest)."""
    run_dir = _build_fake_run(tmp_path)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = emit_run_manifests(run_dir, manifest_out=out_a)
    result_b = emit_run_manifests(run_dir, manifest_out=out_b)

    assert result_a["stream_count"] == result_b["stream_count"]
    assert result_a["trace_event_count"] == result_b["trace_event_count"]

    trace_a = read_trace(out_a / "execution_trace.json")
    trace_b = read_trace(out_b / "execution_trace.json")
    assert replay_trace(trace_a) == replay_trace(trace_b)


def test_trace_falls_back_to_step_dirs_without_summary(tmp_path: Path) -> None:
    """Without a summary, the trace is built from the sorted step directories."""
    run_dir = _build_fake_run(tmp_path)
    # Remove the summary so the directory-fallback path is exercised.
    (run_dir / "00_pipeline_summary" / "pipeline_execution_summary.json").unlink()

    result = emit_run_manifests(run_dir, manifest_out=tmp_path / "nosummary")

    # Two step dirs -> two trace events, still sound, manifests still emitted.
    assert result["trace_event_count"] == 2
    assert result["trace_integrity_ok"] is True
    assert result["stream_count"] == 3
    assert verify_run_manifests(tmp_path / "nosummary", run_dir) == []


def test_missing_run_dir_raises(tmp_path: Path) -> None:
    """A non-existent run output dir raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        emit_run_manifests(tmp_path / "nope")


def test_stream_ids_are_stable_and_unique(tmp_path: Path) -> None:
    """Each emitted manifest carries a unique, path-derived stream id."""
    run_dir = _build_fake_run(tmp_path)
    emit_run_manifests(run_dir, manifest_out=tmp_path / "ids")
    index = json.loads((tmp_path / "ids" / "index.json").read_text(encoding="utf-8"))
    ids = [m["stream_id"] for m in index["manifests"]]
    assert len(ids) == len(set(ids)) == 3
    # Ids are path-derived (readable slug prefix) and injective (hash suffix).
    assert any(i.startswith("3_gnn_output_gnn_processing_summary_json_") for i in ids)


def test_collision_prone_filenames_each_get_a_manifest(tmp_path: Path) -> None:
    """NEGATIVE: distinct files that slugify identically (a.b.json / a-b.json) must
    each get their OWN manifest — no silent overwrite that leaves bytes unbound."""
    run_dir = tmp_path / "output"
    step = run_dir / "3_gnn_output"
    step.mkdir(parents=True)
    (step / "a.b.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    (step / "a-b.json").write_text(json.dumps({"x": 2}), encoding="utf-8")
    out = tmp_path / "m"
    result = emit_run_manifests(run_dir, manifest_out=out)

    manifest_files = list(out.glob("*.manifest.json"))
    # stream_count must equal the number of manifest files actually on disk.
    assert result["stream_count"] == len(manifest_files) == 2
    assert verify_run_manifests(out, run_dir) == []
    # Tampering EITHER artifact is now detected (both are bound).
    (step / "a-b.json").write_text(json.dumps({"x": 999}), encoding="utf-8")
    assert any("checksum mismatch" in p for p in verify_run_manifests(out, run_dir))


def test_negative_tampered_summary_breaks_trace_binding(tmp_path: Path) -> None:
    """NEGATIVE: flipping a step status in the summary AFTER emission is caught by
    the trace's re-binding to ground truth (not just structural integrity)."""
    run_dir = _build_fake_run(tmp_path)
    out = tmp_path / "bind"
    emit_run_manifests(run_dir, manifest_out=out)
    assert verify_run_manifests(out, run_dir) == []

    summary_path = run_dir / "00_pipeline_summary" / "pipeline_execution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["steps"][1]["status"] = "FAILED"  # flip 3_gnn.py SUCCESS -> FAILED
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    problems = verify_run_manifests(out, run_dir)
    assert any("re-derived from the live run summary" in p for p in problems)
