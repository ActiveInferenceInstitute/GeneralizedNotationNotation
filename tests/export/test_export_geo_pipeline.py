#!/usr/bin/env python3
"""Opt-in GEO-INFER export wiring through process_export and Step 7.

Covers:
  - Five default pipeline formats are unchanged when no geo options are given
  - process_export with geo_infer options produces the .geo-infer.json artifact
  - Missing geo_infer.step_seconds fails visibly and writes no geo artifact
  - Step 7 geo_* flags reach process_export (arg-parsing + adapter level)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = SRC
SOURCE = ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"


def _load_step7_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "step7_export_under_test", SRC / "src" / "gnn" / "7_export.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_step3_results(output_dir: Path, parsed_model_file: Path) -> None:
    """Stage the step-3 results file process_export expects."""
    from gnn.pipeline.config import get_output_dir_for_script

    gnn_dir = get_output_dir_for_script("3_gnn.py", output_dir)
    gnn_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "processed_files": [
            {
                "file_name": SOURCE.name,
                "file_path": str(SOURCE),
                "parse_success": True,
                "parsed_model_file": str(parsed_model_file),
            }
        ]
    }
    (gnn_dir / "gnn_processing_results.json").write_text(json.dumps(results))


def _run_export(output_dir: Path, **kwargs: Any) -> bool:
    from gnn.export.processor import process_export

    parsed = output_dir / "parsed_model.json"
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text(json.dumps({"sections": {}}))
    _make_step3_results(output_dir, parsed)
    return process_export(
        target_dir=ROOT / "input/gnn_files/pomdp_gridworld",
        output_dir=output_dir,
        **kwargs,
    )


def _default_formats() -> list[str]:
    from gnn.export.registry import DEFAULT_PIPELINE_FORMATS

    return list(DEFAULT_PIPELINE_FORMATS)


# -- (i) Defaults unchanged ----------------------------------------------------


def test_pipeline_defaults_are_five_without_geo() -> None:
    from gnn.export.registry import DEFAULT_PIPELINE_FORMATS, resolve_format_writer

    assert DEFAULT_PIPELINE_FORMATS == ("json", "xml", "graphml", "gexf", "pickle")
    from gnn.export.processor import _PIPELINE_WRITERS

    assert set(_PIPELINE_WRITERS) == set(DEFAULT_PIPELINE_FORMATS)
    assert "geo_infer" not in _PIPELINE_WRITERS
    assert resolve_format_writer("geo_infer") is not None


def test_process_export_without_geo_writes_five_and_no_geo_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    assert _run_export(output_dir) is True
    file_dir = output_dir / SOURCE.stem
    written = sorted(p.name for p in file_dir.iterdir())
    assert written == sorted(
        [
            f"{SOURCE.stem}_graphml.graphml",
            f"{SOURCE.stem}_gexf.gexf",
            f"{SOURCE.stem}_json.json",
            f"{SOURCE.stem}_pickle.pkl",
            f"{SOURCE.stem}_xml.xml",
        ]
    )
    summary = json.loads((output_dir / "export_summary.json").read_text())
    assert set(summary["formats_generated"]) == set(_default_formats())


# -- (ii) Opt-in geo_infer artifact --------------------------------------------


def test_process_export_with_geo_options_writes_geo_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert (
        _run_export(
            output_dir,
            formats=[*_default_formats(), "geo_infer"],
            geo_infer={"step_seconds": 1},
        )
        is True
    )
    geo_file = output_dir / SOURCE.stem / f"{SOURCE.stem}_geo_infer.geo-infer.json"
    assert geo_file.exists()
    artifact = json.loads(geo_file.read_text())
    assert artifact["schema_version"] == "gnn-geo-infer/1"
    # The five defaults are still produced alongside.
    assert (output_dir / SOURCE.stem / f"{SOURCE.stem}_json.json").exists()
    summary = json.loads((output_dir / "export_summary.json").read_text())
    assert summary["formats_generated"]["geo_infer"] == 1


# -- (iii) Visible failure without step_seconds ---------------------------------


def test_geo_infer_without_step_seconds_fails_visibly_and_writes_nothing(
    tmp_path: Path,
) -> None:
    from gnn.export.processor import process_export

    output_dir = tmp_path / "out"
    parsed = output_dir / "parsed_model.json"
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text(json.dumps({"sections": {}}))
    _make_step3_results(output_dir, parsed)
    result = process_export(
        target_dir=ROOT / "input/gnn_files/pomdp_gridworld",
        output_dir=output_dir,
        formats=[*_default_formats(), "geo_infer"],
    )
    assert result is False
    assert not list(output_dir.rglob("*geo_infer*"))


# -- (iv) Step 7 CLI flag opt-in -------------------------------------------------


def test_step7_parser_accepts_geo_flags() -> None:
    from gnn.utils.arguments.arg_parsing import ArgumentParser

    parser = ArgumentParser.create_step_parser("7_export.py")
    parsed = parser.parse_args(
        [
            "--geo-step-seconds",
            "2.5",
            "--geo-state-ids",
            "/tmp/ids.json",
            "--geo-space-kind",
            "h3",
        ]
    )
    assert parsed.geo_step_seconds == 2.5
    assert parsed.geo_state_ids == "/tmp/ids.json"
    assert parsed.geo_space_kind == "h3"


def test_step7_adapter_requires_step_seconds_and_builds_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_step7_module()
    captured: dict[str, Any] = {}

    def fake_process_export(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(module, "process_export", fake_process_export)
    with pytest.raises(ValueError, match="step_seconds"):
        module._export_with_geo(geo_state_ids="ids.json")
    assert module._export_with_geo(
        target_dir=tmp_path,
        output_dir=tmp_path,
        geo_step_seconds=1.5,
        geo_state_ids="ids.json",
        geo_space_kind="h3",
    )
    assert captured["geo_infer"] == {
        "step_seconds": 1.5,
        "state_ids_path": "ids.json",
        "space_kind": "h3",
    }
    assert "geo_infer" in captured["formats"]
    assert captured["formats"][:5] == _default_formats()
