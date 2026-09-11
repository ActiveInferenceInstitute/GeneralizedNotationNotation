#!/usr/bin/env python3
"""Notation-derived GEO-INFER metadata (GNN-05 / S2-10).

Covers:
  - Derivation on an exemplar without time hints fails visibly, never defaulting
  - Derivation on an exemplar with Time hints matches the explicit path
  - Explicit options beat derived values when both are present
  - The structured provenance record lands in the artifact
  - Explicit options cannot claim derivation provenance
  - Step 7 --geo-derive-metadata wiring (parser + adapter)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROOT = SRC
GRIDWORLD = ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
GAUSSIAN = Path(__file__).with_name("gaussian_rectangular.md")
GAUSSIAN_UNITS: dict[str, list[str]] = {
    "states": ["m", "m/s", "K"],
    "observations": ["m", "m/s"],
    "controls": ["N"],
}


def _load_step7_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "step7_export_derive_test", SRC / "src" / "gnn" / "7_export.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gridworld_with_time_step(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the gridworld exemplar and declare ``TimeStep=60s`` in ## Time."""
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    model = source_dir / GRIDWORLD.name
    model.write_text(
        GRIDWORLD.read_text().replace(
            "ModelTimeHorizon=15", "ModelTimeHorizon=15\nTimeStep=60s"
        )
    )
    return source_dir, model


def _run_geo_export(
    output_dir: Path, source_dir: Path, model: Path, **kwargs: Any
) -> bool:
    """Stage the Step 3 results file and run ``process_export``."""
    from gnn.export.processor import process_export
    from gnn.pipeline.config import get_output_dir_for_script

    parsed = output_dir / "parsed_model.json"
    parsed.parent.mkdir(parents=True, exist_ok=True)
    parsed.write_text(json.dumps({"sections": {}}))
    gnn_dir = get_output_dir_for_script("3_gnn.py", output_dir)
    gnn_dir.mkdir(parents=True, exist_ok=True)
    (gnn_dir / "gnn_processing_results.json").write_text(
        json.dumps(
            {
                "processed_files": [
                    {
                        "file_name": model.name,
                        "file_path": str(model),
                        "parse_success": True,
                        "parsed_model_file": str(parsed),
                    }
                ]
            }
        )
    )
    return process_export(target_dir=source_dir, output_dir=output_dir, **kwargs)


def _geo_artifact(output_dir: Path, model: Path) -> dict[str, Any]:
    geo_file = next(
        (output_dir / model.stem).glob(f"{model.stem}_geo_infer.geo-infer.json")
    )
    return cast(dict[str, Any], json.loads(geo_file.read_text()))


# -- (a) Visible failure without time hints ------------------------------------


def test_derivation_fails_visibly_without_time_hints() -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    with pytest.raises(NotationMetadataError, match="no time-step declaration"):
        derive_geo_metadata(GRIDWORLD.read_text())


def test_pipeline_derivation_failure_is_recorded_and_writes_nothing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    assert (
        _run_geo_export(
            output_dir,
            GRIDWORLD.parent,
            GRIDWORLD,
            formats=["geo_infer"],
            geo_derive_metadata=True,
        )
        is False
    )
    manifest = json.loads((output_dir / "export_results.json").read_text())
    entry = manifest["files_exported"][0]["exports"]["geo_infer"]
    assert entry["success"] is False
    assert "no time-step declaration" in entry["error"]
    assert "TimeStep" in entry["error"]
    assert not list(output_dir.rglob("*geo_infer*"))


# -- (b) Derivation matches the explicit path -----------------------------------


def test_derived_options_match_explicit_values(tmp_path: Path) -> None:
    from gnn.export.notation_metadata import derive_geo_metadata

    source_dir, model = _gridworld_with_time_step(tmp_path)
    derived = derive_geo_metadata(model.read_text())
    assert derived["options"] == {"step_seconds": 60.0}

    derived_out = tmp_path / "derived"
    assert _run_geo_export(
        derived_out, source_dir, model, formats=["geo_infer"], geo_derive_metadata=True
    )
    explicit_out = tmp_path / "explicit"
    assert _run_geo_export(
        explicit_out,
        source_dir,
        model,
        formats=["geo_infer"],
        geo_infer={"step_seconds": 60, "space_kind": "categorical"},
    )
    derived_artifact = _geo_artifact(derived_out, model)
    explicit_artifact = _geo_artifact(explicit_out, model)
    assert derived_artifact["time"] == explicit_artifact["time"]
    assert derived_artifact["space"] == explicit_artifact["space"]
    assert derived_artifact["dimensions"] == explicit_artifact["dimensions"]
    assert derived_artifact["matrices"] == explicit_artifact["matrices"]


# -- (c) Explicit beats derived --------------------------------------------------


def test_explicit_options_beat_derivation(tmp_path: Path) -> None:
    source_dir, model = _gridworld_with_time_step(tmp_path)
    output_dir = tmp_path / "out"
    assert _run_geo_export(
        output_dir,
        source_dir,
        model,
        formats=["geo_infer"],
        geo_infer={"step_seconds": 120},
        geo_derive_metadata=True,
    )
    artifact = _geo_artifact(output_dir, model)
    assert artifact["time"]["step_seconds"] == 120
    assert "metadata_derivation" not in artifact["provenance"]


# -- (d) Provenance lands in the artifact ----------------------------------------


def test_derivation_provenance_lands_in_artifact(tmp_path: Path) -> None:
    source_dir, model = _gridworld_with_time_step(tmp_path)
    output_dir = tmp_path / "out"
    assert _run_geo_export(
        output_dir, source_dir, model, formats=["geo_infer"], geo_derive_metadata=True
    )
    artifact = _geo_artifact(output_dir, model)
    record = artifact["provenance"]["metadata_derivation"]
    assert record["step_seconds"]["value"] == 60.0
    assert record["step_seconds"]["derived"] is True
    source = record["step_seconds"]["sources"][0]
    assert source["section"] == "Time"
    assert source["key"] == "timestep"
    assert source["raw_value"] == "60s"
    assert record["space_kind"] == {
        "value": "categorical",
        "derived": False,
        "source": "writer default",
    }
    assert record["units"]["derived"] is False
    assert record["time_index"] == {
        "variable": "t",
        "state_space_declaration": "t[1,type=int]",
        "ontology_term": "Time",
    }
    assert record["discrete_time_declared"] is True


def test_derivation_fills_partial_explicit_gaussian_options(tmp_path: Path) -> None:
    """Explicit model_type+units stay; derivation fills the missing seconds."""
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    model = source_dir / GAUSSIAN.name
    model.write_text(GAUSSIAN.read_text())
    options = {
        model.name: {"model_type": "linear_gaussian", "units": GAUSSIAN_UNITS}
    }
    output_dir = tmp_path / "out"
    assert _run_geo_export(
        output_dir,
        source_dir,
        model,
        formats=["geo_infer"],
        geo_infer_options=options,
        geo_derive_metadata=True,
    )
    artifact = _geo_artifact(output_dir, model)
    assert artifact["time"] == {"domain": "discrete", "step_seconds": 2}
    assert artifact["units"] == GAUSSIAN_UNITS
    record = artifact["provenance"]["metadata_derivation"]
    assert record["step_seconds"]["sources"][0]["section"] == "Time"
    assert record["space_kind"]["source"] == "writer default"


def test_explicit_options_cannot_claim_derivation(tmp_path: Path) -> None:
    source_dir, model = _gridworld_with_time_step(tmp_path)
    options = {
        model.name: {
            "step_seconds": 90,
            "metadata_derivation": {"step_seconds": {"value": 999}},
        }
    }
    output_dir = tmp_path / "out"
    assert _run_geo_export(
        output_dir,
        source_dir,
        model,
        formats=["geo_infer"],
        geo_infer_options=options,
        geo_derive_metadata=True,
    )
    artifact = _geo_artifact(output_dir, model)
    assert artifact["time"]["step_seconds"] == 90
    assert "metadata_derivation" not in artifact["provenance"]


# -- Derivation guard rails -------------------------------------------------------


def test_ambiguous_cross_section_declarations_rejected() -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    source = "## Time\nDynamic\nDiscrete\nTimeStep=60s\n\n## ModelParameters\ndt: 0.1\n"
    with pytest.raises(NotationMetadataError, match="ambiguous"):
        derive_geo_metadata(source)


def test_agreeing_cross_section_declarations_accepted() -> None:
    from gnn.export.notation_metadata import derive_geo_metadata

    source = "## Time\nDynamic\nDiscrete\nTimeStep=0.5\n\n## ModelParameters\ndt: 0.5\n"
    assert derive_geo_metadata(source)["options"] == {"step_seconds": 0.5}


@pytest.mark.parametrize(
    ("declaration", "match"),
    [
        ("TimeStep=5parsecs", "unsupported time unit"),
        ("TimeStep=0", "finite and positive"),
        ("TimeStep=abc", "cannot read time step value"),
        ("TimeStep=1\nTimeUnits=fortnights", "unsupported TimeUnits"),
    ],
)
def test_invalid_declarations_fail_visibly(declaration: str, match: str) -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    with pytest.raises(NotationMetadataError, match=match):
        derive_geo_metadata(f"## Time\nDynamic\nDiscrete\n{declaration}\n")


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("TimeStep=500ms", 0.5),
        ("TimeStep=1\nTimeUnits=minutes", 60.0),
        ("StepSeconds=2h", 7200.0),
        ("dt=90", 90.0),
    ],
)
def test_unit_interpretations(declaration: str, expected: float) -> None:
    from gnn.export.notation_metadata import derive_geo_metadata

    options = derive_geo_metadata(
        f"## Time\nDynamic\nDiscrete\n{declaration}\n"
    )["options"]
    assert options == {"step_seconds": expected}


def test_continuous_declaration_rejected() -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    with pytest.raises(NotationMetadataError, match="Continuous"):
        derive_geo_metadata("## Time\nDynamic\nContinuousTime=t\nTimeStep=60s\n")


def test_discrete_continuous_contradiction_rejected() -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    with pytest.raises(NotationMetadataError, match="contradiction"):
        derive_geo_metadata("## Time\nDiscrete\nContinuous\nTimeStep=60s\n")


def test_missing_time_section_rejected() -> None:
    from gnn.export.notation_metadata import NotationMetadataError, derive_geo_metadata

    with pytest.raises(NotationMetadataError, match="## Time section"):
        derive_geo_metadata("## ModelName\nNo time here\n")


def test_declared_h3_space_kind_flags_missing_ids() -> None:
    import tempfile

    from gnn.export.geo_infer import export_to_geo_infer
    from gnn.export.notation_metadata import derive_geo_metadata

    derived = derive_geo_metadata("## Time\nDiscrete\nTimeStep=60\nSpaceKind=h3\n")
    assert derived["options"] == {"step_seconds": 60.0, "space_kind": "h3"}
    with pytest.raises(ValueError, match="H3"):
        export_to_geo_infer(
            {"raw_content": GRIDWORLD.read_text(), "geo_infer": derived["options"]},
            Path(tempfile.mkdtemp()) / "h3.geo-infer.json",
        )


# -- Step 7 wiring -----------------------------------------------------------------


def test_step7_parser_accepts_derive_flag() -> None:
    from gnn.utils.arg_parsing import ArgumentParser

    parser = ArgumentParser.create_step_parser("7_export.py")
    parsed = parser.parse_args(["--geo-derive-metadata"])
    assert parsed.geo_derive_metadata is True
    unparsed = parser.parse_args([])
    assert unparsed.geo_derive_metadata is False


def test_step7_adapter_derive_without_options_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_step7_module()
    captured: dict[str, Any] = {}


    def fake_process_export(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(module, "process_export", fake_process_export)
    assert module._export_with_geo(
        target_dir=tmp_path, output_dir=tmp_path, geo_derive_metadata=True
    )
    assert captured["geo_derive_metadata"] is True
    assert "geo_infer" in captured["formats"]
    assert "geo_infer" not in captured
    assert captured.get("geo_infer_options_file") is None


def test_step7_adapter_derive_with_options_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_step7_module()
    captured: dict[str, Any] = {}


    def fake_process_export_cli(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(module, "process_export_cli", fake_process_export_cli)
    assert module._export_with_geo(
        target_dir=tmp_path,
        output_dir=tmp_path,
        geo_infer_options_file="opts.json",
        geo_derive_metadata=True,
    )
    assert captured["geo_infer_options_file"] == "opts.json"
    assert captured["geo_derive_metadata"] is True
    assert "geo_infer" in captured["formats"]


def test_gaussian_export_rejects_space_options() -> None:
    import tempfile

    from gnn.export.geo_infer import export_to_geo_infer

    with pytest.raises(ValueError, match="does not accept \\['space_kind'\\]"):
        export_to_geo_infer(
            {
                "raw_content": GAUSSIAN.read_text(),
                "geo_infer": {
                    "model_type": "linear_gaussian",
                    "step_seconds": 2,
                    "units": GAUSSIAN_UNITS,
                    "space_kind": "h3",
                },
            },
            Path(tempfile.mkdtemp()) / "g.geo-infer.json",
        )


def test_step7_adapter_without_flag_keeps_requirement() -> None:
    module = _load_step7_module()
    with pytest.raises(ValueError, match="step_seconds"):
        module._export_with_geo(geo_state_ids="ids.json")


def test_step7_adapter_derive_with_explicit_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_step7_module()
    captured: dict[str, Any] = {}


    def fake_process_export(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(module, "process_export", fake_process_export)
    assert module._export_with_geo(
        target_dir=tmp_path,
        output_dir=tmp_path,
        geo_step_seconds=1.5,
        geo_derive_metadata=True,
    )
    assert captured["geo_infer"]["step_seconds"] == 1.5
    assert captured["geo_derive_metadata"] is True
    assert "geo_infer" in captured["formats"]
