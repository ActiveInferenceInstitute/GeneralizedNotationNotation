"""Strict v2 extraction and explicit Step 7 metadata regressions."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from export.geo_infer import export_to_geo_infer
from export.geo_infer_gaussian import build_geo_infer_gaussian_artifact
from export.processor import process_export
from pipeline.config import get_output_dir_for_script

SOURCE = Path(__file__).with_name("gaussian_rectangular.md")
UNITS = dict(states=["m", "m/s", "K"], observations=["m", "m/s"], controls=["N"])


def test_explicit_rectangular_gaussian_export(tmp_path: Path) -> None:
    content = SOURCE.read_text()
    artifact = build_geo_infer_gaussian_artifact(content, step_seconds=2, units=UNITS)
    assert artifact["dimensions"] == dict(states=3, observations=2, controls=1)
    assert artifact["matrices"]["G"] == [[1.0], [2.0], [0.0]]
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(content.encode()).hexdigest()
    )
    dest = tmp_path / "gaussian.json"
    export_to_geo_infer(
        dict(
            raw_content=content,
            geo_infer=dict(model_type="linear_gaussian", step_seconds=2, units=UNITS),
        ),
        dest,
    )
    assert json.loads(dest.read_text()) == artifact


@pytest.mark.parametrize(
    "change",
    ["generator", "missing_G", "bad_units", "singular_R", "negative_Q", "boolean_F"],
)
def test_invalid_gaussian_not_repaired(change: str) -> None:
    content = SOURCE.read_text()
    units = UNITS
    if change == "generator":
        content = content.replace("\nDiscrete\n", "\nContinuous\n")
    elif change == "missing_G":
        content = content.replace("G={(1.0,),(2.0,),(0.0,)}", "")
    elif change == "bad_units":
        units = dict(UNITS, controls=[])
    elif change == "singular_R":
        content = content.replace("R={(0.5,0.0),(0.0,2.0)}", "R={(0.0,0.0),(0.0,2.0)}")
    elif change == "negative_Q":
        content = content.replace("Q={(0.1", "Q={(-0.1")
    elif change == "boolean_F":
        content = content.replace("F={(2.0", "F={(true")
    with pytest.raises(ValueError):
        build_geo_infer_gaussian_artifact(content, step_seconds=2, units=units)


def _pipeline(tmp_path: Path, sources: list[str]) -> tuple[Path, Path]:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    output = tmp_path / "7_export_output"
    stage3 = get_output_dir_for_script("3_gnn.py", tmp_path)
    stage3.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for name in sources:
        dest = source_dir / name
        dest.write_text(SOURCE.read_text())
        entries.append(dict(file_name=name, file_path=str(dest), parse_success=True))
    (stage3 / "gnn_processing_results.json").write_text(
        json.dumps(dict(processed_files=entries))
    )
    return source_dir, output


def test_pipeline_opt_in_uses_original_source(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    opts = dict(model_type="linear_gaussian", step_seconds=2, units=UNITS)
    assert process_export(
        source, output, formats=["geo_infer"], geo_infer_options={"a.md": opts}
    )
    manifest = json.loads((output / "export_results.json").read_text())
    result = manifest["files_exported"][0]["exports"]["geo_infer"]
    artifact = json.loads(Path(result["export_file"]).read_text())
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256((source / "a.md").read_bytes()).hexdigest()
    )
    assert artifact["time"]["step_seconds"] == 2


def test_pipeline_missing_one_models_metadata_fails_overall(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md", "b.md"])
    opts = dict(model_type="linear_gaussian", step_seconds=2, units=UNITS)
    assert not process_export(
        source, output, formats=["geo_infer"], geo_infer_options={"a.md": opts}
    )
    manifest = json.loads((output / "export_results.json").read_text())
    assert manifest["summary"]["successful_exports"] == 1
    assert manifest["summary"]["failed_exports"] == 1
    assert (
        "geo_infer_options"
        in manifest["files_exported"][1]["exports"]["geo_infer"]["error"]
    )


def test_default_pipeline_remains_five_formats(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    assert process_export(source, output)
    manifest = json.loads((output / "export_results.json").read_text())
    assert set(manifest["files_exported"][0]["exports"]) == {
        "json",
        "xml",
        "graphml",
        "gexf",
        "pickle",
    }


def test_gaussian_cli_roundtrip(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    units_file = tmp_path / "units.json"
    units_file.write_text(json.dumps(UNITS))
    artifact_file = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "export.geo_infer",
            str(SOURCE),
            str(artifact_file),
            "--step-seconds",
            "2",
            "--model-type",
            "linear_gaussian",
            "--units",
            str(units_file),
        ],
        env=dict(os.environ, PYTHONPATH=str(SOURCE.parents[2])),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(artifact_file.read_text())["dimensions"] == dict(
        states=3, observations=2, controls=1
    )


def test_pipeline_source_symlink_cannot_escape_target(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    outside = tmp_path / "outside.md"
    outside.write_text(SOURCE.read_text())
    (source / "a.md").unlink()
    (source / "a.md").symlink_to(outside)
    opts = dict(model_type="linear_gaussian", step_seconds=2, units=UNITS)
    assert not process_export(
        source, output, formats=["geo_infer"], geo_infer_options={"a.md": opts}
    )
    manifest = json.loads((output / "export_results.json").read_text())
    assert (
        "inside target_dir"
        in manifest["files_exported"][0]["exports"]["geo_infer"]["error"]
    )


def test_pipeline_categorical_opt_in(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    categorical = (
        SOURCE.parents[3] / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
    )
    (source / "a.md").write_text(categorical.read_text())
    assert process_export(
        source,
        output,
        formats=["geo_infer"],
        geo_infer_options={"a.md": dict(step_seconds=60)},
    )
    manifest = json.loads((output / "export_results.json").read_text())
    artifact = json.loads(
        Path(
            manifest["files_exported"][0]["exports"]["geo_infer"]["export_file"]
        ).read_text()
    )
    assert artifact["schema_version"] == "gnn-geo-infer/1"


@pytest.mark.parametrize(
    "unsafe_name", ["../escape.md", "/tmp/escape.md", r"..\escape.md"]
)
def test_pipeline_rejects_filename_traversal_before_output(
    tmp_path: Path, unsafe_name: str
) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    stage3 = get_output_dir_for_script("3_gnn.py", tmp_path)
    manifest_path = stage3 / "gnn_processing_results.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["processed_files"][0]["file_name"] = unsafe_name
    manifest_path.write_text(json.dumps(manifest))
    assert not process_export(source, output, formats=["geo_infer"])
    assert list(output.iterdir()) == []


def test_pipeline_rejects_output_directory_symlink_escape(tmp_path: Path) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    outside = tmp_path / "outside"
    outside.mkdir()
    output.mkdir()
    (output / "a").symlink_to(outside, target_is_directory=True)
    assert not process_export(source, output)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "link_name",
    ["a/a_geo_infer.geo-infer.json", "export_results.json", "export_summary.json"],
)
def test_pipeline_rejects_output_file_symlink_escape(
    tmp_path: Path, link_name: str
) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    outside = tmp_path / "outside.json"
    outside.write_text("preserve")
    destination = output / link_name
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside)
    opts = dict(model_type="linear_gaussian", step_seconds=2, units=UNITS)
    assert not process_export(
        source, output, formats=["geo_infer"], geo_infer_options={"a.md": opts}
    )
    assert outside.read_text() == "preserve"


@pytest.mark.parametrize(
    "before,after",
    [
        ("u[1,1", "u[99,1"),
        ("x[3,1", "x[99,1"),
        ("F[3,3", "F[99,99"),
        ("H[2,3", "H[3,2"),
        ("prior_mean[3", "prior_mean[99"),
    ],
)
def test_contradictory_source_dimensions_rejected(before: str, after: str) -> None:
    with pytest.raises(ValueError, match="declaration"):
        build_geo_infer_gaussian_artifact(
            SOURCE.read_text().replace(before, after), step_seconds=2, units=UNITS
        )


def test_additional_hidden_coordinate_rejected() -> None:
    content = SOURCE.read_text().replace(
        "x[3,1,type=float]", "x[3,1,type=float]\nz[3,1,type=float]"
    )
    with pytest.raises(ValueError, match="declaration"):
        build_geo_infer_gaussian_artifact(content, step_seconds=2, units=UNITS)


@pytest.mark.parametrize("field", ["Q", "R", "covariance"])
def test_extreme_indefinite_covariance_rejected(field: str) -> None:
    from export.geo_infer_gaussian import validate_gaussian_artifact

    data: dict[str, Any] = dict(
        schema_version="gnn-geo-infer/2",
        model_type="linear_gaussian",
        model_name="extreme",
        dimensions=dict(states=2, observations=2, controls=1),
        matrices=dict(
            F=[[1.0, 0.0], [0.0, 1.0]],
            G=[[1.0], [0.0]],
            H=[[1.0, 0.0], [0.0, 1.0]],
            Q=[[1.0, 0.0], [0.0, 1.0]],
            R=[[1.0, 0.0], [0.0, 1.0]],
        ),
        initial_belief=dict(mean=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 1.0]]),
        units=dict(states=["m", "m"], observations=["m", "m"], controls=["N"]),
        time=dict(domain="discrete", step_seconds=1),
        provenance=dict(producer="test", source_sha256="0" * 64),
    )
    container = data["initial_belief"] if field == "covariance" else data["matrices"]
    container[field] = [[-1e308, 0.0], [0.0, 1e308]]
    with pytest.raises(ValueError, match="positive"):
        validate_gaussian_artifact(data)


def test_pipeline_nested_same_basename_preserves_source_identity(
    tmp_path: Path,
) -> None:
    source, output = _pipeline(tmp_path, ["a.md"])
    entries: list[dict[str, Any]] = []
    options = {}
    for subdir, seconds in [("first", 2), ("second", 4)]:
        nested = source / subdir / "a.md"
        nested.parent.mkdir()
        nested.write_text(SOURCE.read_text() + f"\n# {subdir}\n")
        entries.append(
            dict(file_name="a.md", file_path=str(nested), parse_success=True)
        )
        options[f"{subdir}/a.md"] = dict(
            model_type="linear_gaussian", step_seconds=seconds, units=UNITS
        )
    stage3 = get_output_dir_for_script("3_gnn.py", tmp_path)
    (stage3 / "gnn_processing_results.json").write_text(
        json.dumps(dict(processed_files=entries))
    )
    assert process_export(
        source, output, formats=["geo_infer"], geo_infer_options=options
    )
    manifest = json.loads((output / "export_results.json").read_text())
    paths = [
        Path(entry["exports"]["geo_infer"]["export_file"])
        for entry in manifest["files_exported"]
    ]
    assert len(set(paths)) == 2
    for artifact_path, source_entry, seconds in zip(paths, entries, [2, 4]):
        artifact = json.loads(artifact_path.read_text())
        assert artifact["time"]["step_seconds"] == seconds
        assert (
            artifact["provenance"]["source_sha256"]
            == hashlib.sha256(Path(source_entry["file_path"]).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize(
    "extra", ["\n## Time\nContinuous\n", "\n## StateSpaceBlock\nx[99,1,type=float]\n"]
)
def test_duplicate_semantic_sections_rejected(extra: str) -> None:
    with pytest.raises(ValueError):
        build_geo_infer_gaussian_artifact(
            SOURCE.read_text() + extra, step_seconds=2, units=UNITS
        )


def test_integer_state_cannot_be_exported_as_gaussian() -> None:
    with pytest.raises(ValueError, match="type=float"):
        build_geo_infer_gaussian_artifact(
            SOURCE.read_text().replace("x[3,1,type=float]", "x[3,1,type=int]"),
            step_seconds=2,
            units=UNITS,
        )
