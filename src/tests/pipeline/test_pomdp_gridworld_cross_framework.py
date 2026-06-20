#!/usr/bin/env python3
"""Strict GridWorld POMDP contract tests across PyMDP and Julia backends."""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from analysis.processor import process_analysis
from execute.processor import process_execute
from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.processor import process_render, render_gnn_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
GRIDWORLD_DIR = REPO_ROOT / "input" / "gnn_files" / "pomdp_gridworld"
GRIDWORLD_FILE = GRIDWORLD_DIR / "pomdp_gridworld_3x3.md"
FRAMEWORKS = ("pymdp", "rxinfer", "activeinference_jl")
SCHEMAS = {
    "pymdp": "pymdp_simulation_v1",
    "rxinfer": "rxinfer_simulation_v1",
    "activeinference_jl": "activeinference_jl_simulation_v1",
}


def _gridworld_spec() -> dict[str, Any]:
    pomdp = extract_pomdp_from_file(GRIDWORLD_FILE, strict_validation=True)
    assert pomdp is not None
    return POMDPRenderProcessor(GRIDWORLD_DIR)._pomdp_to_gnn_spec(pomdp)


def _assert_julia_packages() -> None:
    cmd = [
        "julia",
        "--startup-file=no",
        "-e",
        "using RxInfer, ActiveInference, JSON, Distributions, StatsBase",
    ]
    try:
        result = subprocess.run(  # nosec B603 B607
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise AssertionError(
            f"Julia executable not available for strict backend gate: {exc}"
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "Julia package gate timed out before backend execution; "
            f"command={cmd!r} timeout={exc.timeout}s"
        )
    if (
        result.returncode != 0
        and "Package " in result.stderr
        and " not found" in result.stderr
    ):
        raise AssertionError(
            "Optional Julia backend packages are not installed for strict "
            f"cross-framework execution: {result.stderr.strip()}"
        )
    assert result.returncode == 0, (
        "Strict Julia package gate failed:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def _assert_julia_parse(script: Path) -> None:
    result = subprocess.run(  # nosec B603 B607
        [
            "julia",
            "--startup-file=no",
            "-e",
            f'Meta.parseall(read("{script}", String)); println("parsed")',
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Julia parse failed for {script}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_julia_package_gate_fails_missing_optional_backend_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["julia"],
            returncode=1,
            stdout="",
            stderr=(
                "ERROR: ArgumentError: Package RxInfer not found in current path.\n"
                '- Run `import Pkg; Pkg.add("RxInfer")` to install the RxInfer package.'
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AssertionError, match="Optional Julia backend packages"):
        _assert_julia_packages()


def _payload_for(exec_out: Path, framework: str) -> dict[str, Any]:
    matches = sorted(
        exec_out.glob(f"**/{framework}/simulation_data/*simulation_results.json")
    )
    assert matches, f"No collected simulation_results.json for {framework}"
    return cast(dict[str, Any], json.loads(matches[0].read_text(encoding="utf-8")))


@pytest.mark.integration
def test_gridworld_extraction_uses_canonical_dimensions() -> None:
    spec = _gridworld_spec()
    init = spec["initialparameterization"]

    assert spec["model_parameters"]["num_hidden_states"] == 9
    assert spec["model_parameters"]["num_obs"] == 9
    assert spec["model_parameters"]["num_actions"] == 5
    assert spec["model_parameters"]["num_timesteps"] == 15
    assert spec["model_parameters"]["random_seed"] == 42

    assert np.asarray(init["A"], dtype=float).shape == (9, 9)
    b_matrix = np.asarray(init["B"], dtype=float)
    assert b_matrix.shape == (9, 9, 5)
    np.testing.assert_allclose(b_matrix.sum(axis=0), np.ones((9, 5)))
    assert np.asarray(init["C"], dtype=float).shape == (9,)
    assert np.asarray(init["D"], dtype=float).shape == (9,)
    assert np.asarray(init["E"], dtype=float).shape == (5,)
    assert spec["matrix_provenance"]["B"]["canonical_order"] == (
        "next_state_previous_state_action"
    )


@pytest.mark.integration
def test_gridworld_render_helpers_use_canonical_framework_renderers(
    tmp_path: Path,
) -> None:
    spec = _gridworld_spec()
    for framework in FRAMEWORKS:
        output_dir = tmp_path / framework
        ok, message, artifacts = render_gnn_spec(spec, framework, output_dir)
        assert ok, f"{framework} render failed: {message}"
        assert artifacts
        rendered = Path(artifacts[0])
        assert rendered.exists()
        text = rendered.read_text(encoding="utf-8")
        assert "POMDP GridWorld 3x3" in text
        assert "next_state_previous_state_action" in text
        if framework in {"rxinfer", "activeinference_jl"}:
            assert "simulation_results.json" in text
            _assert_julia_parse(rendered)


@pytest.mark.integration
def test_gridworld_fixture_directory_renders_only_model_sources(
    tmp_path: Path,
) -> None:
    render_out = tmp_path / "render"
    render_ok = process_render(
        GRIDWORLD_DIR,
        render_out,
        verbose=False,
        frameworks=list(FRAMEWORKS),
        strict_validation=True,
        strict_framework_success=True,
    )
    assert render_ok

    summary = json.loads(
        (render_out / "render_processing_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total_files"] == 1
    assert summary["successful_files"] == 1
    assert summary["failed_framework_renderings"] == []
    assert sorted(summary["file_results"]) == [str(GRIDWORLD_FILE)]


@pytest.mark.pipeline
@pytest.mark.integration
@pytest.mark.slow
def test_gridworld_render_execute_analyze_visualize_strict(tmp_path: Path) -> None:
    _assert_julia_packages()

    input_dir = tmp_path / "input" / "gnn_files" / "pomdp_gridworld"
    input_dir.mkdir(parents=True)
    shutil.copy2(GRIDWORLD_FILE, input_dir / GRIDWORLD_FILE.name)

    base = tmp_path / "output"
    render_out = base / "11_render_output"
    exec_out = base / "12_execute_output"
    analysis_out = base / "16_analysis_output"

    render_ok = process_render(
        input_dir,
        render_out,
        verbose=False,
        frameworks=list(FRAMEWORKS),
        strict_validation=True,
        strict_framework_success=True,
    )
    assert render_ok

    for framework in FRAMEWORKS:
        scripts = sorted((render_out / GRIDWORLD_FILE.stem / framework).glob("*"))
        assert scripts, f"No rendered script for {framework}"
        if framework in {"rxinfer", "activeinference_jl"}:
            _assert_julia_parse(scripts[0])

    exec_ok = process_execute(
        input_dir,
        exec_out,
        verbose=False,
        frameworks="pymdp,rxinfer,activeinference_jl",
        render_output_dir=render_out,
        timeout=240,
    )
    assert exec_ok

    payloads = {
        framework: _payload_for(exec_out, framework) for framework in FRAMEWORKS
    }
    pymdp_provenance = payloads["pymdp"].get("matrix_provenance", {})
    for framework, payload in payloads.items():
        assert payload.get("schema_version") == SCHEMAS[framework]
        assert payload.get("success") is True
        assert payload.get("num_timesteps") == 15
        assert payload.get("model_parameters", {}).get("B_shape") == [9, 9, 5]
        assert payload.get("validation", {}).get("all_valid") is True
        assert len(payload.get("observations", [])) == 15
        assert len(payload.get("actions", [])) == 15
        assert len(payload.get("beliefs", [])) == 15
        assert payload.get("matrix_provenance", {}) == pymdp_provenance

    analysis_ok = process_analysis(input_dir, analysis_out, verbose=False)
    assert analysis_ok
    visualizations = list(analysis_out.rglob("*.png"))
    assert visualizations, "Step 16 should create visualizations"
    cross_framework = (
        analysis_out / "cross_framework" / "cross_framework_comparison.png"
    )
    assert cross_framework.exists()
    gifs = list(analysis_out.rglob("*.gif"))
    assert gifs, "Step 16 should create GridWorld GIF animations"
    cross_framework_gif = (
        analysis_out
        / "cross_framework"
        / "gridworld_animations"
        / "gridworld_cross_framework_trajectory.gif"
    )
    assert cross_framework_gif.exists()
    assert cross_framework_gif.stat().st_size > 0
    for framework in FRAMEWORKS:
        assert list((analysis_out / framework).glob("*.png")), (
            f"Step 16 should create {framework} visualizations"
        )
        framework_gifs = list(
            (analysis_out / "cross_framework" / "gridworld_animations").glob(
                f"*_{framework}_*.gif"
            )
        )
        assert len(framework_gifs) >= 2, (
            f"Step 16 should create belief and trajectory GIFs for {framework}"
        )
        assert all(path.stat().st_size > 0 for path in framework_gifs)

    manifest_path = (
        analysis_out / "cross_framework" / "gridworld_analysis_manifest.json"
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "gridworld_analysis_manifest_v1"
    assert sorted(manifest["frameworks"]) == sorted(FRAMEWORKS)
    assert manifest["matrix_provenance_equal"] is True
    assert len(manifest["outputs"]["gif"]) >= 7
    assert manifest["outputs"]["png"]
    assert manifest["outputs"]["statistics"]
