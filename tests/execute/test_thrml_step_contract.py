"""Generic Step 12 must enforce the same fresh THRML artifact boundary."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gnn.execute import processor
from gnn.execute.processor.single import execute_single_script
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus
from tests.helpers.thrml_results import write_protocol_script


def _execute(script: Path, output: Path) -> dict:
    return execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "thrml",
            "model_name": "stable_model",
            "executor": "python",
        },
        output,
        False,
        logging.getLogger(__name__),
        timeout=5,
    )


@pytest.fixture
def protocol_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    # Protocol-only children are genuine processes, without a native inference
    # claim. Released-library acceptance is a separate opt-in lane.
    monkeypatch.setattr(
        processor,
        "_check_framework_by_name",
        lambda *args, **kwargs: FrameworkStatus("thrml", True),
    )


def test_successful_child_requires_fresh_native_identity(
    tmp_path: Path, protocol_ready: None
) -> None:
    folder = tmp_path / "render" / "stable_model" / "thrml"
    folder.mkdir(parents=True)
    script = write_protocol_script(folder / "case_thrml.py", once=True)
    first = _execute(script, tmp_path / "out")
    assert first["success"], first
    assert first["scientific_analysis"]["inference_mode"] == "gibbs_monte_carlo"
    second = _execute(script, tmp_path / "out")
    assert not second["success"] and second["status"] != "success"
    assert "identity" in second["error"]
    assert (
        tmp_path / "out/stable_model/thrml/simulation_data/simulation_results.json"
    ).exists()


def test_successful_exit_without_artifact_is_unsuccessful(
    tmp_path: Path, protocol_ready: None
) -> None:
    folder = tmp_path / "render" / "stable_model" / "thrml"
    folder.mkdir(parents=True)
    script = folder / "missing_thrml.py"
    script.write_text("print('finished without result')")
    result = _execute(script, tmp_path / "out")
    assert not result["success"] and "native result" in result["error"]


def test_relative_output_root_survives_child_working_directory(
    tmp_path, protocol_ready, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "render/stable_model/thrml"
    folder.mkdir(parents=True)
    script = write_protocol_script(folder / "relative_thrml.py")
    result = _execute(script, Path("out"))
    assert result["success"], result
    assert (
        tmp_path / "out/stable_model/thrml/simulation_data/simulation_results.json"
    ).is_file()
    assert not (folder / "out").exists()
    assert Path(result["implementation_directory"]).is_absolute()
    assert Path(result["structured_result_file"]).is_absolute()


@pytest.mark.parametrize(
    "framework, variable",
    [
        ("pymdp", "PYMDP_OUTPUT_DIR"),
        ("jax", "GNN_OUTPUT_DIR"),
        ("numpyro", "NUMPYRO_OUTPUT_DIR"),
        ("pytorch", "PYTORCH_OUTPUT_DIR"),
        ("stan", "STAN_OUTPUT_DIR"),
        ("bnlearn", "BNLEARN_OUTPUT_DIR"),
        ("ngclearn", "NGCLEARN_OUTPUT_DIR"),
        ("cpomdp", "CPOMDP_OUTPUT_DIR"),
        ("thrml", "THRML_OUTPUT_DIR"),
    ],
)
def test_all_backend_output_environments_are_absolute(
    tmp_path, monkeypatch, framework, variable
):
    from gnn.execute.types import ScriptExecutionContext

    monkeypatch.chdir(tmp_path)
    context = ScriptExecutionContext(
        tmp_path / "render/case.py", "case.py", framework, "stable_model", "python"
    )
    env = processor._build_execution_environment(context, Path("out"))
    expected = tmp_path / "out/stable_model" / framework
    if framework != "thrml":
        expected /= "simulation_data"
    assert Path(env[variable]) == expected
    assert expected.is_absolute()


def test_pymdp_environment_preserves_explicit_checkout_override(tmp_path, monkeypatch):
    from gnn.execute.types import ScriptExecutionContext

    monkeypatch.setenv("GNN_PROJECT_ROOT", str(tmp_path / "explicit-checkout"))
    context = ScriptExecutionContext(
        tmp_path / "case.py", "case.py", "pymdp", "model", "python"
    )
    env = processor._build_execution_environment(context, tmp_path / "out")
    assert env["GNN_PROJECT_ROOT"] == str(tmp_path / "explicit-checkout")
