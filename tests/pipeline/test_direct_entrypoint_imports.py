"""Numbered scripts must start without incidental setuptools startup imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gnn.utils.arguments.arg_parsing import build_step_command_args
from gnn.utils.arguments.pipeline_arguments import PipelineArguments

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = sorted((ROOT / "src/gnn").glob("[0-9]*_*.py")) + [
    ROOT / "src/gnn/main.py"
]


@pytest.mark.parametrize("script", ENTRYPOINTS, ids=lambda path: path.name)
def test_direct_entrypoint_starts_without_setuptools_preload(
    script: Path, tmp_path: Path
) -> None:
    # Keep real site/coverage hooks while disabling the distutils shim that used
    # to preload types and hide this collision. Preserve instrumentation paths.
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-s", str(script), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_supervised_step_cannot_import_adjacent_types(tmp_path: Path) -> None:
    (tmp_path / "types.py").write_text(
        "raise RuntimeError('adjacent types was imported')\n"
    )
    script = tmp_path / "3_gnn.py"
    script.write_text(
        "import types, json; print(json.dumps({'stdlib_types': hasattr(types, 'GenericAlias')}))\n"
    )
    command = build_step_command_args(
        "3_gnn", PipelineArguments(), sys.executable, script
    )
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"stdlib_types": True}
