"""Pins for ``utils/system_env/venv_utils.get_venv_python`` (previously untested)."""

from __future__ import annotations

import sys
from pathlib import Path

from gnn.utils.system_env.venv_utils import get_venv_python


def _make_venv(root: Path, style: str = "posix") -> Path:
    venv = root / ".venv"
    python = (
        venv / "Scripts" / "python.exe"
        if style == "windows"
        else venv / "bin" / "python"
    )
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return venv


def test_finds_project_root_venv_python_and_site_packages(tmp_path: Path) -> None:
    script_dir = tmp_path / "src"
    script_dir.mkdir()
    venv = _make_venv(script_dir.parent)

    python, site_packages = get_venv_python(script_dir)

    assert python == venv / "bin" / "python"
    assert site_packages is not None
    assert site_packages.name == "site-packages"


def test_windows_style_python_executable_is_found(tmp_path: Path) -> None:
    script_dir = tmp_path / "src"
    script_dir.mkdir()
    _make_venv(script_dir.parent, style="windows")

    python, _ = get_venv_python(script_dir)

    assert python is not None
    assert python.name == "python.exe"


def test_recovers_venv_inside_script_directory(tmp_path: Path) -> None:
    script_dir = tmp_path / "src"
    script_dir.mkdir()
    _make_venv(script_dir)

    python, _ = get_venv_python(script_dir)

    assert python == script_dir / ".venv" / "bin" / "python"


def test_no_venv_falls_back_to_current_interpreter(tmp_path: Path) -> None:
    script_dir = tmp_path / "src"
    script_dir.mkdir()

    python, site_packages = get_venv_python(script_dir)

    assert python == Path(sys.executable)
    assert site_packages is None


def test_non_executable_venv_dir_is_skipped(tmp_path: Path) -> None:
    script_dir = tmp_path / "src"
    script_dir.mkdir()
    # .venv exists but contains no python executable: not a usable venv.
    (script_dir.parent / ".venv").mkdir()

    python, _ = get_venv_python(script_dir)

    assert python == Path(sys.executable)
