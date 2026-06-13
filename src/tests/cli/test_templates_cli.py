from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cli import main
from cli.templates import (
    _template_record_from_raw,
    list_templates,
    pull_template,
    show_template,
)


def test_template_index_contains_documented_pull_target() -> None:
    names = {template["name"] for template in list_templates()}
    assert "actinf-pomdp-2state" in names
    assert "pomdp-gridworld-3x3" in names


def test_template_index_is_externalized_and_has_three_entries() -> None:
    templates = list_templates()
    assert len(templates) >= 3
    for template in templates:
        assert template["source"].startswith("package://")
        assert Path(template["source"]).suffix == ".md"
        assert len(template["sha256"]) == 64


@pytest.mark.parametrize(
    "template_path",
    sorted(
        (Path(__file__).resolve().parents[3] / "src" / "cli" / "template_assets").glob(
            "*.md"
        )
    ),
    ids=lambda path: path.name,
)
def test_packaged_templates_pass_strict_cli_validation(
    template_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", str(template_path), "--strict"]) == 0
    captured = capsys.readouterr()
    assert "valid" in captured.out


@pytest.mark.parametrize(
    "record",
    [
        {
            "name": "bad",
            "description": "bad",
            "source": "template_assets/actinf_pomdp_2state.md",
            "filename": "../escape.md",
        },
        {
            "name": "bad",
            "description": "bad",
            "source": "template_assets/actinf_pomdp_2state.md",
            "filename": "/tmp/escape.md",
        },
        {
            "name": "bad",
            "description": "bad",
            "source": "../input/gnn_files/demo.md",
            "filename": "demo.md",
        },
        {
            "name": "bad",
            "description": "bad",
            "source": "input/gnn_files/demo.md",
            "filename": "demo.md",
        },
        {
            "name": "bad",
            "description": "bad",
            "source": "template_assets/demo.txt",
            "filename": "demo.md",
        },
    ],
)
def test_template_index_rejects_paths_outside_package_contract(
    record: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        _template_record_from_raw(record)


def test_show_template_returns_gridworld_record() -> None:
    template = show_template("pomdp-gridworld-3x3")
    assert template["filename"] == "pomdp_gridworld_3x3.md"
    assert template["source"].endswith("pomdp_gridworld_3x3.md")


def test_pull_template_dry_run_does_not_copy(tmp_path: Path) -> None:
    result = pull_template("actinf-pomdp-2state", tmp_path, dry_run=True)
    assert result["dry_run"] is True
    assert result["copied"] is False
    assert not Path(result["destination"]).exists()


def test_pull_template_copies_with_checksum(tmp_path: Path) -> None:
    result = pull_template("actinf-pomdp-2state", tmp_path)
    destination = Path(result["destination"])
    assert destination.exists()
    assert result["copied"] is True
    assert len(result["sha256"]) == 64


def test_pull_template_rejects_checksum_collision_without_overwrite(
    tmp_path: Path,
) -> None:
    first = pull_template("actinf-pomdp-2state", tmp_path)
    destination = Path(first["destination"])
    destination.write_text("different content\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        pull_template("actinf-pomdp-2state", tmp_path)


def test_pull_template_overwrite_replaces_checksum_collision(tmp_path: Path) -> None:
    first = pull_template("actinf-pomdp-2state", tmp_path)
    destination = Path(first["destination"])
    destination.write_text("different content\n", encoding="utf-8")

    result = pull_template("actinf-pomdp-2state", tmp_path, overwrite=True)

    assert result["overwritten"] is True
    assert result["copied"] is True
    assert len(result["sha256"]) == 64


def test_pull_template_rejects_symlink_destination(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    destination = tmp_path / "actinf_pomdp_2state.md"
    try:
        destination.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")

    with pytest.raises(FileExistsError, match="symlink"):
        pull_template("actinf-pomdp-2state", tmp_path, dry_run=True)

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_templates_list_cli_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["templates", "list"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["templates"]


def test_pull_cli_dry_run_outputs_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            ["pull", "actinf-pomdp-2state", "--output-dir", str(tmp_path), "--dry-run"]
        )
        == 0
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["template"] == "actinf-pomdp-2state"
    assert payload["dry_run"] is True


def test_templates_show_cli_outputs_one_template(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["templates", "show", "pomdp-gridworld-3x3"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["template"]["name"] == "pomdp-gridworld-3x3"


def test_pull_cli_unknown_template_fails() -> None:
    assert main(["pull", "missing-template", "--dry-run"]) == 1


@pytest.mark.slow
def test_template_cli_works_from_installed_wheel_outside_repo(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv executable is required for wheel smoke")

    repo_root = Path(__file__).resolve().parents[3]
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    wheel = next(dist_dir.glob("*.whl"))

    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    pip = bin_dir / ("pip.exe" if sys.platform == "win32" else "pip")
    gnn = bin_dir / ("gnn.exe" if sys.platform == "win32" else "gnn")
    subprocess.run(
        [str(pip), "install", "--no-deps", str(wheel)],
        check=True,
        text=True,
        capture_output=True,
    )

    outside_repo = tmp_path / "outside"
    outside_repo.mkdir()
    list_result = subprocess.run(
        [str(gnn), "templates", "list"],
        cwd=outside_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "pomdp-gridworld-3x3" in list_result.stdout

    show_result = subprocess.run(
        [str(gnn), "templates", "show", "pomdp-gridworld-3x3"],
        cwd=outside_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "pomdp_gridworld_3x3.md" in show_result.stdout

    pull_result = subprocess.run(
        [
            str(gnn),
            "pull",
            "pomdp-gridworld-3x3",
            "--output-dir",
            str(tmp_path / "pulled"),
            "--dry-run",
        ],
        cwd=outside_repo,
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"copied": false' in pull_result.stdout
