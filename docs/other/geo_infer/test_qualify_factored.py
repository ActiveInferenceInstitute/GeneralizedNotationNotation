from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("qualify_factored.py")
SOURCE = ROOT / "tests/export/factored_example.json"


def invoke(
    source: Path, artifact: Path, *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(artifact),
            "--step-seconds",
            "86400",
            *(["--check"] if check else []),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_export_and_read_only_reproduction(tmp_path: Path) -> None:
    artifact = tmp_path / "model.json"
    produced = invoke(SOURCE, artifact)
    assert produced.returncode == 0, produced.stderr
    initial = artifact.read_bytes()
    reproduced = invoke(SOURCE, artifact, check=True)
    assert reproduced.returncode == 0 and reproduced.stdout == produced.stdout
    assert artifact.read_bytes() == initial
    assert invoke(SOURCE, artifact).returncode != 0
    assert artifact.read_bytes() == initial


@pytest.mark.parametrize(
    "mutation", ["duplicate", "clock", "provenance", "oversized", "symlink"]
)
def test_reproduction_rejects_corruption(tmp_path: Path, mutation: str) -> None:
    artifact = tmp_path / "model.json"
    assert invoke(SOURCE, artifact).returncode == 0
    if mutation == "duplicate":
        artifact.write_text(
            artifact.read_text().replace("{", '{"schema_version":"ignored",', 1)
        )
    elif mutation == "oversized":
        artifact.write_bytes(artifact.read_bytes() + b" " * (4 * 1024 * 1024))
    elif mutation == "symlink":
        target = tmp_path / "target.json"
        artifact.rename(target)
        artifact.symlink_to(target)
    else:
        data = json.loads(artifact.read_bytes())
        if mutation == "clock":
            data["time"]["step_seconds"] = 1
        else:
            data["provenance"]["source_sha256"] = "0" * 64
        artifact.write_text(json.dumps(data))
    before = artifact.read_bytes()
    result = invoke(SOURCE, artifact, check=True)
    assert result.returncode != 0, result.stdout
    assert artifact.read_bytes() == before
