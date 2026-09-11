"""Paired GEO-INFER revision pin and runner contract (GNN-side, GNN-04).

The committed pin ``.github/gnn-pair.json`` is consumed by the hosted
workflow (``.github/workflows/geo-infer-interchange.yml``) and by
``scripts/run_geo_interchange_checks.py``. A malformed or silently retargeted
pin must fail fast here in the export suite instead of surfacing as a
confusing hosted-CI failure, and the runner must refuse a checkout that is
not at the pinned revision before running any round trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = ROOT / ".github/gnn-pair.json"
RUNNER = ROOT / "scripts/run_geo_interchange_checks.py"
EXPECTED_REPOSITORY = "ActiveInferenceInstitute/GEO-INFER"


def _read_pin() -> dict[str, str]:
    return json.loads(PIN_FILE.read_text(encoding="utf-8"))


def test_pin_is_well_formed_and_targets_the_known_geo_repository() -> None:
    pin = _read_pin()
    assert set(pin) == {"repository", "revision"}
    assert pin["repository"] == EXPECTED_REPOSITORY
    revision = pin["revision"]
    assert isinstance(revision, str)
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)


def _init_repo_with_commit(directory: Path, message: str) -> str:
    directory.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
    subprocess.run(
        ["git", "-C", str(directory), "config", "user.email", "pin-test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(directory), "config", "user.name", "pin test"], check=True
    )
    (directory / "README.md").write_text("pin test\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(directory), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(directory), "commit", "-q", "-m", message], check=True
    )
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_runner_refuses_checkout_that_is_not_at_the_pinned_revision(
    tmp_path: Path,
) -> None:
    """Exit code 2, no receipts, before any validator invocation."""
    mismatched = _init_repo_with_commit(tmp_path / "geo", "not the pinned revision")
    pin = _read_pin()
    assert mismatched != pin["revision"], "test fixture must not equal the pin"
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(pin), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--pin-file",
            str(pin_file),
            "--geo-root",
            str(tmp_path / "geo"),
            "--receipts-dir",
            str(tmp_path / "receipts"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
    assert pin["revision"] in completed.stderr
    assert mismatched in completed.stderr
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    "broken",
    [
        {"repository": "hum-lab/geo-infer", "revision": "a" * 40},
        {"repository": EXPECTED_REPOSITORY, "revision": "short"},
        {"repository": EXPECTED_REPOSITORY, "revision": "z" * 40},
        {"repository": EXPECTED_REPOSITORY},
    ],
)
def test_runner_rejects_malformed_pins(tmp_path: Path, broken: dict[str, str]) -> None:
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps(broken), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--pin-file",
            str(pin_file),
            "--geo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
