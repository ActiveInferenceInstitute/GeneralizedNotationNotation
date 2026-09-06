"""Toolchain-guarded test for the Lean verification runner receipt flow.

Skipped unless the fep_lean checkout resolves, ``lake`` is on PATH, and the
bridge exposes the ``verify-document`` operation.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import pytest

from gnn.execute.lean.lean_runner import (
    FEP_LEAN_ROOT_ENV,
    resolve_fep_lean_root,
    run_lean_scripts,
)


def _fep_lean_root() -> Path | None:
    return resolve_fep_lean_root()


@functools.cache
def _bridge_has_verify_document(root_str: str) -> bool:
    try:
        completed = subprocess.run(
            ["uv", "run", "fep-lean", "bridge", "--help"],
            cwd=root_str,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "verify-document" in completed.stdout


LEAN_TOOLCHAIN_AVAILABLE = bool(
    shutil.which("lake")
    and (root := _fep_lean_root()) is not None
    and _bridge_has_verify_document(str(root))
)

pytestmark = pytest.mark.skipif(
    not LEAN_TOOLCHAIN_AVAILABLE,
    reason="fep_lean toolchain or bridge verify-document op unavailable",
)


def test_lean_runner_receipt_status_ok(tmp_path: Path) -> None:
    """run_lean_scripts verifies a known-good emitted document to status ok."""
    root = _fep_lean_root()
    assert root is not None

    fixture = (
        root
        / "specs"
        / "gnn-bridge-p1-finite-spike"
        / "gnn-input"
        / "FepLeanSymmetricBool.md"
    )
    assert fixture.is_file(), "known-good finite fixture missing from fep_lean"

    target = tmp_path / "target"
    target.mkdir()
    (target / fixture.name).write_text(fixture.read_text(encoding="utf-8"))

    output_dir = tmp_path / "lean"
    ok = run_lean_scripts(target, output_dir)
    assert ok, "lean verification of the known-good fixture failed"

    receipt = output_dir / f"{fixture.stem}-receipt.json"
    assert receipt.is_file(), "receipt was not written"
    import json

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"


def test_lean_runner_skips_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FEP_LEAN_ROOT_ENV, "/nonexistent/fep_lean")
    target = tmp_path / "target"
    target.mkdir()
    assert run_lean_scripts(target, tmp_path / "lean") is False
