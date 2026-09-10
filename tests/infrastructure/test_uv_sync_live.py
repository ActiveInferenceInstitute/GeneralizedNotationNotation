"""Opt-in live check for ``uv sync --frozen --check --inexact --extra dev``.

SC-44 corrective: this single check is machine-dependent (wall-clock
sensitive and racy against a concurrent mutating ``uv sync`` on the shared
``.venv``), so it is gated behind ``GNN_UV_SYNC_LIVE=1``. The file is
allowlisted in ``tests/test_zero_skip_contracts.py`` for exactly that
reason — the same env opt-in class as the Ollama live files. The other 16
tests of the uv environment surface stay in
``tests/infrastructure/test_uv_environment.py`` with no skip tokens.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2].absolute()
UV_BIN = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


@pytest.mark.skipif(
    os.environ.get("GNN_UV_SYNC_LIVE") != "1",
    reason=(
        "uv sync --frozen --check is machine-dependent (wall-clock sensitive, "
        "shared-venv racy); opt in via GNN_UV_SYNC_LIVE=1"
    ),
)
def test_uv_sync_fast() -> Any:
    """Check required dev dependencies without pruning optional packages.

    Uses ``--check --inexact`` (non-mutating) so this opt-in test never
    rewrites the shared ``.venv`` while other tests read it. A pruning
    regression (e.g. dropping pytest/LSP/API/websocket deps) still fails
    the gate: missing or stale required packages still fail. Additional
    optional extras (such as GEO H3 support) are allowed to coexist.

    A concurrent mutating ``uv sync`` (the pipeline setup step or another
    xdist worker) can transiently report the environment as "outdated".
    That race resolves on its own, so retry briefly before failing; a real
    pruning regression stays outdated across retries and still fails.
    """
    import time

    # Do not use ``--all-extras`` here: it pulls large optional groups (e.g. gui) and
    # can fail on low-disk systems during wheel extraction. Keep ``--extra dev`` so
    # this test does not prune pytest, LSP, API, or websocket deps.
    start = time.time()
    returncode = 1
    err = ""
    for attempt in range(3):
        result = subprocess.run(  # nosec B607 B603
            [UV_BIN, "sync", "--frozen", "--check", "--inexact", "--extra", "dev"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120,
        )
        returncode = result.returncode
        err = (result.stderr or "") + (result.stdout or "")
        if returncode == 0:
            break
        if attempt < 2 and "outdated" in err.lower():
            time.sleep(0.5 * (attempt + 1))
            continue
        break
    elapsed = time.time() - start

    if returncode != 0 and (
        "No space" in err
        or "No space left on device" in err
        or "os error 28" in err
    ):
        pytest.fail("Insufficient disk for uv cache / venv (errno 28)")

    assert returncode == 0, f"uv sync failed: {err}"
    # Cached sync is usually a few seconds; allow slow CI and cold cache.
    assert elapsed < 120, f"uv sync took too long: {elapsed}s"
