"""Light-import regression test for the ``gnn.utils`` facade (S2-33 Step 0).

Design §5 Step 0: ``import gnn.utils`` must not execute any submodule, so
heavy module-scope dependencies (psutil via structured_logging /
resource_manager, matplotlib via simulation_utils) stay deferred until an
exported name actually resolves through ``__getattr__``. The check runs in a
subprocess because the pytest process itself imports psutil through the test
stack before this test executes.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast


def test_import_gnn_utils_does_not_import_heavy_dependencies() -> None:
    """``import gnn.utils`` pulls neither psutil nor matplotlib into sys.modules."""
    code = (
        "import gnn.utils, sys\n"
        "heavy = sorted(\n"
        "    name\n"
        "    for name in sys.modules\n"
        "    if name in ('psutil', 'matplotlib')\n"
        "    or name.startswith('psutil.')\n"
        "    or name.startswith('matplotlib.')\n"
        ")\n"
        "assert not heavy, f'import gnn.utils executed heavy modules: {heavy}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"import gnn.utils is no longer light (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
