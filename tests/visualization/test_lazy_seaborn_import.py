#!/usr/bin/env python3
"""Import-cost guards: visualization packages must not import seaborn eagerly.

``gnn/visualization/compat/viz_compat.py`` resolves ``sns`` lazily (PEP 562 +
``get_sns``), so importing the visualization packages must never pull seaborn
(or its transitive scipy/pandas/IPython tree) into ``sys.modules``. Seaborn
loads only when a consumer actually asks for it.

Each check runs in a fresh subprocess so a sibling test importing seaborn
cannot pollute ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every package whose import graph used to reach the eager seaborn import in
# viz_compat (directly or through analysis.viz_base / matrix.visualizer).
LAZY_MODULES: list[str] = [
    "gnn.visualization",
    "gnn.advanced_visualization",
    "gnn.analysis",
    "gnn.llm",
]


@pytest.mark.parametrize("module_name", LAZY_MODULES)
def test_import_does_not_load_seaborn(module_name: str) -> None:
    """Importing a visualization package must not load seaborn."""
    code = (
        "import sys; import "
        f"{module_name}; assert 'seaborn' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_import_does_not_load_scipy_stats() -> None:
    """analysis_statistics must not import scipy.stats at module load."""
    code = (
        "import sys; import gnn.analysis; "
        "assert 'scipy.stats' not in sys.modules, sorted(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "import_stmt,var_name",
    [
        ("from gnn.visualization.compat.viz_compat import sns", "sns"),
        ("from gnn.visualization.compat import sns", "sns"),
        ("from gnn.visualization._viz_compat import sns", "sns"),
        ("from gnn.analysis.viz_base import sns", "sns"),
        ("from gnn.analysis.simulation_visualizations import sns", "sns"),
        ("from gnn.analysis.analysis_statistics import stats", "stats"),
        ("from gnn.advanced_visualization._shared import sns", "sns"),
    ],
)
def test_lazy_attribute_consumers_still_bind(import_stmt: str, var_name: str) -> None:
    """Documented consumers of the lazy aliases keep working."""
    code = f"{import_stmt}; print(type({var_name}).__name__)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in ("module", "NoneType")
