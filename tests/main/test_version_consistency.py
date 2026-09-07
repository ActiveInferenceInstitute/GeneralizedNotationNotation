"""Version consistency across the ``gnn`` package surfaces.

``pyproject.toml`` is the single source of truth; every ``__version__``
literal and the FastAPI app metadata must agree with it.
"""

from __future__ import annotations

import importlib.metadata


def test_single_version() -> None:
    import gnn
    import gnn.api as api
    import gnn.cli as cli
    expected = "3.3.0"
    assert gnn.__version__ == expected
    assert gnn.cli.__version__ == expected
    assert gnn.api.__version__ == expected
    assert gnn.api.MODULE_VERSION == expected
    assert importlib.metadata.version("generalized-notation-notation") == expected
