"""Isolation acceptance: POMDP extraction works with heavy dependencies blocked.

Cluster 2.1 acceptance test — with every heavy pipeline dependency denied at
import time (meta_path blocklist; numpy explicitly allowed), both package
entry points must import and the extractor must fully parse matrices from the
canonical discrete exemplar instead of silently degrading.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
ACTINF_EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"

# Heavy dependencies that must NOT be imported by the light extraction path.
# numpy is deliberately absent: numeric fallbacks may use it.
BLOCKED_ROOTS = frozenset(
    {
        "psutil",
        "matplotlib",
        "networkx",
        "jax",
        "pymdp",
        "openai",
        "ollama",
        "httpx",
        "aiohttp",
        "pandas",
        "plotly",
        "seaborn",
        "h5py",
        "numpyro",
        "discopy",
        "flax",
        "equinox",
    }
)


class _HeavyDepBlocker(MetaPathFinder):
    """meta_path finder that raises ImportError for blocked dependency roots."""

    def __init__(self, blocked: frozenset[str]) -> None:
        self.blocked = blocked

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(
                f"No module named {fullname!r} (blocked by isolation test)"
            )
        return None


def _purge_packages(roots: frozenset[str]) -> dict[str, ModuleType]:
    """Remove ``roots`` packages (and gnn/utils) from sys.modules, returning them."""
    removed: dict[str, ModuleType] = {}
    for name in list(sys.modules):
        if name.split(".")[0] in roots or name.split(".")[0] in {"gnn", "utils"}:
            removed[name] = sys.modules.pop(name)
    return removed


def test_extract_pomdp_from_file_imports_and_extracts_with_heavy_deps_blocked() -> None:
    """THE 2.1 acceptance test: light extraction path needs no heavy deps."""
    blocker = _HeavyDepBlocker(BLOCKED_ROOTS)
    previously_loaded = _purge_packages(BLOCKED_ROOTS)
    sys.meta_path.insert(0, blocker)
    try:
        from gnn.pomdp_extractor import extract_pomdp_from_file
        from utils.safe_eval import MATRIX_MAX_LEN, safe_literal_eval

        assert MATRIX_MAX_LEN > 0
        assert safe_literal_eval("(1, 2)") == (1, 2)

        spec = extract_pomdp_from_file(ACTINF_EXEMPLAR, strict_validation=True)
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        # Drop anything half-imported under the blocklist so later tests get a
        # clean re-import, then restore modules that existed before the purge.
        for name in list(sys.modules):
            if name.split(".")[0] in {"gnn", "utils"}:
                sys.modules.pop(name, None)
        sys.modules.update(previously_loaded)

    assert spec is not None
    assert not isinstance(spec, tuple)
    assert spec.A_matrix is not None, (
        "A_matrix dropped: extraction silently degraded under import blocklist"
    )
    assert spec.B_matrix is not None, (
        "B_matrix dropped: extraction silently degraded under import blocklist"
    )


def test_blocklist_only_strikes_declared_roots() -> None:
    """The blocker itself must not interfere with numpy or stdlib imports."""
    import numpy

    blocker = _HeavyDepBlocker(BLOCKED_ROOTS)
    sys.meta_path.insert(0, blocker)
    try:
        assert numpy.zeros(3).sum() == 0.0
        import json as _json

        assert _json.loads("[1]") == [1]
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)


def test_exemplar_fixture_exists() -> None:
    """Guard the fixture path so acceptance never silently no-ops."""
    assert ACTINF_EXEMPLAR.is_file(), f"missing exemplar: {ACTINF_EXEMPLAR}"
