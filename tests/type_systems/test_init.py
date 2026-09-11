"""Smoke tests for the ``gnn.type_systems`` package surface.

The package currently contains only a docstring-only ``__init__.py`` (its
type-system artifacts are non-Python sources: ``scala.scala``, ``haskell.hs``,
and ``examples/``). These tests pin the import surface so the directory keeps
a collected test module and any future Python modules land beside them.
"""

from __future__ import annotations

import gnn.type_systems


class TestTypeSystemsPackage:
    def test_package_imports_cleanly(self) -> None:
        assert gnn.type_systems.__doc__ is not None
        assert "type system" in gnn.type_systems.__doc__.lower()

    def test_package_declares_no_runtime_exports(self) -> None:
        assert not hasattr(gnn.type_systems, "__all__")
