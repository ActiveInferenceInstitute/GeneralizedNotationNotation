"""Single parametrized import-smoke suite over the GNN module registry.

Replaces the per-module copy-pasted ``test_module_imports`` /
``test_module_importable`` / ``test_get_module_info`` clone blocks (previously
duplicated across 6+ test files). Coverage parity: the registry is the union of
every module path the deleted clones exercised — 28 params replace the 27
deleted functions, covering 28 distinct module paths (a superset of the 23
distinct paths the clones touched).
"""

from __future__ import annotations

import pytest

from tests.helpers_import_smoke import MODULE_REGISTRY, assert_module_import_smoke

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize("module_name", sorted(MODULE_REGISTRY))
def test_module_import_smoke(module_name: str) -> None:
    """Each registered module imports and exposes its metadata contract."""
    assert_module_import_smoke(module_name)
