"""Pins for the ``type_checker.estimation_strategies`` re-export surface.

The module is a pure re-export shim over ``estimation/strategies``; the
contract is that every advertised name resolves to the same implementation
object the subpackage exposes (a broken alias fails loudly here).
"""

from __future__ import annotations

import inspect

from gnn.type_checker import estimation_strategies
from gnn.type_checker.estimation import strategies


def test_advertised_names_match_subpackage_implementations() -> None:
    for name in estimation_strategies.__all__:
        assert hasattr(estimation_strategies, name), name
        assert getattr(estimation_strategies, name) is getattr(strategies, name), name


def test_shim_exposes_every_subpackage_strategy() -> None:
    own_functions = {
        name
        for name, obj in vars(strategies).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and getattr(obj, "__module__", None) == strategies.__name__
    }
    missing = own_functions - set(estimation_strategies.__all__)
    assert not missing, f"shim is missing strategies: {sorted(missing)}"
