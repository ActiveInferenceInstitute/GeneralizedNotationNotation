"""Regression: runner and factory must resolve to real implementations.

Note: ``conftest.py`` registers a minimal ``sys.modules['tests']`` placeholder for
collection; ``import tests`` may not expose ``run_tests``. Pipeline code uses
``from tests import run_tests`` with ``src`` on ``sys.path`` before that stub is
installed. Submodules ``tests.runner`` and ``tests.test_runner_modular`` are always
the canonical sources.
"""

from __future__ import annotations


def test_runner_and_factory_modules() -> None:
    from tests import runner as runner_mod
    from tests import test_runner_modular as modular_mod

    assert runner_mod.run_tests.__module__ == "tests.runner"
    assert modular_mod.create_test_runner.__module__ == "tests.test_runner_modular"
