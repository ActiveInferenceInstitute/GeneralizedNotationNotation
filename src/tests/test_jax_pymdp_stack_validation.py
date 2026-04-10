"""
Package integrity: JAX + Optax + Flax + pymdp 1.x must work in the project ``.venv``.

Uses the same subprocess probe as Step 1 and ``validate_uv_setup``, so the test passes
even when ``pytest`` is launched from a different interpreter (``uv run pytest`` still
validates the lockfile environment).
"""

from pathlib import Path

import pytest

from setup.constants import PROJECT_ROOT, VENV_PYTHON
from utils.jax_stack_validation import run_jax_stack_probe_subprocess


@pytest.mark.jax_stack
@pytest.mark.unit
def test_jax_stack_probe_matches_setup() -> None:
    """``utils.jax_stack_validation`` must succeed under ``.venv/bin/python``."""
    if not Path(VENV_PYTHON).exists():
        pytest.skip(f"No project interpreter at {VENV_PYTHON}; run uv sync")
    ok, out = run_jax_stack_probe_subprocess(Path(VENV_PYTHON), PROJECT_ROOT)
    assert ok, out[:4000] if out else "(no output)"
