"""
JAX + PyMDP stack validation for Step 12 (execute) and package integrity.

Single source of truth used by:
  - ``setup.dependency_setup.install_jax_and_test`` (subprocess with PYTHONPATH=src)
  - ``setup.uv_management.validate_uv_setup``
  - Pytest (``test_jax_pymdp_stack_validation.py``)

Requires core dependencies: ``jax``, ``jaxlib``, ``optax``, ``flax``, ``inferactively-pymdp`` (pymdp 1.x).
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 -- controlled invocation of venv Python
from pathlib import Path
from typing import Optional, Tuple

_stack_ok_cache: Optional[bool] = None


def verify_jax_pymdp_stack() -> None:
    """
    Run import, compile, and runtime checks for JAX, Optax, Flax, and pymdp 1.x.

    On success sets the process-wide cache used by :func:`jax_pymdp_stack_ok`.

    Raises:
        RuntimeError: If any check fails (with a short actionable message).
    """
    global _stack_ok_cache
    try:
        _verify_jax_pymdp_stack_impl()
    except Exception:
        _stack_ok_cache = False
        raise
    _stack_ok_cache = True


def _verify_jax_pymdp_stack_impl() -> None:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices — check jax/jaxlib install and platform support")

    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.sum(jnp.sin(x))
    if not jnp.isfinite(y):
        raise RuntimeError("JAX NumPy ops produced non-finite values")

    @jax.jit
    def _jit_sum_sin(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.sin(z))

    r = _jit_sum_sin(x)
    if not jnp.isfinite(r):
        raise RuntimeError("JAX JIT path failed")

    def _sin_row(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(v)

    vm = jax.vmap(_sin_row)
    _ = vm(jnp.ones((2, 2), dtype=jnp.float32))

    @jax.jit
    def _scale(z: jnp.ndarray) -> jnp.ndarray:
        return z * 2.0

    xla_res = _scale(jnp.ones(10, dtype=jnp.float32)).block_until_ready()
    if xla_res.shape != (10,) or not jnp.all(jnp.isfinite(xla_res)):
        raise RuntimeError("XLA compile/run or block_until_ready failed")

    opt = optax.adam(0.01)
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    opt.init(params)

    class _Tiny(nn.Module):
        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            return nn.Dense(1)(z)

    model = _Tiny()
    key = jax.random.PRNGKey(0)
    variables = model.init(key, jnp.ones((1, 2), dtype=jnp.float32))
    out = model.apply(variables, jnp.ones((1, 2), dtype=jnp.float32))
    if out.shape != (1, 1) or not jnp.isfinite(out).all():
        raise RuntimeError("Flax module forward pass failed")

    from pymdp.agent import Agent

    if not hasattr(Agent, "update_empirical_prior"):
        raise RuntimeError(
            "pymdp Agent missing update_empirical_prior — install inferactively-pymdp>=1.0.0"
        )


def jax_pymdp_stack_ok(*, use_cache: bool = True) -> bool:
    """
    Return True if :func:`verify_jax_pymdp_stack` succeeds.

    Results are cached per process (skip guards import many modules; one real probe is enough).
    Pass ``use_cache=False`` to force a fresh run (e.g. after installing deps in-process).
    """
    global _stack_ok_cache
    if use_cache and _stack_ok_cache is not None:
        return _stack_ok_cache
    try:
        verify_jax_pymdp_stack()
        return True
    except Exception:
        return False


def run_jax_stack_probe_subprocess(venv_python: Path, project_root: Path) -> Tuple[bool, str]:
    """
    Run :func:`verify_jax_pymdp_stack` inside ``venv_python`` with ``PYTHONPATH=<project>/src``.

    Used by setup and validation so the probe always uses the same interpreter as the lockfile.
    """
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    proc = subprocess.run(  # nosec B603 -- controlled/trusted paths
        [
            str(venv_python),
            "-c",
            "from utils.jax_stack_validation import verify_jax_pymdp_stack; verify_jax_pymdp_stack()",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()
