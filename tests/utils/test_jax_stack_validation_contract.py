"""Pins for ``utils/jax_stack_validation`` (previously 14% coverage).

The verifier imports and executes the real jax/pymdp/flax stack (core
dependencies) — deterministic and offline.
"""

from __future__ import annotations

from gnn.utils.jax_stack_validation import (
    jax_pymdp_stack_ok,
    verify_jax_pymdp_stack,
)


def test_verify_jax_pymdp_stack_completes_without_raising() -> None:
    # Executes real jax/jnp/flax/pymdp probes; a failure raises.
    verify_jax_pymdp_stack()


def test_stack_ok_reports_true_with_cache() -> None:
    assert jax_pymdp_stack_ok(use_cache=True) is True


def test_stack_ok_bypasses_cache_when_asked() -> None:
    assert jax_pymdp_stack_ok(use_cache=False) is True
