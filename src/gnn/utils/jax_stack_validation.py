"""Earlier name; implementation moved to
``gnn/utils/runtime_safety/jax_stack_validation.py`` (S2-33 Step 4)."""

import warnings

warnings.warn(
    "gnn.utils.jax_stack_validation is the earlier name; import gnn.utils.runtime_safety.jax_stack_validation instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.jax_stack_validation import (  # noqa: E402,F401
    jax_pymdp_stack_ok,
    run_jax_stack_probe_subprocess,
    verify_jax_pymdp_stack,
)

__all__ = [
    "jax_pymdp_stack_ok",
    "run_jax_stack_probe_subprocess",
    "verify_jax_pymdp_stack",
]
