#!/usr/bin/env python3
"""Runtime bridge for framework execution shared with analysis-side runners.

``gnn.analysis.rxinfer.cross_framework`` renders and executes GNN models
across RxInfer.jl, PyMDP, and ActiveInference.jl, but must not import the
``gnn.execute`` package statically: the executor/runner stack is heavy at
import time and is execute-owned. :class:`FrameworkRuntime` is the small
frozen handle an execute-owning caller passes in; :func:`default_julia_runtime`
resolves the stock handle from ``gnn.execute`` lazily at call time, so a
broken execute package surfaces at comparison time (a per-framework
gate-unavailable verdict, or an envelope import error) and never at
analysis import time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameworkRuntime:
    """Execute-side runtime bridge injected into framework runners.

    Carries the ``gnn.execute`` capabilities the runners need — the
    canonical julia probe, the headless julia subprocess environment, the
    shared subprocess envelope, and the pre-execution security gate — so
    consumers never import ``gnn.execute`` directly.

    The gate is bound lazily: :attr:`load_security_gate` is a zero-arg
    loader returning the gate function, so a broken execute package still
    yields the per-framework fail-closed verdict instead of crashing the
    caller. :attr:`never_started` is the subprocess-envelope contract
    sentinel (the ``return_code`` value for a process that never ran);
    the default matches ``gnn.execute.subprocess_envelope.NEVER_STARTED``.
    """

    julia_executable: Callable[[], str | None]
    julia_subprocess_env: Callable[[], dict[str, str]]
    run_envelope: Callable[..., dict[str, Any]]
    load_security_gate: Callable[[], Callable[[Path], dict[str, Any]]]
    never_started: int = -1


def default_julia_runtime() -> FrameworkRuntime:
    """Resolve the stock runtime from the execute-owning modules.

    All ``gnn.execute`` imports happen here, at call time, so importing
    this module (and the analysis stack above it) never pays the
    executor/runner import cost.
    """
    from gnn.execute.julia_env import julia_subprocess_env
    from gnn.execute.julia_setup import julia_executable
    from gnn.execute.subprocess_envelope import NEVER_STARTED, run_subprocess_envelope

    def _load_security_gate() -> Callable[[Path], dict[str, Any]]:
        # Bound lazily rather than eagerly: importing
        # ``gnn.execute.security_gate`` routes through the full execute
        # package init, and its failure must stay a per-framework
        # gate-unavailable outcome, not a construction crash.
        from gnn.execute.security_gate import check_script_allowed

        return check_script_allowed

    return FrameworkRuntime(
        julia_executable=julia_executable,
        julia_subprocess_env=julia_subprocess_env,
        run_envelope=run_subprocess_envelope,
        load_security_gate=_load_security_gate,
        never_started=NEVER_STARTED,
    )


__all__ = ["FrameworkRuntime", "default_julia_runtime"]
