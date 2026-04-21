#!/usr/bin/env python3
"""Execute module public-API sanity checks.

Phase 6: recovery.py fallback stubs were removed as dead code. The execute
submodules are all in-tree — their imports must succeed unconditionally in
any working install. This file verifies the promoted contract: real
validators return structured dicts with documented keys, and public symbols
are actually callable (no stale aliases hanging around).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execute import (  # noqa: E402
    check_dependencies,
    check_file_permissions,
    check_network_connectivity,
    check_python_environment,
    check_system_resources,
    execute_gnn_model,
    execute_script_safely,
    execute_simulation_from_gnn,
    GNNExecutor,
    process_execute,
    run_simulation,
    validate_execution_environment,
    validate_pymdp_environment,
)


def test_public_symbols_are_callable_or_class():
    """Every public name in the module must be a live callable or class."""
    for sym in (
        check_dependencies,
        check_file_permissions,
        check_network_connectivity,
        check_python_environment,
        check_system_resources,
        execute_gnn_model,
        execute_script_safely,
        execute_simulation_from_gnn,
        process_execute,
        run_simulation,
        validate_execution_environment,
        validate_pymdp_environment,
    ):
        assert callable(sym), f"{sym!r} is not callable"
    assert isinstance(GNNExecutor, type), "GNNExecutor should be a class"


def test_validate_pymdp_environment_returns_structured_dict():
    """Real validator exposes at least one status field per its contract."""
    result = validate_pymdp_environment()
    assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
    candidates = {"valid", "pymdp_available", "overall_health", "status"}
    assert candidates & set(result.keys()), (
        f"validate_pymdp_environment missing status key; got keys: {list(result.keys())}"
    )


def test_check_network_connectivity_has_status():
    result = check_network_connectivity()
    # Real validator returns a ValidationResult dataclass exposing .status.
    if isinstance(result, dict):
        assert "status" in result
    else:
        assert hasattr(result, "status")


def test_validate_execution_environment_returns_dict():
    result = validate_execution_environment()
    assert isinstance(result, dict)
    # Real validator reports python_version + dependencies per its API.
    assert any(k in result for k in ("python_version", "dependencies", "overall_status"))


def test_gnnexecutor_is_instantiable():
    engine = GNNExecutor()
    assert engine is not None
    # Basic interface checks — the class must expose the public execution
    # method callers depend on.
    assert hasattr(engine, "run_simulation") or hasattr(engine, "execute")
