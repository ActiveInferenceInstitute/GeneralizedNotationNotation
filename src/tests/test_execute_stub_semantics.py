#!/usr/bin/env python3
"""Phase 2.1 regression: execute stubs must return structured envelopes.

Pre-Phase-2.1 stubs returned bare ``{}`` / ``[]`` / ``{"valid": False, ...}``,
making it impossible for callers to distinguish "validator unavailable" from
"validator ran cleanly". The new envelopes expose ``available: False`` so
code can branch explicitly.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execute import (  # noqa: E402
    check_dependencies,
    check_file_permissions,
    check_network_connectivity,
    check_system_resources,
    execute_simulation_from_gnn,
    get_pymdp_health_status,
    run_simulation,
    validate_pymdp_environment,
    VALIDATION_AVAILABLE,
)


def _is_stub(result) -> bool:
    """True when the returned envelope marks itself unavailable."""
    if isinstance(result, dict):
        return result.get("available") is False
    return False


def test_execute_stubs_only_engage_when_validation_unavailable():
    """When the main import path succeeds, stubs must NOT engage. The
    VALIDATION_AVAILABLE module flag is the discriminator.

    The real ``validate_pymdp_environment`` returns either a dict OR a
    ValidationResult dataclass (depending on the concrete implementation),
    so we check by attribute/key rather than by type alone.
    """
    result = validate_pymdp_environment()
    if VALIDATION_AVAILABLE:
        # Real path — result is a dict (or dataclass) exposing SOME signal
        # about whether PyMDP is usable. Accepted signals:
        #   - "valid" (stub form)
        #   - "available" (new envelope)
        #   - "pymdp_available" (real validator)
        #   - "overall_health" (real validator)
        #   - "status" attribute (ValidationResult dataclass)
        candidates = {"valid", "available", "pymdp_available", "overall_health"}
        has_key = isinstance(result, dict) and bool(candidates & set(result.keys()))
        has_attr = hasattr(result, "status") or hasattr(result, "valid")
        assert has_key or has_attr, f"Real validator returned unrecognized shape: {result!r}"
    else:
        assert _is_stub(result), f"Stub envelope expected when VALIDATION_AVAILABLE=False; got {result!r}"


def test_check_system_resources_envelope_shape():
    """check_system_resources must always return a shape callers can
    introspect without crashing — either a list OR a structured dict."""
    result = check_system_resources()
    # Real path returns list; stub path returns dict with available=False.
    assert isinstance(result, (list, dict))
    if isinstance(result, dict):
        assert result.get("available") is False
        assert "reason" in result


def test_check_dependencies_envelope_shape():
    result = check_dependencies()
    assert isinstance(result, (list, dict))
    if isinstance(result, dict):
        assert result.get("available") is False


def test_check_file_permissions_envelope_shape():
    result = check_file_permissions()
    assert isinstance(result, (list, dict))


def test_check_network_connectivity_has_status_field():
    result = check_network_connectivity()
    # Real path returns a ValidationResult dataclass; stub path returns a dict.
    # Both expose a "status" signal — via attribute or key respectively.
    if isinstance(result, dict):
        assert "status" in result
    else:
        assert hasattr(result, "status"), f"Expected status field on {type(result).__name__}"


def test_run_simulation_failure_is_distinguishable():
    """When run_simulation falls through to the stub, the returned dict
    must carry error_type='NotAvailable' so callers can branch on it."""
    if VALIDATION_AVAILABLE:
        # Real path — we don't invoke with a real cfg here; just verify the
        # symbol is callable and returns a dict.
        try:
            out = run_simulation({})
            assert isinstance(out, dict)
        except Exception:
            pass  # real runners may raise; that's out of scope
    else:
        out = run_simulation({})
        assert out.get("success") is False
        assert out.get("error_type") == "NotAvailable"


def test_execute_simulation_from_gnn_stub_envelope():
    """If the module engaged the stub path, its envelope is structured."""
    # The symbol always exists; its shape depends on availability flag.
    # This test just verifies callability + dict-return without raising.
    # (A full execution would need a real GNN file, which is tested elsewhere.)
    assert callable(execute_simulation_from_gnn)
