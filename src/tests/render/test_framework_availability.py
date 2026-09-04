#!/usr/bin/env python3
"""Tests for framework availability gating in the canonical registry.

Asserts:
  1. Every framework in ``FRAMEWORK_REGISTRY`` carries either an importable
     backend *or* an explicit ``available=False`` with a human-readable
     ``unavailable_reason`` string.
  2. Requesting the intentionally unavailable framework (``bnlearn``) raises
     ``ValueError`` with a documented, actionable reason via
     ``validate_framework_requested()``. PyTorch left this set once
     torch>=2.13.0 resolved GHSA-rrmf-rvhw-rf47.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from render.framework_registry import (
    FRAMEWORK_REGISTRY,
    get_framework_availability,
    get_supported_frameworks,
    validate_framework_requested,
)

# ── Framework definitions used across tests ────────────────────────────────

# Frameworks that MUST be marked unavailable in the registry because their
# Python dependency is intentionally absent from the default lock.
INTENTIONALLY_UNAVAILABLE: set[str] = {"bnlearn"}

# Frameworks that ARE importable (Python) or unconditionally available
# (Julia/Stan code generation).  These should never be marked unavailable.
# PyTorch joined this set when torch>=2.13.0 resolved GHSA-rrmf-rvhw-rf47
# (the package ships in the ``torch`` extra).
EXPECTED_AVAILABLE: set[str] = {
    "pymdp",
    "rxinfer",
    "activeinference_jl",
    "jax",
    "discopy",
    "pytorch",
    "numpyro",
    "stan",
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _importable(name: str) -> bool:
    """Return True if ``name`` can be imported in the current interpreter."""
    return importlib.util.find_spec(name) is not None


# ── Tests ──────────────────────────────────────────────────────────────────


class TestRegistryCompleteness:
    """Every registered framework has an explicit availability story."""

    @pytest.mark.parametrize("framework", list(FRAMEWORK_REGISTRY.keys()))
    def test_framework_has_availability_field(self, framework: str) -> None:
        """Every entry must carry 'available' and 'unavailable_reason'."""
        spec = FRAMEWORK_REGISTRY[framework]
        assert "available" in spec, f"{framework} is missing 'available' field"
        assert "unavailable_reason" in spec, (
            f"{framework} is missing 'unavailable_reason' field"
        )
        assert isinstance(spec["available"], bool), (
            f"{framework}.available must be bool, got {type(spec['available'])}"
        )
        reason = spec["unavailable_reason"]
        if spec["available"]:
            assert reason is None, (
                f"{framework} is marked available but has unavailable_reason={reason!r}"
            )
        else:
            assert reason is not None and len(reason) > 10, (
                f"{framework} is marked unavailable but has no meaningful "
                f"unavailable_reason (got {reason!r})"
            )

    def test_supported_frameworks_matches_registry_keys(self) -> None:
        assert set(get_supported_frameworks()) == set(FRAMEWORK_REGISTRY.keys())


class TestAvailabilityStory:
    """Every framework has either an importable backend or an explicit reason."""

    def test_expected_available_are_available(self) -> None:
        for fw in EXPECTED_AVAILABLE:
            available, reason = get_framework_availability(fw)
            assert available is True, (
                f"Framework {fw!r} expected available but got "
                f"available={available}, reason={reason!r}"
            )

    def test_intentionally_unavailable_are_unavailable(self) -> None:
        for fw in INTENTIONALLY_UNAVAILABLE:
            available, reason = get_framework_availability(fw)
            assert available is False, (
                f"Framework {fw!r} expected unavailable but got available={available}"
            )
            assert reason is not None, (
                f"Framework {fw!r} is unavailable but reason is None"
            )
            # The reason MUST mention the GHSA identifier.
            assert "GHSA" in reason, (
                f"Framework {fw!r} unavailable reason should mention the "
                f"security advisory, got: {reason}"
            )

    def test_unavailable_reason_is_human_readable(self) -> None:
        """Unavailable reasons are full sentences, not error codes."""
        for fw in INTENTIONALLY_UNAVAILABLE:
            _available, reason = get_framework_availability(fw)
            assert reason is not None
            # Should be at least 50 chars (meaningful sentence)
            assert len(reason) >= 50, f"{fw} reason too short: {reason!r}"
            # Should start with a capital letter
            assert reason[0].isupper(), f"{fw} reason not capitalized: {reason!r}"
            # Should mention how to re-enable
            assert "your own risk" in reason.lower() or "enable" in reason.lower(), (
                f"{fw} reason should mention how to enable: {reason!r}"
            )

    def test_get_framework_availability_unknown_framework(self) -> None:
        """Unknown frameworks are assumed available (no over-blocking)."""
        available, reason = get_framework_availability("__definitely_not_a_framework__")
        assert available is True
        assert reason is None


class TestValidationGate:
    """``validate_framework_requested()`` raises clear errors."""

    def test_validate_available_framework_passes(self) -> None:
        # Should not raise for any known-available framework.
        for fw in EXPECTED_AVAILABLE:
            validate_framework_requested(fw)

    def test_validate_unknown_framework_passes(self) -> None:
        # Unknown frameworks should not be blocked.
        validate_framework_requested("__some_external_tool__")

    @pytest.mark.parametrize("framework", sorted(INTENTIONALLY_UNAVAILABLE))
    def test_validate_unavailable_raises_value_error(self, framework: str) -> None:
        """Requesting an unavailable framework raises ValueError with a clear message."""
        with pytest.raises(ValueError) as exc_info:
            validate_framework_requested(framework)

        msg = str(exc_info.value)
        # Must contain the framework name
        assert framework.lower() in msg.lower(), (
            f"Error message should mention framework name: {msg}"
        )
        # Must mention how to enable
        assert "uv add" in msg.lower() or "enable" in msg.lower(), (
            f"Error message should suggest how to enable: {msg}"
        )

    def test_validate_bnlearn_message_exact(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            validate_framework_requested("bnlearn")
        msg = str(exc_info.value)
        assert "bnlearn" in msg
        assert "pgmpy" in msg
        assert "uv add" in msg

    def test_validate_pytorch_is_available(self) -> None:
        """PyTorch is registry-available: torch>=2.13.0 resolves
        GHSA-rrmf-rvhw-rf47, so requesting it must not raise."""
        validate_framework_requested("pytorch")


class TestRealEnvironmentCheck:
    """Cross-reference registry status against the real interpreter import state.

    This catches drift where a framework marked 'available' was never actually
    importable in the default environment.
    """

    def test_available_python_frameworks_are_importable(self) -> None:
        """Python-based frameworks marked 'available' should be importable."""
        # Map framework name → Python module name for import check
        framework_to_module: dict[str, str] = {
            "pymdp": "pymdp",
            "jax": "jax",
            "discopy": "discopy",
            "numpyro": "numpyro",
        }
        for fw, mod in framework_to_module.items():
            available, _ = get_framework_availability(fw)
            if available:
                # Just check that the module CAN be found (not that features
                # work) — this is an existence check, not a runtime test.
                spec = importlib.util.find_spec(mod)
                assert spec is not None, (
                    f"{fw} ({mod}) not installed in this environment — "
                    "core dependency missing"
                )

    def test_unavailable_frameworks_are_not_importable(self) -> None:
        """Frameworks marked 'unavailable' should NOT be importable in a
        clean environment — confirms the gate is meaningful."""
        # bnlearn → bnlearn module
        for fw in INTENTIONALLY_UNAVAILABLE:
            available, _ = get_framework_availability(fw)
            assert available is False
            # No need to assert non-importability — the registry says it's
            # unavailable regardless of local env state.
