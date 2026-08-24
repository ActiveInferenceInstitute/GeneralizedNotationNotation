#!/usr/bin/env python3
"""Tests for optional API-key auth and secure-bind enforcement (api.auth)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.auth import key_matches, require_secure_bind


class TestApiAuth:
    """Pure-function behavior of the API auth boundary."""

    def test_key_matches_when_auth_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GNN_API_KEY", raising=False)
        assert key_matches(None) is True
        assert key_matches("anything") is True

    def test_key_matches_with_configured_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GNN_API_KEY", "secret-token")
        assert key_matches("secret-token") is True
        assert key_matches("wrong") is False
        assert key_matches(None) is False

    def test_require_secure_bind_loopback_is_always_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GNN_API_KEY", raising=False)
        monkeypatch.delenv("GNN_ALLOW_INSECURE_BIND", raising=False)
        assert require_secure_bind("127.0.0.1") is True
        assert require_secure_bind("localhost") is True

    def test_require_secure_bind_non_loopback_needs_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GNN_API_KEY", raising=False)
        monkeypatch.delenv("GNN_ALLOW_INSECURE_BIND", raising=False)
        assert require_secure_bind("0.0.0.0") is False

    def test_require_secure_bind_non_loopback_with_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GNN_API_KEY", "secret-token")
        monkeypatch.delenv("GNN_ALLOW_INSECURE_BIND", raising=False)
        assert require_secure_bind("0.0.0.0") is True

    def test_require_secure_bind_explicit_insecure_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GNN_API_KEY", raising=False)
        monkeypatch.setenv("GNN_ALLOW_INSECURE_BIND", "1")
        assert require_secure_bind("0.0.0.0") is True

    def test_auth_rejection_uses_canonical_error_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Middleware-generated 401 responses must match endpoint errors."""
        from fastapi.testclient import TestClient

        from api.server import app

        monkeypatch.setenv("GNN_API_KEY", "secret-token")
        monkeypatch.setenv("GNN_RATE_LIMIT", "0")
        response = TestClient(app).get("/api/v1/jobs")
        assert response.status_code == 401
        payload = response.json()
        assert set(payload) == {"status", "data", "error", "meta"}
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "unauthorized"
