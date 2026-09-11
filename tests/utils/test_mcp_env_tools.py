"""Pins for ``utils/mcp`` environment redaction and info tools (19% coverage)."""

from __future__ import annotations

import os

from gnn.utils.mcp.server import (
    SENSITIVE_ENV_KEY_MARKERS,
    get_environment_info,
    get_system_info,
    is_sensitive_env_key,
    redact_environment,
)


def test_sensitive_marker_list_is_comprehensive() -> None:
    assert "password" in SENSITIVE_ENV_KEY_MARKERS
    assert "token" in SENSITIVE_ENV_KEY_MARKERS
    assert "credential" in SENSITIVE_ENV_KEY_MARKERS


def test_is_sensitive_env_key_matches_case_insensitively() -> None:
    assert is_sensitive_env_key("GNN_API_TOKEN")
    assert is_sensitive_env_key("my_password")
    assert is_sensitive_env_key("OPENAI_API_KEY")
    assert not is_sensitive_env_key("HOME")
    assert not is_sensitive_env_key("PATH")
    assert not is_sensitive_env_key("PYTHONUNBUFFERED")


def test_redact_environment_removes_secret_carrying_keys(
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("W2_SECRET_VALUE", "s3cret")  # type: ignore[attr-defined]
    monkeypatch.setenv("W2_PLAIN_VALUE", "visible")  # type: ignore[attr-defined]

    redacted = redact_environment()

    assert "W2_SECRET_VALUE" not in redacted
    assert redacted.get("W2_PLAIN_VALUE") == "visible"
    # os.environ itself is untouched: redaction is a copy.
    assert os.environ["W2_SECRET_VALUE"] == "s3cret"


def test_get_system_info_returns_platform_contract() -> None:
    info = get_system_info(object())  # ref unused by the implementation

    # The tool nests platform, memory, and CPU blocks under one dict.
    assert isinstance(info["system"], dict)
    assert isinstance(info["system"]["python_path"], list)
    assert "memory" in info
    assert "cpu" in info


def test_get_environment_info_redacts_secrets(monkeypatch: object) -> None:
    monkeypatch.setenv("W2_TOKEN_PROBE", "s3cret")  # type: ignore[attr-defined]

    info = get_environment_info(object())

    env_view = str(info)
    assert "s3cret" not in env_view
