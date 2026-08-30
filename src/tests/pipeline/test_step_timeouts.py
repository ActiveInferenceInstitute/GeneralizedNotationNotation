"""Real-behavior tests for ``src/pipeline/step_timeouts.py``.

Timeout configuration decides whether a legitimate long-running step (e.g.
Step 12 executing every model across all frameworks) is killed mid-run, so
the resolution order — env override, explicit per-step config, default — is
product behavior, not administrative metadata.
"""

from __future__ import annotations

import pytest

from pipeline.step_timeouts import DEFAULT_TIMEOUT, STEP_TIMEOUTS, get_step_timeout


def test_known_step_returns_configured_timeout():
    assert get_step_timeout("3_gnn.py") == 300
    assert get_step_timeout("12_execute.py") == 7200


def test_comprehensive_flag_selects_dict_variant(monkeypatch):
    monkeypatch.delenv("GNN_STEP_TIMEOUT_2", raising=False)
    assert get_step_timeout("2_tests.py") == 900
    assert get_step_timeout("2_tests.py", comprehensive=True) == 1200


def test_unknown_step_returns_default():
    assert get_step_timeout("99_unknown.py") == DEFAULT_TIMEOUT


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "42")
    assert get_step_timeout("3_gnn.py") == 42
    # Override applies even to steps without explicit config
    monkeypatch.setenv("GNN_STEP_TIMEOUT_99", "77")
    assert get_step_timeout("99_unknown.py") == 77


def test_invalid_env_value_falls_back_to_config(monkeypatch):
    monkeypatch.setenv("GNN_STEP_TIMEOUT_3", "not-a-number")
    assert get_step_timeout("3_gnn.py") == 300


def test_every_registered_step_has_positive_timeout():
    # Guard against a typo introducing a zero/negative timeout that would
    # kill a step instantly.
    for name, cfg in STEP_TIMEOUTS.items():
        values = cfg.values() if isinstance(cfg, dict) else [cfg]
        for v in values:
            assert isinstance(v, int) and v > 0, f"{name}: bad timeout {v}"
