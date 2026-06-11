"""Regression tests for pipeline test output isolation."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from tests import test_runner_helper, test_runner_modes
from tests.test_runner_modular import _isolated_pipeline_output_dir


def test_runner_helper_default_output_dir_honors_pipeline_env(
    monkeypatch: Any,
) -> None:
    """Nested test helpers should not fall back to tracked ``output/``."""
    monkeypatch.setenv("GNN_PIPELINE_TEST_OUTPUT_DIR", "/tmp/gnn-isolated-output")

    assert test_runner_helper._default_output_dir() == "/tmp/gnn-isolated-output"


def test_fast_pipeline_tests_pass_isolated_output_env(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """Step 2's fast pytest subprocess should inherit an isolated output root."""
    monkeypatch.delenv("GNN_PIPELINE_TEST_OUTPUT_DIR", raising=False)
    captured: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured["cmd"] = args[0]
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s", stderr="")

    monkeypatch.setattr(test_runner_modes.subprocess, "run", fake_run)

    logger = logging.getLogger("test-fast-pipeline-isolation")
    assert test_runner_modes.run_fast_pipeline_tests(logger, tmp_path)

    expected = str(tmp_path / "isolated_pipeline_outputs")
    assert captured["env"]["GNN_PIPELINE_TEST_OUTPUT_DIR"] == expected
    assert captured["cwd"].name == "GeneralizedNotationNotation"


def test_modular_runner_uses_isolated_pipeline_output_dir(tmp_path: Any) -> None:
    """Category subprocesses should share the same isolation convention."""
    assert _isolated_pipeline_output_dir(tmp_path) == str(
        tmp_path / "isolated_pipeline_outputs"
    )
