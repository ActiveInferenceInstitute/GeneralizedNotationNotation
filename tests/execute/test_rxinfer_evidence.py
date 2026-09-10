"""Deterministic evidence-persistence tests for the RxInfer runner (MED-03b).

The runner must persist ``{stem}_stdout.txt``, ``{stem}_stderr.txt``, and
``{stem}_execution_log.json`` on EVERY run — success, failure, timeout, or
never-started — mirroring the JAX runner's evidence files. The subprocess
envelope is monkeypatched so no Julia toolchain is required.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.execute.rxinfer import rxinfer_runner

pytestmark = pytest.mark.execute


def _envelope(**overrides: object) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "success": True,
        "return_code": 0,
        "stdout": "rxinfer-stdout-line\n",
        "stderr": "",
        "duration_seconds": 1.25,
    }
    envelope.update(overrides)
    return envelope


def _fake_envelope_runner(
    monkeypatch: pytest.MonkeyPatch, envelope: Dict[str, Any]
) -> None:
    """Replace the subprocess envelope with a canned, isolated envelope."""
    monkeypatch.setattr(
        rxinfer_runner, "run_subprocess_envelope", lambda command, **kwargs: envelope
    )


def _make_script(tmp_path: Path) -> Path:
    script = tmp_path / "demo_model_rxinfer.jl"
    script.write_text("using RxInfer\nprintln(\"demo\")\n", encoding="utf-8")
    return script


class TestEvidencePersistence:
    """Every run leaves a durable stdout/stderr/log triple."""

    @pytest.mark.unit
    def test_success_writes_evidence_triple(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _make_script(tmp_path)
        _fake_envelope_runner(monkeypatch, _envelope())
        output_dir = tmp_path / "evidence"

        ok = rxinfer_runner.execute_rxinfer_script(script, output_dir=output_dir)

        assert ok is True
        stem = "demo_model_rxinfer"
        stdout_file = output_dir / f"{stem}_stdout.txt"
        stderr_file = output_dir / f"{stem}_stderr.txt"
        log_file = output_dir / f"{stem}_execution_log.json"
        assert stdout_file.read_text(encoding="utf-8") == "rxinfer-stdout-line\n"
        assert stderr_file.read_text(encoding="utf-8") == ""
        log = json.loads(log_file.read_text(encoding="utf-8"))
        assert log["success"] is True
        assert log["return_code"] == 0
        assert log["elapsed_seconds"] == 1.25
        assert log["error_type"] is None
        assert log["timeout"] == 300
        assert log["script"].endswith("demo_model_rxinfer.jl")

    @pytest.mark.unit
    def test_failure_writes_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _make_script(tmp_path)
        _fake_envelope_runner(
            monkeypatch,
            _envelope(
                success=False,
                return_code=1,
                stdout="partial\n",
                stderr="LoadError: bad model\n",
                duration_seconds=0.5,
            ),
        )
        output_dir = tmp_path / "evidence"

        ok = rxinfer_runner.execute_rxinfer_script(script, output_dir=output_dir)

        assert ok is False
        stem = "demo_model_rxinfer"
        output_files = sorted(p.name for p in output_dir.iterdir())
        assert output_files == [
            f"{stem}_execution_log.json",
            f"{stem}_stderr.txt",
            f"{stem}_stdout.txt",
        ]
        assert (
            output_dir / f"{stem}_stderr.txt"
        ).read_text(encoding="utf-8") == "LoadError: bad model\n"
        log = json.loads(
            (output_dir / f"{stem}_execution_log.json").read_text(encoding="utf-8")
        )
        assert log["success"] is False
        assert log["return_code"] == 1

    @pytest.mark.unit
    def test_timeout_writes_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _make_script(tmp_path)
        _fake_envelope_runner(
            monkeypatch,
            _envelope(
                success=False,
                return_code=-1,
                stdout="partial-before-kill\n",
                stderr="partial-err\n",
                error_type="TimeoutExpired",
                error="Execution timed out after 300s",
                duration_seconds=300.01,
            ),
        )
        output_dir = tmp_path / "evidence"

        ok = rxinfer_runner.execute_rxinfer_script(script, output_dir=output_dir)

        assert ok is False
        stem = "demo_model_rxinfer"
        log = json.loads(
            (output_dir / f"{stem}_execution_log.json").read_text(encoding="utf-8")
        )
        assert log["success"] is False
        assert log["error_type"] == "TimeoutExpired"
        # Partial output captured by the kill+drain envelope survives on disk.
        assert (
            output_dir / f"{stem}_stdout.txt"
        ).read_text(encoding="utf-8") == "partial-before-kill\n"

    @pytest.mark.unit
    def test_default_output_dir_is_script_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = _make_script(tmp_path)
        _fake_envelope_runner(monkeypatch, _envelope())

        ok = rxinfer_runner.execute_rxinfer_script(script)

        assert ok is True
        assert (tmp_path / "demo_model_rxinfer_stdout.txt").exists()
        assert (tmp_path / "demo_model_rxinfer_execution_log.json").exists()

    @pytest.mark.unit
    def test_run_rxinfer_scripts_threads_output_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rendered = tmp_path / "rendered" / "rxinfer"
        rendered.mkdir(parents=True)
        script = rendered / "pooled_model_rxinfer.jl"
        script.write_text("using RxInfer\n", encoding="utf-8")
        evidence_dir = tmp_path / "evidence"
        monkeypatch.setattr(rxinfer_runner, "is_julia_available", lambda: True)
        _fake_envelope_runner(monkeypatch, _envelope())

        ok = rxinfer_runner.run_rxinfer_scripts(
            tmp_path / "rendered", execution_output_dir=evidence_dir
        )

        assert ok is True
        log = json.loads(
            (evidence_dir / "pooled_model_execution_log.json").read_text(
                encoding="utf-8"
            )
        )
        assert log["success"] is True
        assert (evidence_dir / "pooled_model_stdout.txt").exists()
