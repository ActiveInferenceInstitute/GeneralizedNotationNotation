#!/usr/bin/env python3
"""Tests for the consolidated in-process step executor (S2-11 / V4-STAGE).

Proves the equivalence contract from ADR 0001: consolidated in-process
execution produces the same downstream artifacts as the canonical subprocess
path for a small target dir, receipts record the execution mode, and steps
outside the whitelist are refused.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from gnn.pipeline.step_executor import (  # noqa: E402
    UnsupportedStepError,
    can_execute_in_process,
    execute_step_in_process,
)
from gnn.utils.pipeline_arguments import PipelineArguments  # noqa: E402

PROJECT_ROOT = SRC

LOGGER = logging.getLogger("test_step_executor")
BASICS_DIR = PROJECT_ROOT / "input" / "gnn_files" / "basics"

def _pipeline_args(output_dir: Path) -> PipelineArguments:
    """Build pipeline args pointing at the small basics fixture dir."""
    return PipelineArguments(target_dir=BASICS_DIR, output_dir=output_dir)


def _artifact_files(step_output: Path) -> list[str]:
    """Sorted artifact-relative file list under a step's output dir."""
    return sorted(
        p.relative_to(step_output).as_posix()
        for p in step_output.rglob("*")
        if p.is_file()
    )


def _run_step3_subprocess(output_dir: Path) -> None:
    """Run the canonical numbered-script subprocess for step 3."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "gnn" / "3_gnn.py"),
        "--target-dir",
        str(BASICS_DIR),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(  # nosec B603 - fixed argv, no shell
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr[-2000:]


class TestConsolidatedEquivalence:
    """In-process mode produces the same downstream artifacts as subprocess."""

    def test_step3_in_process_matches_subprocess_artifacts(
        self, tmp_path: Path
    ) -> None:
        """Step 3 via the executor matches the subprocess artifact contract."""
        subprocess_out = tmp_path / "subprocess"
        in_process_out = tmp_path / "inprocess"
        _run_step3_subprocess(subprocess_out)

        args = _pipeline_args(in_process_out)
        step_result = execute_step_in_process("3_gnn.py", args, LOGGER)

        assert step_result["exit_code"] == 0
        assert step_result["status"] == "SUCCESS"

        sub_dir = subprocess_out / "3_gnn_output"
        in_dir = in_process_out / "3_gnn_output"
        assert sub_dir.is_dir() and in_dir.is_dir()
        sub_files = _artifact_files(sub_dir)
        in_files = _artifact_files(in_dir)
        assert in_files == sub_files
        assert in_files, "no step-3 artifacts produced"

        # Model count: every source GNN file parses successfully, and the
        # per-model directories match the source file names.
        sub_summary = json.loads((sub_dir / "gnn_processing_summary.json").read_text())
        in_summary = json.loads((in_dir / "gnn_processing_summary.json").read_text())
        model_count = len(list(BASICS_DIR.glob("*.md")))
        assert in_summary["total_files"] == sub_summary["total_files"]
        assert in_summary["successful_parses"] == sub_summary["successful_parses"]
        assert in_summary["total_files"] == model_count
        assert in_summary["formats_per_file"] == sub_summary["formats_per_file"]


class TestConsolidatedReceipt:
    """Receipt fields record which execution mode ran."""

    def test_receipt_records_consolidated_mode(self, tmp_path: Path) -> None:
        """The executor stamps execution_mode="consolidated" on its receipt."""
        step_result = execute_step_in_process(
            "3_gnn.py", _pipeline_args(tmp_path / "receipt"), LOGGER
        )

        assert step_result["execution_mode"] == "consolidated"
        assert step_result["status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}
        assert step_result["exit_code"] in (0, 2)
        assert isinstance(step_result["prerequisite_check"], bool)

    def test_record_step_result_defaults_execution_mode(self) -> None:
        """Subprocess receipts default to execution_mode="subprocess" and the
        consolidated stamp is never overwritten by the shared recording tail."""
        from gnn.main import _record_step_result

        def recorded_summary() -> dict[str, Any]:
            return {
                "steps": [],
                "performance_summary": {
                    "peak_memory_mb": 0.0,
                    "total_steps": 1,
                    "failed_steps": 0,
                    "critical_failures": 0,
                    "successful_steps": 0,
                    "warnings": 0,
                },
            }

        summary = recorded_summary()
        _record_step_result(
            {"status": "SUCCESS", "stdout": "", "stderr": "", "exit_code": 0},
            1,
            "3_gnn.py",
            "GNN file processing",
            datetime.now(),
            datetime.now(),
            0.1,
            summary,
            1,
            None,
            LOGGER,
        )
        assert summary["steps"][0]["execution_mode"] == "subprocess"

        summary = recorded_summary()
        _record_step_result(
            {
                "status": "SUCCESS",
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "execution_mode": "consolidated",
            },
            1,
            "3_gnn.py",
            "GNN file processing",
            datetime.now(),
            datetime.now(),
            0.1,
            summary,
            1,
            None,
            LOGGER,
        )
        assert summary["steps"][0]["execution_mode"] == "consolidated"


class TestConsolidatedRefusal:
    """Unknown or unsupported steps are refused by the executor."""

    def test_unknown_step_refused(self, tmp_path: Path) -> None:
        """Steps outside the whitelist raise UnsupportedStepError."""
        args = _pipeline_args(tmp_path)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("11_render.py", args, LOGGER)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("99_nonsense.py", args, LOGGER)

    def test_alias_resolved_step_outside_whitelist_refused(
        self, tmp_path: Path
    ) -> None:
        """Consolidated aliases resolve first, then the whitelist decides:
        13_audio canonicalizes to 15_audio, which is not in the slice."""
        args = _pipeline_args(tmp_path)
        with pytest.raises(UnsupportedStepError):
            execute_step_in_process("13_audio.py", args, LOGGER)

    def test_gate_matches_whitelist_and_matrix(self, tmp_path: Path) -> None:
        """can_execute_in_process encodes exactly the slice gate."""
        args = _pipeline_args(tmp_path)
        assert can_execute_in_process("0_template.py", args)
        assert can_execute_in_process("3_gnn.py", args)
        assert can_execute_in_process("5_type_checker.py", args)
        assert not can_execute_in_process("11_render.py", args)
        assert not can_execute_in_process("99_nonsense.py", args)
        # Matrix enabled alone does not block: execute_pipeline_step only
        # forks per-folder when a subfolder actually allows the step.
        matrix_config = {"testing_matrix": {"enabled": True, "default_steps": [3]}}
        assert can_execute_in_process("3_gnn.py", args, pipeline_config=matrix_config)
        # A dispatching subfolder keeps the step on the subprocess path.
        dispatch_target = tmp_path / "dispatch"
        (dispatch_target / "model_a").mkdir(parents=True)
        dispatch_args = PipelineArguments(
            target_dir=dispatch_target, output_dir=tmp_path
        )
        assert not can_execute_in_process(
            "3_gnn.py", dispatch_args, pipeline_config=matrix_config
        )

    def test_global_steps_disabled_produces_skipped_receipt(
        self, tmp_path: Path
    ) -> None:
        """A disabled global step mirrors the subprocess skip without running."""
        args = _pipeline_args(tmp_path)
        matrix_config = {
            "testing_matrix": {"enabled": True, "global_steps": {"0_template": False}}
        }
        step_result = execute_step_in_process(
            "0_template.py", args, LOGGER, pipeline_config=matrix_config
        )
        assert step_result["status"] == "SKIPPED"
        assert step_result["exit_code"] == 0
        assert step_result["execution_mode"] == "consolidated"
        assert not (tmp_path / "0_template_output").exists()


class TestFirstSliceSteps:
    """Every whitelisted step runs in-process and writes its standard dir."""

    @pytest.mark.parametrize(
        ("script", "dir_name"),
        [
            ("0_template.py", "0_template_output"),
            ("5_type_checker.py", "5_type_checker_output"),
        ],
    )
    def test_step_produces_standard_output_dir(
        self, tmp_path: Path, script: str, dir_name: str
    ) -> None:
        step_result = execute_step_in_process(script, _pipeline_args(tmp_path), LOGGER)
        assert step_result["execution_mode"] == "consolidated"
        assert step_result["exit_code"] in (0, 2), step_result["stderr"]
        assert (tmp_path / dir_name).is_dir()
