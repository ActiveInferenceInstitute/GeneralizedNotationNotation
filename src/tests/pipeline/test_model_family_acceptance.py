from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from pipeline.model_family_acceptance import (
    load_model_family_manifest,
    run_model_family_acceptance,
)


def test_model_family_manifest_covers_required_families() -> None:
    families = load_model_family_manifest(Path("input/model_family_manifest.json"))
    names = {family.name for family in families}

    assert {
        "basics",
        "discrete",
        "continuous",
        "hierarchical",
        "multiagent",
        "precision",
        "structured",
        "gridworld",
        "scaling-study",
    }.issubset(names)


def test_model_family_acceptance_writes_ledger_with_step_statuses(
    tmp_path: Path,
) -> None:
    seen_commands: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen_commands.append(list(command))
        _write_pipeline_summary(Path(command[command.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="pipeline ok",
            stderr="",
        )

    ledger = run_model_family_acceptance(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics", "structured"],
        runner=runner,
        strict=True,
    )

    assert ledger["schema"] == "gnn_model_family_acceptance_ledger_v1"
    assert ledger["status"] == "passed"
    assert ledger["family_count"] == 2
    assert len(seen_commands) == 2
    assert all("--skip-llm" in command for command in seen_commands)
    for family in ledger["families"]:
        assert family["steps"]["3"] == "passed"
        assert family["steps"]["11"] == "passed"
        assert family["steps"]["0"] == "skipped"
        assert family["interpretability_summary"]["model_count"] >= 1
    assert (tmp_path / "model_family_acceptance_ledger.json").exists()
    assert (tmp_path / "model_family_acceptance_ledger.md").exists()


def test_model_family_acceptance_strict_fails_on_pipeline_failure(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=1,
            stdout="",
            stderr="failed",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict model-family acceptance should fail")


def test_model_family_acceptance_strict_fails_without_pipeline_summary(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="exit code alone is not acceptance evidence",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict acceptance should fail without pipeline summary")


def test_model_family_acceptance_rejects_omitted_profile_steps(
    tmp_path: Path,
) -> None:
    called = False

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess(args=list(command), returncode=0)

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            only_steps="3",
            runner=runner,
            strict=True,
        )
    except ValueError as exc:
        assert "omitted" in str(exc)
        assert "5" in str(exc)
    else:
        raise AssertionError("acceptance should reject missing profile steps")
    assert called is False


def test_model_family_acceptance_fails_when_summary_omits_selected_step(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        _write_pipeline_summary(
            Path(command[command.index("--output-dir") + 1]),
            present_steps={3},
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="summary is truncated",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict acceptance should fail on missing step evidence")

    ledger = json.loads(
        (tmp_path / "model_family_acceptance_ledger.json").read_text(encoding="utf-8")
    )
    family = ledger["families"][0]
    assert family["steps"]["5"] == "failed"
    assert family["step_evidence"]["5"]["reason"] == "missing_summary_evidence"


def test_model_family_acceptance_clears_stale_family_output(
    tmp_path: Path,
) -> None:
    calls = 0

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        output_dir = Path(command[command.index("--output-dir") + 1])
        if calls == 1:
            _write_pipeline_summary(output_dir)
            return subprocess.CompletedProcess(args=list(command), returncode=0)
        return subprocess.CompletedProcess(args=list(command), returncode=1)

    run_model_family_acceptance(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics"],
        runner=runner,
        strict=True,
    )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("stale prior summaries must not certify a later run")

    ledger = json.loads(
        (tmp_path / "model_family_acceptance_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["families"][0]["pipeline_summary"]["available"] is False


def test_model_family_acceptance_fails_synthetic_success_without_artifacts(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        _write_pipeline_summary(
            Path(command[command.index("--output-dir") + 1]),
            include_artifacts=False,
        )
        return subprocess.CompletedProcess(args=list(command), returncode=0)

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict acceptance should require concrete artifacts")

    ledger = json.loads(
        (tmp_path / "model_family_acceptance_ledger.json").read_text(encoding="utf-8")
    )
    family = ledger["families"][0]
    assert family["steps"]["3"] == "failed"
    assert family["step_evidence"]["3"]["reason"] == "missing_artifact_evidence"


def test_model_family_acceptance_treats_warning_exit_code_as_passed(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        _write_pipeline_summary(
            Path(command[command.index("--output-dir") + 1]),
            overall_status="SUCCESS_WITH_WARNINGS",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=2,
            stdout="pipeline completed with warnings",
            stderr="",
        )

    ledger = run_model_family_acceptance(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["basics"],
        runner=runner,
        strict=True,
    )

    assert ledger["status"] == "passed"
    assert ledger["families"][0]["status"] == "passed"
    assert ledger["families"][0]["steps"]["3"] == "passed"


def test_model_family_acceptance_fails_contradictory_failed_overall_status(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        _write_pipeline_summary(
            Path(command[command.index("--output-dir") + 1]),
            overall_status="FAILED",
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=1,
            stdout="summary rows contradict overall failure",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict acceptance should fail contradictory summaries")


def test_model_family_acceptance_fails_code_two_when_selected_step_failed(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        _write_pipeline_summary(
            Path(command[command.index("--output-dir") + 1]),
            failed_steps={12},
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=2,
            stdout="pipeline completed with partial failure",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["basics"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "basics" in str(exc)
    else:
        raise AssertionError("strict acceptance should fail when Step 12 failed")


def test_model_family_acceptance_profiles_unsupported_steps_as_explicit_skips(
    tmp_path: Path,
) -> None:
    seen_commands: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen_commands.append(list(command))
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_pipeline_summary(output_dir, skipped_steps={11, 12})
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="unsupported renderer and executor steps were explicitly skipped",
            stderr="",
        )

    ledger = run_model_family_acceptance(
        Path("input/model_family_manifest.json"),
        tmp_path,
        family_names=["continuous"],
        runner=runner,
        strict=True,
    )

    family = ledger["families"][0]
    assert ledger["status"] == "passed"
    assert seen_commands and "--skip-steps" in seen_commands[0]
    assert seen_commands[0][seen_commands[0].index("--skip-steps") + 1] == "11,12"
    assert family["raw_steps"]["11"] == "skipped"
    assert family["raw_steps"]["12"] == "skipped"
    assert family["steps"]["11"] == "skipped"
    assert family["steps"]["12"] == "skipped"
    assert family["step_evidence"]["11"]["acceptance"] == "profiled_unsupported_skip"
    assert family["step_evidence"]["12"]["acceptance"] == "profiled_unsupported_skip"
    assert (
        "Continuous fixtures use non-POMDP" in family["step_evidence"]["11"]["reason"]
    )
    assert "No executable script is expected" in family["step_evidence"]["12"]["reason"]


def test_model_family_acceptance_rejects_failed_profiled_unsupported_steps(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_pipeline_summary(output_dir, failed_steps={11, 12})
        _write_unsupported_render_execute_summaries(output_dir)
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=1,
            stdout="profiled unsupported steps still failed",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["continuous"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "continuous" in str(exc)
    else:
        raise AssertionError("profiled unsupported steps must not hide raw failures")

    ledger = json.loads(
        (tmp_path / "model_family_acceptance_ledger.json").read_text(encoding="utf-8")
    )
    family = ledger["families"][0]
    assert family["raw_steps"]["11"] == "failed"
    assert family["step_evidence"]["11"]["acceptance"] == "required"


def test_model_family_acceptance_rejects_partial_render_as_unsupported_skip(
    tmp_path: Path,
) -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_pipeline_summary(output_dir, failed_steps={11, 12})
        _write_unsupported_render_execute_summaries(
            output_dir,
            successful_framework_renderings=1,
        )
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=1,
            stdout="partial renderer regression",
            stderr="",
        )

    try:
        run_model_family_acceptance(
            Path("input/model_family_manifest.json"),
            tmp_path,
            family_names=["continuous"],
            runner=runner,
            strict=True,
        )
    except RuntimeError as exc:
        assert "continuous" in str(exc)
    else:
        raise AssertionError("partial renderer failures are not unsupported skips")


def _write_pipeline_summary(
    output_dir: Path,
    failed_steps: set[int] | None = None,
    skipped_steps: set[int] | None = None,
    present_steps: set[int] | None = None,
    overall_status: str | None = None,
    include_artifacts: bool = True,
) -> None:
    failed_steps = failed_steps or set()
    skipped_steps = skipped_steps or set()
    present_steps = present_steps or {3, 5, 6, 11, 12, 15, 16, 23}
    if include_artifacts:
        _write_step_artifacts(output_dir, present_steps)
    summary_dir = output_dir / "00_pipeline_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    for step in sorted(present_steps):
        status = "SUCCESS"
        if step in failed_steps:
            status = "FAILED"
        elif step in skipped_steps:
            status = "SKIPPED"
        steps.append(
            {
                "script_name": f"{step}_step.py",
                "status": status,
            }
        )
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(
            {
                "overall_status": overall_status
                or ("PARTIAL_SUCCESS" if failed_steps else "SUCCESS"),
                "steps": steps,
                "performance_summary": {
                    "failed_steps": len(failed_steps),
                    "successful_steps": len(steps)
                    - len(failed_steps)
                    - len(skipped_steps),
                    "skipped_steps": len(skipped_steps),
                },
            }
        ),
        encoding="utf-8",
    )


def _write_step_artifacts(output_dir: Path, present_steps: set[int]) -> None:
    artifact_payloads = {
        3: (
            "3_gnn_output/gnn_processing_summary.json",
            "3_gnn_output/gnn_processing_results.json",
        ),
        5: ("5_type_checker_output/type_check_results.json",),
        6: (
            "6_validation_output/validation_summary.json",
            "6_validation_output/validation_results.json",
        ),
        11: ("11_render_output/render_processing_summary.json",),
        12: ("12_execute_output/summaries/execution_summary.json",),
        15: ("15_audio_output/audio_results.json",),
        16: ("16_analysis_output/analysis_results.json",),
        23: ("23_report_output/report_processing_summary.json",),
    }
    for step in present_steps:
        for relative in artifact_payloads.get(step, ()):
            path = output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"step": step, "success": True}),
                encoding="utf-8",
            )


def _write_unsupported_render_execute_summaries(
    output_dir: Path,
    *,
    successful_framework_renderings: int = 0,
) -> None:
    render_dir = output_dir / "11_render_output"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "render_processing_summary.json").write_text(
        json.dumps(
            {
                "successful_framework_renderings": successful_framework_renderings,
                "failed_framework_renderings": [
                    {
                        "file": "continuous_navigation.md",
                        "framework": "jax",
                        "message": "POMDP not compatible with jax: Missing required matrices: ['D']",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    execute_dir = output_dir / "12_execute_output" / "summaries"
    execute_dir.mkdir(parents=True, exist_ok=True)
    (execute_dir / "execution_summary.json").write_text(
        json.dumps(
            {
                "total_scripts_found": 0,
                "success": True,
                "skipped_reason": "no_executable_scripts",
                "message": "No executable scripts found",
                "render_failures": [
                    {
                        "file": "continuous_navigation.md",
                        "framework": "jax",
                        "message": "POMDP not compatible with jax: Missing required matrices: ['D']",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
