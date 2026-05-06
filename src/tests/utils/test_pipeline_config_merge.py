"""Tests for merging ``input/config.yaml`` defaults into pipeline arguments."""

import argparse

import pytest

from utils.argument_utils import PipelineArguments, build_step_command_args
from utils.pipeline_config_merge import apply_input_config_defaults


def test_apply_yaml_sets_dev_when_cli_omits_flag() -> None:
    args = PipelineArguments()
    parsed = argparse.Namespace()
    full = {"setup": {"dev": True}}
    apply_input_config_defaults(args, full, parsed)
    assert args.dev is True


def test_apply_yaml_legacy_uv_sync_dev() -> None:
    args = PipelineArguments()
    parsed = argparse.Namespace()
    full = {"uv": {"sync": {"dev": True}}}
    apply_input_config_defaults(args, full, parsed)
    assert args.dev is True


def test_cli_dev_wins_over_yaml() -> None:
    args = PipelineArguments(dev=True)
    parsed = argparse.Namespace(dev=True)
    full = {"setup": {"dev": False}}
    apply_input_config_defaults(args, full, parsed)
    assert args.dev is True


def test_apply_yaml_fast_only_false() -> None:
    args = PipelineArguments()
    parsed = argparse.Namespace()
    full = {"pipeline": {"fast_only": False}}
    apply_input_config_defaults(args, full, parsed)
    assert args.fast_only is False


def test_build_step_command_includes_fast_only_by_default() -> None:
    from pathlib import Path

    cmd = build_step_command_args(
        "2_tests.py",
        PipelineArguments(),
        "python",
        Path("src/2_tests.py"),
    )
    assert "--fast-only" in cmd


def test_build_step_command_includes_dev_when_set() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.dev = True
    cmd = build_step_command_args(
        "1_setup.py",
        args,
        "python",
        Path("src/1_setup.py"),
    )
    assert "--dev" in cmd


def test_build_step_command_includes_install_all_extras() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.install_all_extras = True
    cmd = build_step_command_args(
        "1_setup.py",
        args,
        "python",
        Path("src/1_setup.py"),
    )
    assert "--install-all-extras" in cmd


def test_build_step_command_forwards_serialize_preset() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.serialize_preset = "minimal"
    cmd = build_step_command_args(
        "3_gnn.py",
        args,
        "python",
        Path("src/3_gnn.py"),
    )
    assert "--serialize-preset" in cmd
    assert cmd[cmd.index("--serialize-preset") + 1] == "minimal"


def test_build_step_command_forwards_execution_benchmark_repeats() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.execution_benchmark_repeats = 4
    cmd = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--execution-benchmark-repeats" in cmd
    assert cmd[cmd.index("--execution-benchmark-repeats") + 1] == "4"


def test_build_step_command_step12_omits_benchmark_repeats_when_one() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.execution_benchmark_repeats = 1
    cmd = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--execution-benchmark-repeats" not in cmd


def test_build_step_command_step12_backend_only_when_distributed() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.distributed = False
    args.backend = "dask"
    cmd = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--backend" not in cmd

    args.distributed = True
    cmd2 = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--backend" in cmd2
    assert cmd2[cmd2.index("--backend") + 1] == "dask"


def test_build_step_command_step12_execution_summary_detail_only_when_true() -> None:
    from pathlib import Path

    args = PipelineArguments()
    args.execution_summary_detail = False
    cmd = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--execution-summary-detail" not in cmd

    args.execution_summary_detail = True
    cmd_on = build_step_command_args(
        "12_execute.py",
        args,
        "python",
        Path("src/12_execute.py"),
    )
    assert "--execution-summary-detail" in cmd_on


@pytest.mark.parametrize(
    "kwargs, expect_all_extras, expect_extra_dev",
    [
        ({"install_all_extras": True}, True, False),
        ({"dev": True}, False, True),
        ({}, False, False),
    ],
)
def test_install_uv_dependencies_sync_flags(kwargs, expect_all_extras, expect_extra_dev, monkeypatch) -> None:
    from setup import uv_management

    captured: list[list[str]] = []

    def fake_run(cmd, **kw):  # noqa: ANN001
        captured.append(list(cmd))
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(uv_management.subprocess, "run", fake_run)
    monkeypatch.setattr(uv_management, "get_installed_package_versions", lambda verbose=False: {})

    uv_management.install_uv_dependencies(verbose=False, **kwargs)
    sync_cmd = next(c for c in captured if len(c) >= 2 and c[0].endswith("uv") and c[1] == "sync")
    assert ("--all-extras" in sync_cmd) is expect_all_extras
    assert ("--extra" in sync_cmd and "dev" in sync_cmd) is expect_extra_dev
