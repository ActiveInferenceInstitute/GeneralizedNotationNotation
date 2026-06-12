"""Interface guardrails for step arguments and dependency closure."""

import logging
import sys
from pathlib import Path

import pytest

from utils.argument_utils import ArgumentParser
from utils.pipeline_step_dependencies import resolve_step_dependencies
from utils.pipeline_template import _parse_step_args


def test_step_specific_arguments_are_exposed() -> None:
    step0 = ArgumentParser.parse_step_arguments("0_template.py", ["--simulate-error"])
    assert step0.simulate_error is True

    step2 = ArgumentParser.parse_step_arguments("2_tests.py", ["--fast-only"])
    assert step2.fast_only is True

    step4 = ArgumentParser.parse_step_arguments(
        "4_model_registry.py", ["--registry-path", "custom_registry.json"]
    )
    assert step4.registry_path == Path("custom_registry.json")

    step6 = ArgumentParser.parse_step_arguments(
        "6_validation.py", ["--profile", "--strict"]
    )
    assert step6.profile is True
    assert step6.strict is True

    step9 = ArgumentParser.parse_step_arguments(
        "9_advanced_viz.py",
        [
            "--viz-type",
            "network",
            "--no-interactive",
            "--export-formats",
            "html",
            "json",
        ],
    )
    assert step9.viz_type == "network"
    assert step9.interactive is False
    assert step9.export_formats == ["html", "json"]

    step22 = ArgumentParser.parse_step_arguments(
        "22_gui.py",
        ["--interactive", "--gui-types", "gui_1,oxdraw", "--open-browser"],
    )
    assert step22.interactive is True
    assert step22.gui_types == "gui_1,oxdraw"
    assert step22.open_browser is True

    step24 = ArgumentParser.parse_step_arguments(
        "24_intelligent_analysis.py",
        [
            "--analysis-model",
            "local-test",
            "--bottleneck-threshold",
            "12.5",
            "--skip-llm",
        ],
    )
    assert step24.analysis_model == "local-test"
    assert step24.bottleneck_threshold == 12.5
    assert step24.skip_llm is True


def test_setup_optional_group_arguments_match_docs() -> None:
    args = ArgumentParser.parse_step_arguments(
        "1_setup.py",
        ["--install-optional", "--optional-groups", "gui,audio"],
    )
    assert args.install_optional is True
    assert args.optional_groups == "gui,audio"
    assert args.target_dir == Path("input/gnn_files")


def test_step_specific_defaults_are_preserved() -> None:
    step9 = ArgumentParser.parse_step_arguments("9_advanced_viz.py", [])
    step22 = ArgumentParser.parse_step_arguments("22_gui.py", [])
    assert step9.interactive is True
    assert step22.interactive is False


def test_script_additional_args_do_not_trigger_recovery_parser(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cases = [
        (
            "0_template.py",
            ["--simulate-error"],
            {"simulate_error": {"type": bool, "help": "Simulate an error"}},
            "simulate_error",
            True,
        ),
        (
            "2_tests.py",
            ["--fast-only"],
            {
                "fast_only": {
                    "action": "store_true",
                    "default": True,
                    "flag": "--fast-only",
                }
            },
            "fast_only",
            True,
        ),
        (
            "4_model_registry.py",
            ["--registry-path", "registry.json"],
            {"registry_path": {"type": str, "help": "Registry path"}},
            "registry_path",
            Path("registry.json"),
        ),
        (
            "6_validation.py",
            ["--profile", "--strict"],
            {
                "profile": {"type": bool, "help": "Profile"},
                "strict": {"type": bool, "help": "Strict"},
            },
            "profile",
            True,
        ),
    ]

    caplog.set_level(logging.WARNING)
    for step_name, argv, additional, attr_name, expected in cases:
        caplog.clear()
        monkeypatch.setattr(sys, "argv", [step_name, *argv])
        parsed = _parse_step_args(step_name, "test parser", additional)
        assert getattr(parsed, attr_name) == expected
        assert not any("using recovery parser" in r.message for r in caplog.records)


def test_invalid_step_argument_fails_instead_of_silent_defaults() -> None:
    with pytest.raises(SystemExit):
        ArgumentParser.parse_step_arguments("22_gui.py", ["--not-a-real-flag"])


def test_only_steps_dependency_closure_is_recursive() -> None:
    assert resolve_step_dependencies([23]) == [3, 8, 13, 23]
    assert resolve_step_dependencies([17]) == list(range(3, 18))
    assert resolve_step_dependencies([24]) == list(range(25))
