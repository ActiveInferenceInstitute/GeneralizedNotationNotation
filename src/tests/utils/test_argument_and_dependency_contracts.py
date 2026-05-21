"""Interface guardrails for step arguments and dependency closure."""

from pathlib import Path

import pytest

from utils.argument_utils import ArgumentParser
from utils.pipeline_step_dependencies import resolve_step_dependencies


def test_step_specific_arguments_are_exposed() -> None:
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


def test_invalid_step_argument_fails_instead_of_silent_defaults() -> None:
    with pytest.raises(SystemExit):
        ArgumentParser.parse_step_arguments("22_gui.py", ["--not-a-real-flag"])


def test_only_steps_dependency_closure_is_recursive() -> None:
    assert resolve_step_dependencies([23]) == [3, 8, 13, 23]
    assert resolve_step_dependencies([17]) == list(range(3, 18))
    assert resolve_step_dependencies([24]) == list(range(25))
