"""Direct tests for the ``create_standardized_pipeline_script`` factory.

The factory is the thin-orchestrator backbone: every numbered pipeline script
passes its module function to it, and the factory owns argument parsing,
logging, per-step output-directory resolution, and exit-code coercion
(0=success, 1=error, 2=success with warnings). These tests exercise the real
factory with lightweight module functions.
"""

import sys
from typing import Any, Callable, Dict

import pytest

from utils.pipeline_template import create_standardized_pipeline_script


@pytest.mark.unit
class TestStandardizedPipelineScriptFactory:
    """Factory dispatch contract (real ``create_standardized_pipeline_script``)."""

    def test_dispatches_standard_kwargs_to_module_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Standard CLI args reach the module function as kwargs."""
        calls: Dict[str, Any] = {}

        def delegate(
            target_dir: Any = None,
            output_dir: Any = None,
            logger: Any = None,
            recursive: bool = False,
            verbose: bool = False,
            **kwargs: Any,
        ) -> bool:
            calls.update(
                target_dir=str(target_dir),
                output_dir=str(output_dir),
                verbose=verbose,
            )
            return True

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--target-dir",
                "input/gnn_files",
                "--output-dir",
                "/tmp/factory_out",
                "--verbose",
            ],
        )
        run_script = create_standardized_pipeline_script(
            "0_template.py", delegate, "test description"
        )
        assert run_script() == 0
        assert calls["target_dir"] == "input/gnn_files"
        assert calls["verbose"] is True
        # Per-step output subdirectory is resolved under the base output dir.
        assert calls["output_dir"].endswith("0_template_output")

    @pytest.mark.parametrize(
        ("module_function", "expected_exit"),
        [
            (lambda **kw: True, 0),
            (lambda **kw: False, 1),
            (lambda **kw: 2, 2),
        ],
    )
    def test_exit_code_coercion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        module_function: Callable[..., Any],
        expected_exit: int,
    ) -> None:
        """bool and int returns coerce to canonical exit codes (0/1/2)."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--target-dir",
                "input/gnn_files",
                "--output-dir",
                "/tmp/factory_out",
            ],
        )
        run_script = create_standardized_pipeline_script(
            "0_template.py", module_function, "test description"
        )
        assert run_script() == expected_exit
