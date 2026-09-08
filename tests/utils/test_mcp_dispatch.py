"""Contract tests for ``gnn.utils.mcp_dispatch``.

22 registered MCP tools route through :func:`run_tool_envelope` and 18
pipeline wrappers through :func:`run_pipeline_step_mcp`; this module pins
the contract points that per-module wrapper tests exercise only
incidentally. A regression here would silently change wire behavior for
every dependent tool:

- raw (uncoerced) echo of the step's return value on ``success``;
- message precedence: ``message_builder`` > ``interpret_result`` message >
  ``label`` template > omission, including the fall-through when
  ``interpret_result`` returns ``None``;
- ``echo_resolved`` swapping the echoed strings for the resolved paths;
- ``resolve_paths`` receiving the raw argument strings;
- ``extra_step_kwargs`` in mapping and callable forms;
- ``pass_verbose=False`` omitting the ``verbose`` kwarg;
- static/dynamic extras merge order;
- the error envelope (log line + ``{"success": False, "error": str(e)}``)
  and early-return passthrough in :func:`run_tool_envelope`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from gnn.utils.mcp_dispatch import run_pipeline_step_mcp, run_tool_envelope

logger = logging.getLogger("test_mcp_dispatch")


@pytest.mark.unit
class TestRunPipelineStepCanonicalShape:
    """The success dict mirrors the pre-consolidation wrappers exactly."""

    def test_canonical_shape_and_key_order(self) -> None:
        def step(**kwargs: Any) -> bool:
            return True

        result = run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            label="X processing",
        )
        assert result == {
            "success": True,
            "target_directory": "in",
            "output_directory": "out",
            "message": "X processing completed successfully",
        }
        assert list(result) == [
            "success",
            "target_directory",
            "output_directory",
            "message",
        ]

    def test_raw_echo_no_bool_coercion(self) -> None:
        """A truthy non-bool return passes through uncoerced."""

        def step(**kwargs: Any) -> int:
            return 2

        result = run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            label="X processing",
        )
        assert result["success"] == 2
        assert result["message"] == "X processing completed successfully"

    def test_custom_wording(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            label="LLM processing",
            success_wording="completed",
            failure_wording="failed",
        )
        assert result["message"] == "LLM processing completed"

        result = run_pipeline_step_mcp(
            lambda **kwargs: False,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            label="Audio processing",
            failure_wording="failed",
        )
        assert result["message"] == "Audio processing failed"

    def test_no_message_key_without_label_or_builder(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
        )
        assert "message" not in result
        assert list(result) == ["success", "target_directory", "output_directory"]

    def test_step_receives_path_objects_and_verbose(self) -> None:
        captured: dict[str, Any] = {}

        def step(**kwargs: Any) -> bool:
            captured.update(kwargs)
            return True

        run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            verbose=True,
        )
        assert captured["target_dir"] == Path("in")
        assert captured["output_dir"] == Path("out")
        assert captured["verbose"] is True


@pytest.mark.unit
class TestRunPipelineStepMessagePrecedence:
    """message_builder > interpret_result message > label template > omitted."""

    def test_builder_overrides_everything(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            interpret_result=lambda raw: (bool(raw), {}, "from interpret"),
            label="From label",
            message_builder=lambda ok: "from builder",
        )
        assert result["message"] == "from builder"

    def test_interpret_message_overrides_label(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            interpret_result=lambda raw: (bool(raw), {}, "from interpret"),
            label="From label",
        )
        assert result["message"] == "from interpret"

    def test_interpret_none_message_falls_through_to_label(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            interpret_result=lambda raw: (bool(raw), {}, None),
            label="From label",
        )
        assert result["message"] == "From label completed successfully"

    def test_interpret_extras_merged_and_message_omitted_when_none(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: 0,  # int contract, like render/execute
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            interpret_result=lambda raw: (
                raw in (0, 2),
                {"skipped": raw == 2},
                None,
            ),
        )
        assert result["success"] is True
        assert result["skipped"] is False
        assert "message" not in result

    def test_static_extras_precede_interpret_extras(self) -> None:
        """Dynamic extras win on key conflict (later spread)."""
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            static_extras={"mode": "static"},
            interpret_result=lambda raw: (bool(raw), {"mode": "dynamic"}, None),
        )
        assert result["mode"] == "dynamic"


@pytest.mark.unit
class TestRunPipelineStepPathHooks:
    def test_resolve_paths_receives_raw_strings(self) -> None:
        seen: dict[str, str] = {}

        def resolve(target: str, output: str) -> tuple[Path, Path]:
            seen["target"], seen["output"] = target, output
            return Path("/repo/resolved_in"), Path("/repo/resolved_out")

        captured: dict[str, Any] = {}

        def step(**kwargs: Any) -> bool:
            captured.update(kwargs)
            return True

        run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="raw_in",
            output_directory="raw_out",
            resolve_paths=resolve,
        )
        assert seen == {"target": "raw_in", "output": "raw_out"}
        assert captured["target_dir"] == Path("/repo/resolved_in")
        assert captured["output_dir"] == Path("/repo/resolved_out")

    def test_echo_resolved_swaps_echoed_strings(self) -> None:
        result = run_pipeline_step_mcp(
            lambda **kwargs: True,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="raw_in",
            output_directory="raw_out",
            resolve_paths=lambda t, o: (Path("/repo/ri"), Path("/repo/ro")),
            echo_resolved=True,
        )
        assert result["target_directory"] == str(Path("/repo/ri"))
        assert result["output_directory"] == str(Path("/repo/ro"))

    def test_extra_step_kwargs_mapping_merged_into_call(self) -> None:
        captured: dict[str, Any] = {}

        def step(**kwargs: Any) -> bool:
            captured.update(kwargs)
            return True

        run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            extra_step_kwargs={"gui_types": "gui_1", "headless": True},
        )
        assert captured["gui_types"] == "gui_1"
        assert captured["headless"] is True
        assert captured["verbose"] is False

    def test_extra_step_kwargs_callable_receives_resolved_paths(self) -> None:
        captured: dict[str, Any] = {}
        callable_seen: list[tuple[Path, Path]] = []

        def step(**kwargs: Any) -> bool:
            captured.update(kwargs)
            return True

        def extra(target_path: Path, output_path: Path) -> dict[str, Any]:
            callable_seen.append((target_path, output_path))
            return {"render_output_dir": target_path}

        run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            resolve_paths=lambda t, o: (Path("/repo/ri"), Path("/repo/ro")),
            extra_step_kwargs=extra,
        )
        assert callable_seen == [(Path("/repo/ri"), Path("/repo/ro"))]
        assert captured["render_output_dir"] == Path("/repo/ri")

    def test_pass_verbose_false_omits_verbose_kwarg(self) -> None:
        captured: dict[str, Any] = {}

        def step(**kwargs: Any) -> bool:
            captured.update(kwargs)
            return True

        run_pipeline_step_mcp(
            step,
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            pass_verbose=False,
        )
        assert "verbose" not in captured
        assert captured["target_dir"] == Path("in")


@pytest.mark.unit
class TestRunPipelineStepErrorEnvelope:
    def test_exception_becomes_error_dict_and_is_logged(self, caplog: Any) -> None:
        def step(**kwargs: Any) -> bool:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            result = run_pipeline_step_mcp(
                step,
                wrapper_name="process_x_mcp",
                logger=logger,
                target_directory="in",
                output_directory="out",
                label="X processing",
            )
        assert result == {"success": False, "error": "boom"}
        assert list(result) == ["success", "error"]
        assert "process_x_mcp error: boom" in caplog.text

    def test_resolution_errors_are_caught_too(self, caplog: Any) -> None:
        """resolve_paths runs inside the try block (execute/llm contract)."""

        def resolve(target: str, output: str) -> tuple[Path, Path]:
            raise ValueError("bad path")

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            result = run_pipeline_step_mcp(
                lambda **kwargs: True,
                wrapper_name="process_x_mcp",
                logger=logger,
                target_directory="in",
                output_directory="out",
                resolve_paths=resolve,
            )
        assert result == {"success": False, "error": "bad path"}
        assert "process_x_mcp error: bad path" in caplog.text

    def test_error_log_goes_through_caller_logger(self, caplog: Any) -> None:
        """The caller's logger name is preserved (per-module log routing)."""

        def step(**kwargs: Any) -> bool:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            run_pipeline_step_mcp(
                step,
                wrapper_name="process_x_mcp",
                logger=logger,
                target_directory="in",
                output_directory="out",
            )
        assert any(r.name == "test_mcp_dispatch" for r in caplog.records)


@pytest.mark.unit
class TestRunToolEnvelope:
    def test_success_payload_passthrough(self) -> None:
        payload = {"success": True, "module": "x", "tools": ["a"]}

        result = run_tool_envelope(
            lambda: payload,
            wrapper_name="get_x_module_info_mcp",
            logger=logger,
        )
        assert result == payload
        assert result is not payload  # defensive copy

    def test_early_return_error_dict_passes_through(self) -> None:
        error = {"success": False, "error": "Directory not found: /x"}

        result = run_tool_envelope(
            lambda: error,
            wrapper_name="get_x_mcp",
            logger=logger,
        )
        assert result == error

    def test_exception_becomes_error_dict_and_is_logged(self, caplog: Any) -> None:
        def build() -> dict[str, Any]:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            result = run_tool_envelope(
                build,
                wrapper_name="get_x_mcp",
                logger=logger,
            )
        assert result == {"success": False, "error": "boom"}
        assert "get_x_mcp error: boom" in caplog.text


@pytest.mark.unit
class TestRunPipelineStepMessageBuilderOnFailure:
    def test_message_builder_receives_false_and_wins_over_label(self) -> None:
        """``message_builder`` also drives failure results (MAJ-06 branch at
        mcp_dispatch.py:102-103); its message wins over the label template."""
        calls: list[bool] = []

        def build(success: bool) -> str:
            calls.append(success)
            return f"rendered={'ok' if success else 'with issues'}"

        result = run_pipeline_step_mcp(
            lambda **kwargs: "renderer-died",
            wrapper_name="process_x_mcp",
            logger=logger,
            target_directory="in",
            output_directory="out",
            interpret_result=lambda raw: (False, {}, None),
            label="X processing",
            success_wording="completed successfully",
            failure_wording="completed with issues",
            message_builder=build,
        )
        assert calls == [False]
        assert result["success"] is False
        assert result["message"] == "rendered=with issues"


@pytest.mark.unit
class TestRunPipelineStepErrorEnvelopeExtras:
    def test_static_extras_dropped_when_step_raises(self, caplog: Any) -> None:
        """The error envelope is exactly ``{"success", "error"}``: neither
        ``static_extras`` nor step extras are merged into error results."""

        def step(**kwargs: Any) -> bool:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            result = run_pipeline_step_mcp(
                step,
                wrapper_name="process_x_mcp",
                logger=logger,
                target_directory="in",
                output_directory="out",
                static_extras={"static_key": "static-value"},
            )
        assert result == {"success": False, "error": "boom"}
        assert list(result) == ["success", "error"]

    def test_base_exception_escapes_the_envelope(self) -> None:
        """``except Exception`` deliberately lets ``KeyboardInterrupt`` /
        ``SystemExit`` propagate instead of swallowing them into an error
        dict (both dispatcher functions)."""

        def step(**kwargs: Any) -> bool:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_pipeline_step_mcp(
                step,
                wrapper_name="process_x_mcp",
                logger=logger,
                target_directory="in",
                output_directory="out",
            )


@pytest.mark.unit
class TestRunToolEnvelopeNegativePaths:
    def test_non_mapping_build_result_converted_to_error(self, caplog: Any) -> None:
        """``dict(build())`` on a non-Mapping raises inside the envelope and
        converts to the canonical error dict (accidental-input path)."""

        def bad_build() -> dict[str, Any]:
            return 42  # type: ignore[return-value] — accidental-input probe

        with caplog.at_level(logging.ERROR, logger="test_mcp_dispatch"):
            result = run_tool_envelope(
                bad_build,
                wrapper_name="get_x_mcp",
                logger=logger,
            )
        assert result["success"] is False
        assert "iterable" in result["error"]
        assert "get_x_mcp error:" in caplog.text

    def test_base_exception_escapes_the_envelope(self) -> None:
        def build() -> dict[str, Any]:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_tool_envelope(build, wrapper_name="get_x_mcp", logger=logger)
