"""Generic dispatcher for the ``process_<module>_mcp`` pipeline-step tools.

Every pipeline module historically carried a near-identical MCP wrapper:
convert the ``target_directory`` / ``output_directory`` strings to paths, call
the module's ``process_*`` step, shape a success dict with a human-readable
message, and convert any exception into ``{"success": False, "error": str(e)}``
behind an error log line. MAJ-06 replaces those copies with this dispatcher
plus thin per-module registrations.

MCP-client-visible behavior is unchanged by construction: tool names, input
schemas, and descriptions live in each module's ``register_tools`` (touched
nowhere), and the success / error dict shapes produced here match the
previously hand-written wrappers key for key. The live surface is pinned by
``src/gnn/mcp/audit_report.json``.

Variation points (all observed across the pre-consolidation wrappers):

- ``resolve_paths``: custom repo-path resolution instead of bare ``Path()``
  (e.g. ``execute`` requires a render summary; ``llm`` resolves repo paths).
- ``extra_step_kwargs``: static extra kwargs, or a callable of the resolved
  paths when an extra kwarg must echo a resolved path (``execute``).
- ``pass_verbose``: ``False`` when the step callable takes no ``verbose``
  parameter (``advanced_visualization``).
- ``interpret_result``: non-bool step results (``render`` / ``execute`` return
  ``bool | int``, ``website`` returns ``dict | bool``) mapped to
  ``(success, extra result keys, message or None)``; also used to compute
  post-run result keys (``visualization`` output file count).
- ``label`` / ``success_wording`` / ``failure_wording`` / ``message_builder``:
  the message line, whose wording varied per module. ``label=None`` omits the
  ``message`` key (``report``).
- ``static_extras``: fixed extra result keys (``advanced_visualization``,
  ``gui``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from logging import Logger
from pathlib import Path
from typing import Any


def run_pipeline_step_mcp(
    step: Callable[..., Any],
    *,
    wrapper_name: str,
    logger: Logger,
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    resolve_paths: Callable[[str, str], tuple[Path, Path]] | None = None,
    extra_step_kwargs: Mapping[str, Any]
    | Callable[[Path, Path], Mapping[str, Any]]
    | None = None,
    echo_resolved: bool = False,
    pass_verbose: bool = True,
    interpret_result: Callable[[Any], tuple[bool, Mapping[str, Any], str | None]]
    | None = None,
    label: str | None = None,
    success_wording: str = "completed successfully",
    failure_wording: str = "completed with issues",
    message_builder: Callable[[bool], str] | None = None,
    static_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one pipeline step behind the canonical MCP wrapper envelope.

    The step is invoked as
    ``step(target_dir=<resolved target>, output_dir=<resolved output>, [verbose], **extra)``
    inside a try block. Success and error dicts mirror the previously
    hand-written wrappers exactly; ``message`` precedence is
    ``message_builder(success)`` over an ``interpret_result`` message over the
    ``label`` template over omission.
    """
    try:
        target_path, output_path = (
            resolve_paths(target_directory, output_directory)
            if resolve_paths is not None
            else (Path(target_directory), Path(output_directory))
        )
        step_kwargs: dict[str, Any] = (
            dict(extra_step_kwargs(target_path, output_path))
            if callable(extra_step_kwargs)
            else dict(extra_step_kwargs or {})
        )
        if pass_verbose:
            step_kwargs["verbose"] = verbose
        raw = step(target_dir=target_path, output_dir=output_path, **step_kwargs)
        if interpret_result is not None:
            success, extras, message = interpret_result(raw)
        else:
            success, extras, message = bool(raw), {}, None
        result: dict[str, Any] = {
            "success": success,
            "target_directory": str(target_path) if echo_resolved else target_directory,
            "output_directory": str(output_path) if echo_resolved else output_directory,
            **(static_extras or {}),
            **extras,
        }
        if message_builder is not None:
            result["message"] = message_builder(success)
        elif message is not None:
            result["message"] = message
        elif label is not None:
            result["message"] = (
                f"{label} {success_wording if success else failure_wording}"
            )
        return result
    except Exception as e:
        logger.error(f"{wrapper_name} error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
