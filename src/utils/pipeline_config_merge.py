"""
Merge ``input/config.yaml`` defaults into PipelineArguments after CLI parse.

CLI flags that use argparse.SUPPRESS omit attributes when absent; this module
fills those from YAML so config file defaults apply without fighting argparse
false defaults.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from utils.argument_utils import PipelineArguments


def apply_input_config_defaults(
    args: PipelineArguments,
    full_config: Dict[str, Any],
    parsed: Optional[argparse.Namespace],
) -> None:
    """
    Apply setup/pipeline sections from ``full_config`` when the CLI did not
    set the corresponding option (namespace attribute missing).

    Precedence: explicit CLI > YAML > dataclass defaults (already on ``args``).
    """
    if not full_config or parsed is None:
        return

    setup_cfg = full_config.get("setup") or {}
    uv_block = full_config.get("uv") or {}
    uv_sync = uv_block.get("sync") or {}
    pipe_cfg = full_config.get("pipeline") or {}

    if not hasattr(parsed, "dev"):
        cfg_dev = setup_cfg.get("dev")
        if cfg_dev is None:
            cfg_dev = uv_sync.get("dev")
        if cfg_dev is not None:
            args.dev = bool(cfg_dev)

    if not hasattr(parsed, "install_all_extras"):
        iae = setup_cfg.get("install_all_extras")
        if iae is not None:
            args.install_all_extras = bool(iae)

    if not hasattr(parsed, "recreate_venv"):
        rv = setup_cfg.get("recreate_venv")
        if rv is None:
            rv = uv_block.get("recreate_uv_env")
        if rv is not None:
            args.recreate_venv = bool(rv)

    if not hasattr(parsed, "fast_only"):
        fo = pipe_cfg.get("fast_only")
        args.fast_only = True if fo is None else bool(fo)

    if not hasattr(parsed, "comprehensive"):
        comp = pipe_cfg.get("comprehensive")
        if comp is None:
            comp = pipe_cfg.get("comprehensive_tests")
        if comp is not None:
            args.comprehensive = bool(comp)
