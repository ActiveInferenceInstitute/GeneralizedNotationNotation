"""Earlier name; implementation moved to ``gnn/utils/arguments/arg_parsing.py``
(S2-33 Step 2)."""

import warnings

warnings.warn(
    "gnn.utils.arg_parsing is the earlier name; import gnn.utils.arguments.arg_parsing instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments.arg_parsing import (  # noqa: E402,F401
    ArgumentParser,
    PipelineStepInfo,
    StepAwareArgumentParser,
    audit_step_contracts,
    build_step_command_args,
    fallback_default_for,
    get_pipeline_step_info,
    logger,
    parse_arguments,
    parse_step_arguments,
    parse_step_list,
    validate_arguments,
)

__all__ = [
    "ArgumentParser",
    "PipelineStepInfo",
    "StepAwareArgumentParser",
    "audit_step_contracts",
    "build_step_command_args",
    "fallback_default_for",
    "get_pipeline_step_info",
    "logger",
    "parse_arguments",
    "parse_step_arguments",
    "parse_step_list",
    "validate_arguments",
]
