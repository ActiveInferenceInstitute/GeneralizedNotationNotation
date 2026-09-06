"""
Streamlined Argument Handling for GNN Processing Pipeline.

Provides coherent argument parsing, validation, and passing across
all pipeline steps with centralized configuration and type safety.

This module is the public entry point for argument handling. All
implementations live in focused submodules; this module re-exports the
full public API so existing imports continue to work unchanged:

- arg_definitions: ArgumentDefinition
- pipeline_arguments: PipelineArguments
- step_config: StepConfiguration
- arg_parsing: ArgumentParser, StepAwareArgumentParser, parse helpers
- path_conversion: path/argument normalization helpers
"""

import logging

from .arg_definitions import ArgumentDefinition
from .arg_parsing import (
    ArgumentParser,
    StepAwareArgumentParser,
    audit_step_contracts,
    build_step_command_args,
    get_pipeline_step_info,
    parse_arguments,
    parse_step_arguments,
    parse_step_list,
    validate_arguments,
)
from .path_conversion import (
    convert_path_arguments,
    validate_and_convert_paths,
    validate_pipeline_configuration,
)
from .pipeline_arguments import PipelineArguments
from .step_config import StepConfiguration

logger = logging.getLogger(__name__)

__all__ = [
    "ArgumentDefinition",
    "ArgumentParser",
    "PipelineArguments",
    "StepAwareArgumentParser",
    "StepConfiguration",
    "audit_step_contracts",
    "build_step_command_args",
    "convert_path_arguments",
    "get_pipeline_step_info",
    "parse_arguments",
    "parse_step_arguments",
    "parse_step_list",
    "validate_and_convert_paths",
    "validate_arguments",
    "validate_pipeline_configuration",
    "logger",
]
