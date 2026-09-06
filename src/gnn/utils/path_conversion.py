"""
Path conversion and pipeline configuration validation.

Provides convert_path_arguments (string path to Path conversion),
validate_pipeline_configuration (pipeline configuration checked against
all step requirements), and validate_and_convert_paths (validated path
conversion for parsed pipeline arguments).
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from .pipeline_arguments import PipelineArguments
from .step_config import StepConfiguration

# Preserve the original logger identity so log records and external
# logging configuration keep targeting the "utils.argument_utils" logger.
logger = logging.getLogger("utils.argument_utils")


# Validation utility for the entire pipeline
def convert_path_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Convert string paths to Path objects in parsed arguments."""
    for attr_name in dir(args):
        if not attr_name.startswith("_"):
            attr_value = getattr(args, attr_name)
            if isinstance(attr_value, str) and (
                "dir" in attr_name or "path" in attr_name or "file" in attr_name
            ):
                setattr(args, attr_name, Path(attr_value))
    return args


def validate_pipeline_configuration(
    pipeline_args: PipelineArguments,
) -> Dict[str, List[str]]:
    """
    Validate pipeline configuration against all step requirements.

    Returns:
        Dictionary mapping step names to lists of validation errors
    """
    validation_results: dict[Any, Any] = {}

    for step_name in StepConfiguration.STEP_CONFIGS.keys():
        # Create a namespace from pipeline args for validation
        step_namespace = argparse.Namespace()
        config = StepConfiguration.get_step_config(step_name)
        all_args = config.get("required_args", []) + config.get("optional_args", [])

        for arg_name in all_args:
            if hasattr(pipeline_args, arg_name):
                setattr(step_namespace, arg_name, getattr(pipeline_args, arg_name))

        # Validate this step
        errors = StepConfiguration.validate_step_args(step_name, step_namespace)
        if errors:
            validation_results[step_name] = errors

    return validation_results


def validate_and_convert_paths(args: PipelineArguments, logger: logging.Logger) -> Any:
    """Validate and convert paths."""
    path_args_to_check: list[Any] = [
        "output_dir",
        "target_dir",
        "ontology_terms_file",
        "pipeline_summary_file",
    ]

    for arg_name in path_args_to_check:
        if not hasattr(args, arg_name):
            logger.debug(
                f"Argument --{arg_name.replace('_', '-')} not present in args namespace."
            )
            continue

        arg_value = getattr(args, arg_name)

        if arg_value is not None and not isinstance(arg_value, Path):
            logger.warning(
                f"Argument --{arg_name.replace('_', '-')} was unexpectedly a {type(arg_value).__name__} "
                f"(value: '{arg_value}') instead of pathlib.Path. Converting explicitly. "
                "This might indicate an issue with argument parsing configuration or an external override."
            )
            try:
                setattr(args, arg_name, Path(arg_value))
            except TypeError as e:
                logger.error(
                    f"Failed to convert argument --{arg_name.replace('_', '-')} (value: '{arg_value}') to Path: {e}. "
                    "This could be due to an unsuitable value for a path."
                )
                if arg_name in ["output_dir", "target_dir"]:
                    msg = f"Critical path argument --{arg_name.replace('_', '-')} could not be converted to Path."
                    logger.critical(msg)
                    raise ValueError(msg) from e
        elif arg_value is None and arg_name in ["output_dir", "target_dir"]:
            msg = (
                f"Critical path argument --{arg_name.replace('_', '-')} is None after parsing. "
                "This indicates a problem with default value setup in argparse."
            )
            logger.critical(msg)
            raise ValueError(msg)
