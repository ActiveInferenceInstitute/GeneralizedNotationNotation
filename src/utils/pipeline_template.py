#!/usr/bin/env python3
"""
Pipeline Module Template

This template shows the recommended structure for all GNN pipeline modules.
Copy this structure for consistent argument handling, logging, and error management.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from src.pipeline.config import get_output_dir_for_script
except ImportError:
    from pipeline.config import get_output_dir_for_script

# Standard import pattern for all pipeline modules
try:
    from utils.pipeline import (
        setup_step_logging,
        validate_output_directory,
    )
    from utils.structured_logging import (  # noqa: F401 - standard pipeline imports
        log_step_error,
        log_step_start,
        log_step_success,
        log_step_warning,
    )
    UTILS_AVAILABLE = True

except ImportError:
    # Recovery: use step_logging (always importable, no external deps)
    from utils.logging.logging_utils import (  # noqa: F401 - standard pipeline imports
        log_step_error,
        log_step_start,
        log_step_success,
        log_step_warning,
        setup_step_logging,
    )

    def validate_output_directory(output_dir, step_name):
        try:
            (output_dir / f"{step_name}_step").mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    UTILS_AVAILABLE = False

def _create_fallback_parser(description: str, additional_arguments: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """Create a recovery argument parser with standard arguments."""
    parser = argparse.ArgumentParser(description=description)

    # Standard arguments - use None for defaults that get set by pipeline template
    parser.add_argument("--target-dir", type=Path, default=None,
                       help="Target directory containing files to process")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                       help="Output directory for generated artifacts")
    parser.add_argument("--recursive", action="store_true",
                       help="Process files recursively")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    # Add additional arguments if provided
    if additional_arguments:
        for arg_name, arg_config in additional_arguments.items():
            if isinstance(arg_config, dict):
                # Use custom flag if provided, otherwise default to --{arg_name}
                config_copy = arg_config.copy()
                flag = config_copy.pop("flag", f"--{arg_name}")
                parser.add_argument(flag, **config_copy)
            else:
                parser.add_argument(f"--{arg_name}", default=arg_config)

    return parser



def _preload_step_imports(step_specific_imports: Optional[List[str]]) -> None:
    """Eagerly import step-specific modules; log (don't raise) on ImportError."""
    if not step_specific_imports:
        return
    for import_name in step_specific_imports:
        try:
            __import__(import_name)
        except ImportError as e:
            logging.warning(f"Failed to import {import_name}: {e}")


def _parse_step_args(step_name: str, fallback_parser_description: str,
                     additional_arguments: Optional[Dict[str, Any]]) -> argparse.Namespace:
    """Try the enhanced argument parser; fall back to the recovery parser on any failure."""
    try:
        from utils import ArgumentParser
        return ArgumentParser.parse_step_arguments(step_name)
    except Exception as e:
        logging.warning(f"Enhanced parser failed for {step_name}, using recovery: {e}")
        return _create_fallback_parser(fallback_parser_description, additional_arguments or {}).parse_args()


def _resolve_dirs(parsed_args: argparse.Namespace, step_name: str,
                  default_target_dir: Optional[str]) -> tuple[Path, Path]:
    """Resolve target_dir and the per-step output_dir from parsed args."""
    target_dir_raw = getattr(parsed_args, 'target_dir', None)
    if target_dir_raw is None:
        target_dir = Path(default_target_dir) if default_target_dir else Path('input/gnn_files')
    else:
        target_dir = Path(target_dir_raw)

    output_dir_raw = getattr(parsed_args, 'output_dir', 'output')
    output_dir = Path(output_dir_raw) if output_dir_raw is not None else Path('output')

    try:
        normalized_step = step_name if not step_name.endswith('.py') else step_name[:-3]
        step_output_dir = get_output_dir_for_script(normalized_step, output_dir)
    except Exception:
        step_output_dir = output_dir

    return target_dir, step_output_dir


def _coerce_exit_code(result: Any, step_name: str, logger: logging.Logger) -> int:
    """Coerce module_function return value to an exit code.

    Contract: module_function may return bool or int.
      bool: True→0 (success), False→1 (error)
      int: passthrough (0=success, 1=error, 2=warnings/skipped)
    Other truthy/falsy types are coerced via bool() for backward compatibility.
    """
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        if result == 2:
            log_step_warning(logger, f"{step_name} completed with warnings (exit 2)")
        return result
    return 0 if bool(result) else 1


def create_standardized_pipeline_script(
    step_name: str,
    module_function: Callable,
    fallback_parser_description: str,
    additional_arguments: Optional[Dict[str, Any]] = None,
    step_specific_imports: Optional[List[str]] = None,
    default_target_dir: Optional[str] = None,
    default_recursive: bool = False,
) -> Callable:
    """Create a standardized pipeline script with consistent arg parsing and error handling.

    The returned callable, when invoked, parses CLI args, sets up logging, resolves the
    per-step output directory, calls ``module_function``, and returns the coerced exit
    code. See ``_coerce_exit_code`` for the return-value contract.
    """
    def run_standardized_script():
        try:
            _preload_step_imports(step_specific_imports)
            parsed_args = _parse_step_args(step_name, fallback_parser_description, additional_arguments)
            logger = setup_step_logging(step_name, getattr(parsed_args, 'verbose', False))
            target_dir, step_output_dir = _resolve_dirs(parsed_args, step_name, default_target_dir)

            result = module_function(
                target_dir=target_dir,
                output_dir=step_output_dir,
                logger=logger,
                recursive=getattr(parsed_args, 'recursive', default_recursive),
                verbose=getattr(parsed_args, 'verbose', False),
                **{k: v for k, v in vars(parsed_args).items()
                   if k not in ['target_dir', 'output_dir', 'recursive', 'verbose']}
            )
            return _coerce_exit_code(result, step_name, logger)

        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return 1

    return run_standardized_script


# Common additional parameters
CommonModuleParams = {
    "strict": bool,  # For type checking
    "estimate_resources": bool,  # For type checking
    "duration": float,  # For SAPF
    "llm_tasks": str,  # For LLM
    "llm_timeout": int,  # For LLM
    "ontology_terms_file": Path,  # For ontology
    "recreate_venv": bool,  # For setup
    "dev": bool,  # For setup
    "install_all_extras": bool,  # For setup: uv sync --all-extras
    "setup_core_only": bool,  # For setup: skip execution-frameworks
}

# Standardized naming conventions for module functions
STANDARD_MODULE_FUNCTION_NAMES = {
    "1_setup": "process_setup",
    "2_tests": "run_tests",
    "3_gnn": "process_gnn_files",
    "4_model_registry": "process_model_registry",
    "5_type_checker": "process_type_checking",
    "6_validation": "process_validation",
    "7_export": "process_export",
    "8_visualization": "process_visualization",
    "9_advanced_viz": "process_advanced_viz",
    "10_ontology": "process_ontology",
    "11_render": "process_render",
    "12_execute": "process_execute",
    "13_llm": "process_llm",
    "14_ml_integration": "process_ml_integration",
    "15_audio": "process_audio",
    "16_analysis": "process_analysis",
    "17_integration": "process_integration",
    "18_security": "process_security",
    "19_research": "process_research",
    "20_website": "process_website",
    "21_mcp": "process_mcp",
    "22_gui": "process_gui",
    "23_report": "process_report",
    "24_intelligent_analysis": "process_intelligent_analysis"
}

# Standard additional arguments for each step
STEP_ADDITIONAL_ARGUMENTS = {
    "5_type_checker": {
        "strict": {"type": bool, "help": "Enable strict validation mode"},
        "estimate_resources": {"type": bool, "help": "Estimate computational resources"}
    },
    "13_llm": {
        "llm_tasks": {"type": str, "default": "all", "help": "Comma-separated list of LLM tasks"},
        "llm_timeout": {"type": int, "default": 360, "help": "Timeout for LLM operations in seconds"}
    },
    "20_website": {
        "website_html_filename": {"type": str, "default": "gnn_pipeline_summary_website.html", "help": "Filename for generated HTML website"}
    },
    "23_report": {
        # No additional arguments for report generation
    },
    "10_ontology": {
        "ontology_terms_file": {"type": Path, "help": "Path to ontology terms JSON file", "flag": "--ontology-terms-file"}
    },
    "1_setup": {
        "recreate_venv": {"type": bool, "help": "Recreate virtual environment"},
        "dev": {"type": bool, "help": "Install development dependencies (uv sync --extra dev)"},
        "install_all_extras": {"type": bool, "help": "Install all optional groups (uv sync --all-extras)"},
    },
    "21_mcp.py": {
        "performance-mode": {
            "type": str,
            "default": "low",
            "choices": ["low", "high"],
            "help": "Performance mode for MCP",
            "flag": "--performance-mode"
        }
    }
}
