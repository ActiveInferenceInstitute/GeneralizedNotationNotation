#!/usr/bin/env python3
"""
Pipeline Module Template

This template shows the recommended structure for all GNN pipeline modules.
Copy this structure for consistent argument handling, logging, and error management.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Any, Callable, List, Dict
try:
    from src.pipeline.config import get_output_dir_for_script
except ImportError:
    from pipeline.config import get_output_dir_for_script

# Standard import pattern for all pipeline modules
try:
    from utils.structured_logging import (  # noqa: F401 - standard pipeline imports
        log_step_start,
        log_step_success,
        log_step_warning,
        log_step_error,
    )
    from utils.pipeline import (
        setup_step_logging,
        validate_output_directory,
    )
    UTILS_AVAILABLE = True

except ImportError:
    # Recovery: use step_logging (always importable, no external deps)
    from utils.logging.logging_utils import (  # noqa: F401 - standard pipeline imports
        setup_step_logging,
        log_step_start,
        log_step_success,
        log_step_warning,
        log_step_error,
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



def create_standardized_pipeline_script(
    step_name: str,
    module_function: Callable,
    fallback_parser_description: str,
    additional_arguments: Optional[Dict[str, Any]] = None,
    step_specific_imports: Optional[List[str]] = None,
    default_target_dir: Optional[str] = None,
    default_recursive: bool = False,
) -> Callable:
    """
    Create a standardized pipeline script with consistent argument parsing and error handling.

    Args:
        step_name: Name of the step (e.g., "1_gnn", "5_export")
        module_function: The main function to call for processing
        fallback_parser_description: Description for recovery argument parser
        additional_arguments: Additional arguments to add to the parser
        step_specific_imports: Additional imports needed for the step
        default_target_dir: Default target directory when none is provided on CLI
        default_recursive: Default recursive flag value

    Returns:
        Function that can be called to run the standardized script
    """
    def run_standardized_script():
        """Standardized script execution function."""
        try:
            # Import step-specific modules if provided
            if step_specific_imports:
                for import_name in step_specific_imports:
                    try:
                        __import__(import_name)
                    except ImportError as e:
                        logging.warning(f"Failed to import {import_name}: {e}")

            # Parse arguments - try enhanced first, fall back gracefully
            try:
                # Import enhanced parser locally to avoid import issues
                from utils import ArgumentParser
                parsed_args = ArgumentParser.parse_step_arguments(step_name)
            except Exception as e:
                # Create recovery parser with step-specific arguments
                fallback_additional_args = additional_arguments or {}

                logging.warning(f"Enhanced parser failed for {step_name}, using recovery: {e}")
                parsed_args = _create_fallback_parser(fallback_parser_description, fallback_additional_args).parse_args()

            # Set up logging
            logger = setup_step_logging(step_name, getattr(parsed_args, 'verbose', False))

            # Convert paths with proper handling for None values
            target_dir_raw = getattr(parsed_args, 'target_dir', None)
            if target_dir_raw is None:
                target_dir = Path(default_target_dir) if default_target_dir else Path('input/gnn_files')
            else:
                target_dir = Path(target_dir_raw)

            output_dir_raw = getattr(parsed_args, 'output_dir', 'output')
            output_dir = Path(output_dir_raw) if output_dir_raw is not None else Path('output')

            recursive_default = default_recursive

            # Normalize the output directory to the standardized numbered step folder
            try:
                normalized_step = step_name if not step_name.endswith('.py') else step_name[:-3]
                step_output_dir = get_output_dir_for_script(normalized_step, output_dir)
            except Exception:
                step_output_dir = output_dir

            # Call the module function with the standardized step output dir
            success = module_function(
                target_dir=target_dir,
                output_dir=step_output_dir,
                logger=logger,
                recursive=getattr(parsed_args, 'recursive', recursive_default),
                verbose=getattr(parsed_args, 'verbose', False),
                **{k: v for k, v in vars(parsed_args).items()
                   if k not in ['target_dir', 'output_dir', 'recursive', 'verbose']}
            )

            return 0 if success else 1

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
