"""
Streamlined Argument Handling for GNN Processing Pipeline.

Provides coherent argument parsing, validation, and passing across
all pipeline steps with centralized configuration and type safety.
"""

import argparse
import logging
import re
import sys
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, cast

from .arg_definitions import ArgumentDefinition
from .config_loader import GNNPipelineConfig, load_config
from .error_handling import is_critical_pipeline_step
from .pipeline_arguments import PipelineArguments
from .step_config import StepConfiguration

# Preserve the original logger identity so log records and external
# logging configuration keep targeting the "utils.argument_utils" logger.
logger = logging.getLogger("utils.argument_utils")


class ArgumentParser:
    """Centralized argument parser for the GNN pipeline."""

    # Define all available arguments
    ARGUMENT_DEFINITIONS = MappingProxyType(
        {
            "target_dir": ArgumentDefinition(
                flag="--target-dir",
                arg_type=Path,
                default=Path("input/gnn_files"),
                help_text="Target directory for GNN files",
            ),
            "output_dir": ArgumentDefinition(
                flag="--output-dir",
                arg_type=Path,
                default=Path("output"),
                help_text="Directory to save outputs",
            ),
            "recursive": ArgumentDefinition(
                flag="--recursive",
                action=argparse.BooleanOptionalAction,
                default=True,
                help_text="Recursively process directories",
            ),
            "verbose": ArgumentDefinition(
                flag="--verbose",
                action=argparse.BooleanOptionalAction,
                default=False,
                help_text="Enable verbose output",
            ),
            "log_format": ArgumentDefinition(
                flag="--log-format",
                arg_type=str,
                choices=["human", "json"],
                default="human",
                help_text="Output format for pipeline logs",
            ),
            "enable_round_trip": ArgumentDefinition(
                flag="--enable-round-trip",
                action="store_true",
                help_text="Enable comprehensive round-trip testing across all 21 formats",
            ),
            "enable_cross_format": ArgumentDefinition(
                flag="--enable-cross-format",
                action="store_true",
                help_text="Enable cross-format consistency validation",
            ),
            "skip_steps": ArgumentDefinition(
                flag="--skip-steps",
                default=None,
                help_text="Comma-separated list of steps to skip",
            ),
            "only_steps": ArgumentDefinition(
                flag="--only-steps",
                default=None,
                help_text="Comma-separated list of steps to run exclusively",
            ),
            "parallel": ArgumentDefinition(
                flag="--parallel",
                action="store_true",
                default=False,
                help_text="Execute independent steps within topological tiers in parallel",
            ),
            "autonomous": ArgumentDefinition(
                flag="--autonomous",
                action="store_true",
                default=False,
                help_text=(
                    "Write bounded autonomous proposal artifacts under output/ "
                    "without editing source files"
                ),
            ),
            "skip_llm": ArgumentDefinition(
                flag="--skip-llm",
                action="store_true",
                help_text="Skip LLM-powered processing where supported",
            ),
            "strict": ArgumentDefinition(
                flag="--strict", action="store_true", help_text="Enable strict mode"
            ),
            "profile": ArgumentDefinition(
                flag="--profile",
                action="store_true",
                help_text="Enable performance profiling",
            ),
            "simulate_error": ArgumentDefinition(
                flag="--simulate-error",
                action="store_true",
                use_suppress=True,
                help_text="Simulate an error for testing",
            ),
            "registry_path": ArgumentDefinition(
                flag="--registry-path",
                arg_type=Path,
                default=None,
                help_text="Path to model registry file",
            ),
            "query_ontology": ArgumentDefinition(
                flag="--query-ontology",
                arg_type=str,
                default=None,
                help_text="Filter registered models by ontology concept substring",
            ),
            "estimate_resources": ArgumentDefinition(
                flag="--estimate-resources",
                action=argparse.BooleanOptionalAction,
                default=False,
                help_text="Estimate computational resources",
            ),
            "ontology_terms_file": ArgumentDefinition(
                flag="--ontology-terms-file",
                arg_type=Path,
                help_text="Path to ontology terms file",
            ),
            "pipeline_summary_file": ArgumentDefinition(
                flag="--pipeline-summary-file",
                arg_type=Path,
                help_text="Path to save pipeline summary",
            ),
            "llm_tasks": ArgumentDefinition(
                flag="--llm-tasks", help_text="Comma-separated list of LLM tasks"
            ),
            "llm_timeout": ArgumentDefinition(
                flag="--llm-timeout",
                arg_type=int,
                help_text="Timeout for LLM processing in seconds",
            ),
            "website_html_filename": ArgumentDefinition(
                flag="--website-html-filename",
                help_text="Filename for generated HTML website",
            ),
            "performance_mode": ArgumentDefinition(
                flag="--performance-mode",
                arg_type=str,
                default="low",
                help_text="Performance mode for applicable steps (low, medium, high)",
                choices=["low", "medium", "high"],
            ),
            "mcp_strict_validation": ArgumentDefinition(
                flag="--mcp-strict-validation",
                action="store_true",
                help_text="MCP (step 21): enforce JSON-schema validation on every tool call",
            ),
            "mcp_cache_ttl": ArgumentDefinition(
                flag="--mcp-cache-ttl",
                arg_type=float,
                help_text="MCP (step 21): result-cache TTL in seconds (default 300)",
            ),
            "mcp_per_module_timeout": ArgumentDefinition(
                flag="--mcp-per-module-timeout",
                arg_type=float,
                help_text="MCP (step 21): max seconds to wait per module during discovery (default 30)",
            ),
            "mcp_overall_timeout": ArgumentDefinition(
                flag="--mcp-overall-timeout",
                arg_type=float,
                help_text="MCP (step 21): overall wall-clock budget for parallel discovery (default 120)",
            ),
            "mcp_modules_allowlist": ArgumentDefinition(
                flag="--mcp-modules-allowlist",
                arg_type=str,
                help_text="MCP (step 21): comma-separated module names to restrict discovery to",
            ),
            "frameworks": ArgumentDefinition(
                flag="--frameworks",
                arg_type=str,
                default="all",
                help_text=(
                    "Frameworks to execute/render (all, lite, or comma-separated list: "
                    "pymdp,rxinfer,activeinference_jl,jax,discopy,pytorch,numpyro,stan,bnlearn)"
                ),
            ),
            "strict_framework_success": ArgumentDefinition(
                flag="--strict-framework-success",
                action="store_true",
                help_text="Render step: fail if any requested framework render fails",
            ),
            "render_output_dir": ArgumentDefinition(
                flag="--render-output-dir",
                arg_type=Path,
                default=None,
                help_text="Explicit path to 11_render_output directory (avoids filesystem heuristics)",
            ),
            "distributed": ArgumentDefinition(
                flag="--distributed",
                action="store_true",
                default=False,
                help_text="Enable distributed execution for step 12 (if supported)",
            ),
            "execution_workers": ArgumentDefinition(
                flag="--execution-workers",
                arg_type=int,
                default=1,
                help_text="Number of local or distributed workers for step 12 execution",
            ),
            "backend": ArgumentDefinition(
                flag="--backend",
                arg_type=str,
                default="ray",
                choices=["ray", "dask"],
                help_text="Distributed backend for step 12 (ray or dask)",
            ),
            "serialize_preset": ArgumentDefinition(
                flag="--serialize-preset",
                arg_type=str,
                default="full",
                choices=["full", "minimal"],
                help_text="Step 3: serialization preset (full=all formats; minimal=markdown+json+python)",
            ),
            "execution_benchmark_repeats": ArgumentDefinition(
                flag="--execution-benchmark-repeats",
                arg_type=int,
                default=1,
                help_text="Step 12: sequential benchmark repeats per script; median runtime when >1",
            ),
            "execution_summary_detail": ArgumentDefinition(
                flag="--execution-summary-detail",
                action=argparse.BooleanOptionalAction,
                default=False,
                help_text="Step 12: also write execution_summary_detail.json with full per-script payloads",
            ),
            "recreate_venv": ArgumentDefinition(
                flag="--recreate-uv-env",
                action="store_true",
                use_suppress=True,
                help_text="Recreate UV virtual environment",
            ),
            "dev": ArgumentDefinition(
                flag="--dev",
                action="store_true",
                use_suppress=True,
                help_text="Install development dependencies (uv sync --extra dev)",
            ),
            "install_all_extras": ArgumentDefinition(
                flag="--install-all-extras",
                action="store_true",
                help_text="Install all optional dependency groups (uv sync --all-extras)",
            ),
            "setup_core_only": ArgumentDefinition(
                flag="--setup-core-only",
                action="store_true",
                help_text=(
                    "Step 1: skip the post-sync JAX/PyMDP self-test after installing core dependencies"
                ),
            ),
            "duration": ArgumentDefinition(
                flag="--duration",
                arg_type=float,
                default=30.0,
                help_text="Audio duration in seconds for audio generation",
            ),
            "audio_backend": ArgumentDefinition(
                flag="--audio-backend",
                arg_type=str,
                default="auto",
                help_text="Audio backend to use (auto, sapf, pedalboard, default: auto)",
            ),
            "sonification": ArgumentDefinition(
                flag="--sonification",
                action=argparse.BooleanOptionalAction,
                default=True,
                help_text="Generate model sonification",
            ),
            "full_analysis": ArgumentDefinition(
                flag="--full-analysis",
                action="store_true",
                default=False,
                help_text="Run full audio analysis",
            ),
            "fast_only": ArgumentDefinition(
                flag="--fast-only",
                action="store_true",
                use_suppress=True,
                help_text="Run only fast tests, skip slow and performance tests",
            ),
            "include_performance": ArgumentDefinition(
                flag="--include-performance",
                action="store_true",
                help_text="Include performance test categories",
            ),
            "comprehensive": ArgumentDefinition(
                flag="--comprehensive",
                action="store_true",
                use_suppress=True,
                help_text="Run all test categories including comprehensive suite",
            ),
            "install_optional": ArgumentDefinition(
                flag="--install-optional",
                action="store_true",
                help_text="Install optional dependency groups",
            ),
            "optional_groups": ArgumentDefinition(
                flag="--optional-groups",
                default=None,
                help_text="Comma-separated optional dependency groups to install, e.g. gui,audio",
            ),
            "viz_type": ArgumentDefinition(
                flag="--viz-type",
                arg_type=str,
                default="all",
                choices=[
                    "all",
                    "3d",
                    "interactive",
                    "dashboard",
                    "d2",
                    "diagrams",
                    "pipeline",
                    "statistical",
                    "pomdp",
                    "network",
                ],
                help_text="Step 9: type of advanced visualization to generate",
            ),
            "interactive": ArgumentDefinition(
                flag="--interactive",
                action=argparse.BooleanOptionalAction,
                default=False,
                help_text="Enable interactive mode where supported",
            ),
            "export_formats": ArgumentDefinition(
                flag="--export-formats",
                arg_type=str,
                default=["html", "json"],
                nargs="+",
                help_text="Step 9: visualization export formats",
            ),
            "headless": ArgumentDefinition(
                flag="--headless",
                action="store_true",
                default=False,
                help_text="Step 22: run GUI processors in headless artifact mode",
            ),
            "gui_types": ArgumentDefinition(
                flag="--gui-types",
                arg_type=str,
                default="gui_1,gui_2",
                help_text="Step 22: comma-separated GUI processors to run",
            ),
            "open_browser": ArgumentDefinition(
                flag="--open-browser",
                action="store_true",
                default=False,
                help_text="Step 22: open browser for interactive GUIs",
            ),
            "analysis_model": ArgumentDefinition(
                flag="--analysis-model",
                arg_type=str,
                default=None,
                help_text="Step 24: LLM model tag for intelligent analysis",
            ),
            "bottleneck_threshold": ArgumentDefinition(
                flag="--bottleneck-threshold",
                arg_type=float,
                default=60.0,
                help_text="Step 24: duration threshold in seconds for bottleneck detection",
            ),
            "timesteps": ArgumentDefinition(
                flag="--timesteps",
                arg_type=int,
                default=None,
                help_text="Number of timesteps for simulation",
            ),
            "simulation_params": ArgumentDefinition(
                flag="--simulation-params",
                arg_type=str,
                default="{}",
                help_text="JSON string containing simulation parameters",
            ),
            "timeout": ArgumentDefinition(
                flag="--timeout",
                arg_type=int,
                default=300,
                help_text="Timeout for execution in seconds",
            ),
            "advanced_stats": ArgumentDefinition(
                flag="--advanced-stats",
                action="store_true",
                help_text="Include advanced statistical analysis",
            ),
            "generate_animations": ArgumentDefinition(
                flag="--no-animations",
                action="store_false",
                default=True,
                dest="generate_animations",
                help_text=(
                    "Disable Step 16 GridWorld GIF animation artifacts "
                    "(enabled by default)"
                ),
            ),
        }
    )

    # Define which arguments each step supports
    STEP_ARGUMENTS = MappingProxyType(
        {
            "0_template.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "simulate_error",
            ],
            "1_setup.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "recreate_venv",
                "dev",
                "install_all_extras",
                "setup_core_only",
                "install_optional",
                "optional_groups",
            ],
            "2_tests.py": [
                "target_dir",
                "output_dir",
                "verbose",
                "fast_only",
                "include_performance",
                "comprehensive",
            ],
            "3_gnn.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "enable_round_trip",
                "enable_cross_format",
                "serialize_preset",
            ],
            "4_model_registry.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "registry_path",
                "query_ontology",
            ],
            "5_type_checker.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "strict",
                "estimate_resources",
            ],
            "6_validation.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "strict",
                "profile",
            ],
            "7_export.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "8_visualization.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "9_advanced_viz.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "viz_type",
                "interactive",
                "export_formats",
            ],
            "10_ontology.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "ontology_terms_file",
            ],
            "11_render.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "timesteps",
                "simulation_params",
                "frameworks",
                "strict_framework_success",
            ],
            "12_execute.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "frameworks",
                "timeout",
                "render_output_dir",
                "distributed",
                "execution_workers",
                "backend",
                "execution_benchmark_repeats",
                "execution_summary_detail",
            ],
            "13_llm.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "llm_tasks",
                "llm_timeout",
            ],
            "14_ml_integration.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
            ],
            "15_audio.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "duration",
                "audio_backend",
                "sonification",
                "full_analysis",
            ],
            "16_analysis.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "advanced_stats",
                "generate_animations",
            ],
            "17_integration.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "18_security.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "19_research.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "20_website.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "website_html_filename",
            ],
            "21_mcp.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "performance_mode",
                "mcp_strict_validation",
                "mcp_cache_ttl",
                "mcp_per_module_timeout",
                "mcp_overall_timeout",
                "mcp_modules_allowlist",
            ],
            "22_gui.py": [
                "target_dir",
                "output_dir",
                "recursive",
                "verbose",
                "headless",
                "interactive",
                "gui_types",
                "open_browser",
            ],
            "23_report.py": ["target_dir", "output_dir", "recursive", "verbose"],
            "24_intelligent_analysis.py": [
                "target_dir",
                "output_dir",
                "verbose",
                "analysis_model",
                "skip_llm",
                "bottleneck_threshold",
            ],
            "main.py": list(ARGUMENT_DEFINITIONS.keys()),
        }
    )

    @classmethod
    def create_main_parser(cls) -> argparse.ArgumentParser:
        """Create the main pipeline argument parser with all arguments."""
        parser = argparse.ArgumentParser(
            description="GNN Processing Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add all arguments
        for _, arg_def in cls.ARGUMENT_DEFINITIONS.items():
            arg_def.add_to_parser(parser)

        return parser

    @classmethod
    def create_step_parser(
        cls, step_name: str, description: Optional[str] = None
    ) -> argparse.ArgumentParser:
        """Create a parser for a specific pipeline step."""
        if description is None:
            description = f"GNN Processing Pipeline - {step_name}"

        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add arguments supported by this step
        supported_args = cls.STEP_ARGUMENTS.get(step_name, [])
        config_key = (
            step_name.replace(".py", "") if step_name.endswith(".py") else step_name
        )
        try:
            step_defaults = StepConfiguration.get_step_config(config_key).get(
                "defaults", {}
            )
        except NameError:
            step_defaults = {}

        for arg_name in supported_args:
            if arg_name in cls.ARGUMENT_DEFINITIONS:
                arg_def = cls.ARGUMENT_DEFINITIONS[arg_name]
                if arg_name in step_defaults:
                    arg_def = ArgumentDefinition(
                        flag=arg_def.flag,
                        arg_type=arg_def.arg_type,
                        default=step_defaults[arg_name],
                        required=arg_def.required,
                        help_text=arg_def.help_text,
                        choices=arg_def.choices,
                        action=arg_def.action,
                        nargs=arg_def.nargs,
                        dest=arg_def.dest,
                        use_suppress=arg_def.use_suppress,
                    )
                arg_def.add_to_parser(parser)
            else:
                logger.warning(f"Unknown argument '{arg_name}' for step {step_name}")

        return parser

    @classmethod
    def parse_main_arguments(
        cls, args: Optional[List[str]] = None
    ) -> PipelineArguments:
        """Parse main pipeline arguments and return PipelineArguments object."""
        parser = cls.create_main_parser()
        parsed = parser.parse_args(args)

        # Convert to PipelineArguments (only attributes argparse actually set)
        field_names = {f.name for f in fields(PipelineArguments)}
        kwargs = {k: getattr(parsed, k) for k in field_names if hasattr(parsed, k)}
        pipeline_args = PipelineArguments(**kwargs)

        # Validate arguments
        validation_errors = pipeline_args.validate()
        if validation_errors:
            logger.error("Argument validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Invalid arguments: {'; '.join(validation_errors)}")

        return pipeline_args

    @classmethod
    def parse_step_arguments(
        cls, step_name: str, args: Optional[List[str]] = None
    ) -> argparse.Namespace:
        """Parse arguments for a specific step with guaranteed attribute availability."""
        parser = cls.create_step_parser(step_name)

        try:
            parsed_args = parser.parse_args(args)

            # CRITICAL FIX: Ensure all expected attributes exist with proper defaults
            # This addresses the 'recursive' attribute missing issue in step 13
            step_supported_args = cls.STEP_ARGUMENTS.get(step_name, [])

            for arg_name in step_supported_args:
                if not hasattr(parsed_args, arg_name):
                    # Set appropriate default values
                    if arg_name == "recursive":
                        setattr(parsed_args, arg_name, True)
                    elif arg_name == "verbose":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "strict":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "estimate_resources":
                        setattr(parsed_args, arg_name, True)
                    elif arg_name.endswith("_dir"):
                        setattr(
                            parsed_args,
                            arg_name,
                            Path("output")
                            if "output" in arg_name
                            else Path("input/gnn_files"),
                        )
                    elif arg_name == "llm_timeout":
                        setattr(parsed_args, arg_name, 360)
                    elif arg_name == "llm_tasks":
                        setattr(parsed_args, arg_name, "all")
                    elif arg_name == "website_html_filename":
                        setattr(
                            parsed_args, arg_name, "gnn_pipeline_summary_website.html"
                        )
                    elif arg_name in [
                        "recreate_venv",
                        "dev",
                        "setup_core_only",
                        "install_all_extras",
                        "install_optional",
                        "headless",
                        "open_browser",
                        "skip_llm",
                        "simulate_error",
                        "profile",
                    ]:
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "optional_groups":
                        setattr(parsed_args, arg_name, None)
                    elif arg_name == "viz_type":
                        setattr(parsed_args, arg_name, "all")
                    elif arg_name == "interactive":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "export_formats":
                        setattr(parsed_args, arg_name, ["html", "json"])
                    elif arg_name == "gui_types":
                        setattr(parsed_args, arg_name, "gui_1,gui_2")
                    elif arg_name == "analysis_model":
                        setattr(parsed_args, arg_name, None)
                    elif arg_name == "bottleneck_threshold":
                        setattr(parsed_args, arg_name, 60.0)
                    elif arg_name == "fast_only":
                        setattr(parsed_args, arg_name, True)
                    elif arg_name == "comprehensive":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "duration":
                        setattr(parsed_args, arg_name, 30.0)
                    elif arg_name == "timesteps":
                        setattr(parsed_args, arg_name, None)
                    elif arg_name == "simulation_params":
                        setattr(parsed_args, arg_name, "{}")
                    elif arg_name == "timeout":
                        setattr(parsed_args, arg_name, 300)
                    elif arg_name == "serialize_preset":
                        setattr(parsed_args, arg_name, "full")
                    elif arg_name == "execution_benchmark_repeats":
                        setattr(parsed_args, arg_name, 1)
                    elif arg_name == "execution_summary_detail":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "execution_workers":
                        setattr(parsed_args, arg_name, 1)
                    elif arg_name == "advanced_stats":
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == "generate_animations":
                        setattr(parsed_args, arg_name, True)
                    else:
                        setattr(parsed_args, arg_name, None)

            return parsed_args

        except SystemExit as e:
            # argparse raises SystemExit(0) for --help; must propagate for correct CLI exit code
            code = e.code
            if code == 0 or code is None:
                raise
            logger.error(f"Argument parsing failed for step {step_name}: {e}")
            raise
        except Exception:
            raise

    @classmethod
    def create_default_namespace(cls, step_name: str) -> argparse.Namespace:
        """Create a namespace populated with registered defaults for recovery callers."""
        fallback_args = argparse.Namespace()
        step_supported_args = cls.STEP_ARGUMENTS.get(step_name, [])

        for arg_name in step_supported_args:
            if arg_name == "recursive":
                setattr(fallback_args, arg_name, True)
            elif arg_name == "verbose":
                setattr(fallback_args, arg_name, False)
            elif arg_name == "strict":
                setattr(fallback_args, arg_name, False)
            elif arg_name == "estimate_resources":
                setattr(fallback_args, arg_name, True)
            elif arg_name.endswith("_dir"):
                setattr(
                    fallback_args,
                    arg_name,
                    Path("output") if "output" in arg_name else Path("input/gnn_files"),
                )
            elif arg_name == "llm_timeout":
                setattr(fallback_args, arg_name, 360)
            elif arg_name == "llm_tasks":
                setattr(fallback_args, arg_name, "all")
            elif arg_name == "execution_workers":
                setattr(fallback_args, arg_name, 1)
            elif arg_name == "website_html_filename":
                setattr(fallback_args, arg_name, "gnn_pipeline_summary_website.html")
            elif arg_name in [
                "recreate_venv",
                "dev",
                "setup_core_only",
                "install_all_extras",
                "install_optional",
                "headless",
                "open_browser",
                "skip_llm",
                "simulate_error",
                "profile",
            ]:
                setattr(fallback_args, arg_name, False)
            elif arg_name == "optional_groups":
                setattr(fallback_args, arg_name, None)
            elif arg_name == "viz_type":
                setattr(fallback_args, arg_name, "all")
            elif arg_name == "interactive":
                setattr(fallback_args, arg_name, False)
            elif arg_name == "export_formats":
                setattr(fallback_args, arg_name, ["html", "json"])
            elif arg_name == "gui_types":
                setattr(fallback_args, arg_name, "gui_1,gui_2")
            elif arg_name == "analysis_model":
                setattr(fallback_args, arg_name, None)
            elif arg_name == "bottleneck_threshold":
                setattr(fallback_args, arg_name, 60.0)
            elif arg_name == "fast_only":
                setattr(fallback_args, arg_name, True)
            elif arg_name == "comprehensive":
                setattr(fallback_args, arg_name, False)
            elif arg_name == "duration":
                setattr(fallback_args, arg_name, 30.0)
            elif arg_name == "timeout":
                setattr(fallback_args, arg_name, 300)
            elif arg_name == "serialize_preset":
                setattr(fallback_args, arg_name, "full")
            elif arg_name == "execution_benchmark_repeats":
                setattr(fallback_args, arg_name, 1)
            elif arg_name == "execution_summary_detail":
                setattr(fallback_args, arg_name, False)
            elif arg_name == "generate_animations":
                setattr(fallback_args, arg_name, True)
            else:
                setattr(fallback_args, arg_name, None)

        return fallback_args


class StepAwareArgumentParser:
    """Argument parser with step-specific validation and defaults."""

    @classmethod
    def create_step_parser(
        cls, step_name: str, description: Optional[str] = None
    ) -> argparse.ArgumentParser:
        """Create a parser for a specific pipeline step."""
        # Remove .py extension for config lookup if present
        config_key = (
            step_name.replace(".py", "") if step_name.endswith(".py") else step_name
        )
        config = StepConfiguration.get_step_config(config_key)

        if description is None:
            description = config.get(
                "description", f"GNN Processing Pipeline - {step_name}"
            )

        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
Examples:
  # Basic usage
  python {step_name}.py --target-dir gnn/examples --output-dir ../output
  
  # Verbose mode
  python {step_name}.py --target-dir gnn/examples --output-dir ../output --verbose
  
  # See main pipeline help for all options
  python main.py --help
            """,
        )

        # Add arguments supported by this step
        # Check both with and without .py extension for STEP_ARGUMENTS lookup
        step_args_key = (
            step_name
            if step_name in ArgumentParser.STEP_ARGUMENTS
            else f"{step_name}.py"
        )
        supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_args_key, [])
        for arg_name in supported_args:
            if arg_name in ArgumentParser.ARGUMENT_DEFINITIONS:
                # Get step-specific default if available
                step_defaults = config.get("defaults", {})
                arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg_name]

                # Override default with step-specific value without losing metadata.
                if arg_name in step_defaults:
                    arg_def.with_default(step_defaults[arg_name]).add_to_parser(parser)
                else:
                    arg_def.add_to_parser(parser)
            else:
                logger.warning(f"Unknown argument '{arg_name}' for step {step_name}")

        return parser

    @classmethod
    def parse_step_arguments(
        cls, step_name: str, args: Optional[List[str]] = None
    ) -> argparse.Namespace:
        """Parse and validate arguments for a specific pipeline step."""
        parser = cls.create_step_parser(step_name)
        parsed_args = parser.parse_args(args)

        # Validate step-specific requirements
        config_key = (
            step_name.replace(".py", "") if step_name.endswith(".py") else step_name
        )
        validation_errors = StepConfiguration.validate_step_args(
            config_key, parsed_args
        )
        if validation_errors:
            logger.error(f"Argument validation failed for {step_name}:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError(
                f"Invalid arguments for {step_name}: {'; '.join(validation_errors)}"
            )

        return parsed_args

    @classmethod
    def get_step_help(cls, step_name: str) -> str:
        """Get help text for a specific step."""
        config_key = (
            step_name.replace(".py", "") if step_name.endswith(".py") else step_name
        )
        config = StepConfiguration.get_step_config(config_key)
        if not config:
            return f"Unknown step: {step_name}"

        help_text: list[Any] = [
            f"Step {step_name}: {config.get('description', 'No description available')}"
        ]

        # Required arguments
        req_args = config.get("required_args", [])
        if req_args:
            help_text.append("\nRequired arguments:")
            for arg in req_args:
                if arg in ArgumentParser.ARGUMENT_DEFINITIONS:
                    arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg]
                    help_text.append(f"  {arg_def.flag}: {arg_def.help_text}")

        # Optional arguments
        opt_args = config.get("optional_args", [])
        if opt_args:
            help_text.append("\nOptional arguments:")
            for arg in opt_args:
                if arg in ArgumentParser.ARGUMENT_DEFINITIONS:
                    arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg]
                    default = config.get("defaults", {}).get(arg, arg_def.default)
                    help_text.append(
                        f"  {arg_def.flag}: {arg_def.help_text} (default: {default})"
                    )

        return "\n".join(help_text)


# Command building with validation
def build_step_command_args(
    step_name: str,
    pipeline_args: PipelineArguments,
    python_executable: str,
    script_path: Path,
) -> List[str]:
    """
    Build validated command line arguments for a pipeline step.

    Args:
        step_name: Name of the step (e.g., "1_gnn")
        pipeline_args: Main pipeline arguments
        python_executable: Path to Python executable
        script_path: Path to the step script

    Returns:
        List of command line arguments

    Raises:
        ValueError: If step configuration is invalid
    """
    # Validate step exists
    # Strip .py extension for lookup in STEP_CONFIGS
    config_key = (
        step_name.replace(".py", "") if step_name.endswith(".py") else step_name
    )
    # Also try with .py extension for STEP_ARGUMENTS lookup
    step_key = f"{config_key}.py" if not step_name.endswith(".py") else step_name
    config = StepConfiguration.get_step_config(config_key)
    if not config:
        raise ValueError(f"Unknown pipeline step: {step_name}")

    cmd: list[Any] = [python_executable, str(script_path)]

    # Get all arguments this step supports
    # First try from StepConfiguration
    all_supported_args = config.get("required_args", []) + config.get(
        "optional_args", []
    )

    # If no arguments found, try from STEP_ARGUMENTS as recovery
    if not all_supported_args and step_key in ArgumentParser.STEP_ARGUMENTS:
        all_supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_key, [])

    # Build arguments from pipeline configuration
    for arg_name in all_supported_args:
        if hasattr(pipeline_args, arg_name):
            arg_value = getattr(pipeline_args, arg_name)

            # Step 12: omit misleading defaults when unused
            if step_key == "12_execute.py":
                if arg_name == "backend" and not getattr(
                    pipeline_args, "distributed", False
                ):
                    continue
                if arg_name == "execution_benchmark_repeats":
                    try:
                        if int(arg_value) <= 1:
                            continue
                    except (TypeError, ValueError):
                        continue

            # Skip None values for optional arguments
            if arg_value is None and arg_name not in config.get("required_args", []):
                continue

            # Get argument definition for proper formatting
            if arg_name in ArgumentParser.ARGUMENT_DEFINITIONS:
                arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg_name]
                flag = arg_def.flag

                # Handle different argument types
                if arg_def.action == "store_true":
                    if arg_value:
                        cmd.append(flag)
                elif arg_def.action == "store_false":
                    if arg_value is False:
                        cmd.append(flag)
                elif arg_def.action is argparse.BooleanOptionalAction:
                    if arg_value is True:
                        cmd.append(flag)
                    elif arg_value is False and arg_def.default is True:
                        cmd.append(f"--no-{flag.removeprefix('--')}")
                elif isinstance(arg_value, list):
                    cmd.append(flag)
                    cmd.extend(str(item) for item in arg_value)
                else:
                    # Regular arguments with values
                    cmd.extend([flag, str(arg_value)])

    return cmd


# Utility for step introspection
def get_pipeline_step_info() -> Dict[str, Any]:
    """Get comprehensive information about all pipeline steps."""
    step_info: dict[Any, Any] = {}

    for step_name, config in StepConfiguration.STEP_CONFIGS.items():
        step_info[step_name] = {
            "description": config.get("description", ""),
            "required_args": config.get("required_args", []),
            "optional_args": config.get("optional_args", []),
            "defaults": config.get("defaults", {}),
            "critical": config.get("critical", False),
            "total_args": len(cast("list[Any]", config.get("required_args", [])))
            + len(cast("list[Any]", config.get("optional_args", []))),
        }

    return step_info


def audit_step_contracts(
    python_executable: Optional[str] = None, script_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Audit shared step contracts without executing any numbered step.

    The audit is intentionally static: it verifies that the declarative step
    config, parser defaults, critical-step metadata, and command builder agree.
    """
    issues: list[dict[str, Any]] = []
    python_executable = python_executable or sys.executable
    script_dir = script_dir or Path("src")

    for step_name, config in StepConfiguration.STEP_CONFIGS.items():
        step_key = f"{step_name}.py"
        configured_args = cast("list[Any]", config.get("required_args", [])) + cast(
            "list[Any]", config.get("optional_args", [])
        )
        declared_args = ArgumentParser.STEP_ARGUMENTS.get(step_key)
        if declared_args is None:
            issues.append(
                {
                    "step": step_name,
                    "kind": "missing_step_arguments",
                    "message": f"{step_key} missing from STEP_ARGUMENTS",
                }
            )
        elif list(declared_args) != list(configured_args):
            issues.append(
                {
                    "step": step_name,
                    "kind": "argument_mismatch",
                    "message": "STEP_ARGUMENTS and StepConfiguration differ",
                    "step_arguments": list(declared_args),
                    "step_configuration": list(configured_args),
                }
            )

        expected_critical = is_critical_pipeline_step(step_name)
        if bool(config.get("critical", False)) != expected_critical:
            issues.append(
                {
                    "step": step_name,
                    "kind": "critical_mismatch",
                    "message": "critical metadata does not match canonical set",
                    "expected": expected_critical,
                    "actual": bool(config.get("critical", False)),
                }
            )

        parser = ArgumentParser.create_step_parser(step_key)
        try:
            parsed_defaults = parser.parse_args([])
        except SystemExit as exc:
            issues.append(
                {
                    "step": step_name,
                    "kind": "default_parse_error",
                    "message": f"default parser exited with {exc.code}",
                }
            )
            continue

        for arg_name, expected in cast(
            "dict[str, Any]", config.get("defaults", {})
        ).items():
            if hasattr(parsed_defaults, arg_name):
                actual = getattr(parsed_defaults, arg_name)
                if actual != expected:
                    issues.append(
                        {
                            "step": step_name,
                            "kind": "default_mismatch",
                            "argument": arg_name,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

        if "recursive" in configured_args:
            recursive_enabled = parser.parse_args(["--recursive"])
            recursive_disabled = parser.parse_args(["--no-recursive"])
            if getattr(recursive_enabled, "recursive", None) is not True:
                issues.append(
                    {
                        "step": step_name,
                        "kind": "recursive_parse_mismatch",
                        "message": "--recursive did not parse to True",
                    }
                )
            if getattr(recursive_disabled, "recursive", None) is not False:
                issues.append(
                    {
                        "step": step_name,
                        "kind": "recursive_parse_mismatch",
                        "message": "--no-recursive did not parse to False",
                    }
                )

            disabled_cmd = build_step_command_args(
                step_name,
                PipelineArguments(recursive=False),
                python_executable,
                script_dir / step_key,
            )
            if "--no-recursive" not in disabled_cmd:
                issues.append(
                    {
                        "step": step_name,
                        "kind": "recursive_command_mismatch",
                        "message": "recursive=False did not propagate as --no-recursive",
                    }
                )

    step16_cmd = build_step_command_args(
        "16_analysis",
        PipelineArguments(generate_animations=False),
        python_executable,
        script_dir / "16_analysis.py",
    )
    if "--no-animations" not in step16_cmd:
        issues.append(
            {
                "step": "16_analysis",
                "kind": "animation_command_mismatch",
                "message": "generate_animations=False did not propagate as --no-animations",
            }
        )

    return issues


# Validation utility for the entire pipeline
def parse_step_arguments(
    step_name: str, args: Optional[List[str]] = None
) -> argparse.Namespace:
    """Parse arguments for a specific pipeline step (standalone function)."""
    return ArgumentParser.parse_step_arguments(step_name, args)


def validate_arguments(args: argparse.Namespace) -> List[str]:
    """Validate parsed arguments and return list of errors."""
    errors: list[Any] = []

    # Basic validation
    if hasattr(args, "target_dir") and args.target_dir:
        if not Path(args.target_dir).exists():
            errors.append(f"Target directory does not exist: {args.target_dir}")

    return errors


def parse_step_list(step_str: str) -> List[int]:
    """Parse a comma-separated list of step numbers."""
    if not step_str:
        return []

    steps: list[Any] = []
    for item in step_str.split(","):
        item = item.strip()
        # Extract number from formats like "1", "1_gnn", etc.
        match = re.match(r"^(\d+)", item)
        if match:
            steps.append(int(match.group(1)))
    return steps


def parse_arguments() -> PipelineArguments:
    """Parse command line arguments and load configuration."""
    # Create argument parser for command line options
    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline with YAML configuration support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add configuration file option
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("input/config.yaml"),
        help="Path to configuration file (default: input/config.yaml)",
    )

    # Add all other options that can override config
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Target directory for GNN files (overrides config)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory to save outputs (overrides config)"
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        help="Recursively process directories (overrides config)",
    )
    parser.add_argument(
        "--skip-steps", help="Comma-separated list of steps to skip (overrides config)"
    )
    parser.add_argument(
        "--only-steps", help="Comma-separated list of steps to run (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Enable verbose output (overrides config)",
    )
    parser.add_argument(
        "--enable-round-trip",
        action="store_true",
        help="Enable comprehensive round-trip testing across all 21 formats (overrides config)",
    )
    parser.add_argument(
        "--enable-cross-format",
        action="store_true",
        help="Enable cross-format consistency validation (overrides config)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict type checking mode"
    )
    parser.add_argument(
        "--estimate-resources",
        action=argparse.BooleanOptionalAction,
        help="Estimate computational resources",
    )
    parser.add_argument(
        "--ontology-terms-file",
        type=Path,
        help="Path to ontology terms file (overrides config)",
    )
    parser.add_argument("--llm-tasks", help="Comma-separated list of LLM tasks")
    parser.add_argument(
        "--llm-timeout", type=int, help="Timeout for LLM processing in seconds"
    )
    parser.add_argument(
        "--pipeline-summary-file", type=Path, help="Path to save pipeline summary"
    )
    parser.add_argument(
        "--website-html-filename", help="Filename for generated HTML website"
    )
    parser.add_argument(
        "--duration", type=float, help="Audio duration in seconds for audio generation"
    )
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="auto",
        help="Audio backend to use (auto, sapf, pedalboard, default: auto)",
    )
    parser.add_argument(
        "--recreate-venv", action="store_true", help="Recreate virtual environment"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Install development dependencies"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM processing step (alias for --skip-steps 13)",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Handle --skip-llm convenience flag
    if args.skip_llm:
        existing = args.skip_steps or ""
        existing_nums = (
            [s.strip() for s in existing.split(",") if s.strip()] if existing else []
        )
        if "13" not in existing_nums:
            existing_nums.append("13")
        args.skip_steps = ",".join(existing_nums)

    # Determine project root - this should be the parent of the src directory
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    project_root = (
        current_dir.parent.parent
    )  # Go up from utils/ to src/ to project root

    # Load configuration from YAML file
    try:
        # Resolve config file path relative to project root
        config_path = args.config_file
        if not config_path.is_absolute():
            config_path = project_root / config_path

        # Load the actual configuration from YAML file
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.warning(f"Failed to load configuration from {args.config_file}: {e}")
        logger.info("Using default configuration")
        config = GNNPipelineConfig()

    # Convert config to PipelineArguments
    pipeline_args = PipelineArguments()

    # Set values from config
    config_dict = config.to_pipeline_arguments()
    for key, value in config_dict.items():
        if hasattr(pipeline_args, key):
            setattr(pipeline_args, key, value)

    # Resolve relative paths relative to project root
    if not pipeline_args.target_dir.is_absolute():
        pipeline_args.target_dir = project_root / pipeline_args.target_dir
    if not pipeline_args.output_dir.is_absolute():
        pipeline_args.output_dir = project_root / pipeline_args.output_dir
    if (
        pipeline_args.ontology_terms_file
        and not pipeline_args.ontology_terms_file.is_absolute()
    ):
        pipeline_args.ontology_terms_file = (
            project_root / pipeline_args.ontology_terms_file
        )
    if (
        pipeline_args.pipeline_summary_file
        and not pipeline_args.pipeline_summary_file.is_absolute()
    ):
        pipeline_args.pipeline_summary_file = (
            project_root / pipeline_args.pipeline_summary_file
        )

    # Override with command line arguments if provided
    if args.target_dir is not None:
        pipeline_args.target_dir = args.target_dir
    if args.output_dir is not None:
        pipeline_args.output_dir = args.output_dir
    if args.recursive is not None:
        pipeline_args.recursive = args.recursive
    if args.skip_steps is not None:
        pipeline_args.skip_steps = args.skip_steps
    if args.only_steps is not None:
        pipeline_args.only_steps = args.only_steps
    if args.verbose is not None:
        pipeline_args.verbose = args.verbose
    if args.enable_round_trip:
        pipeline_args.enable_round_trip = True
    if args.enable_cross_format:
        pipeline_args.enable_cross_format = True
    if args.strict:
        pipeline_args.strict = True
    if args.estimate_resources is not None:
        pipeline_args.estimate_resources = args.estimate_resources
    if args.ontology_terms_file is not None:
        pipeline_args.ontology_terms_file = args.ontology_terms_file
    if args.llm_tasks is not None:
        pipeline_args.llm_tasks = args.llm_tasks
    if args.llm_timeout is not None:
        pipeline_args.llm_timeout = args.llm_timeout
    if args.pipeline_summary_file is not None:
        pipeline_args.pipeline_summary_file = args.pipeline_summary_file
    if args.website_html_filename is not None:
        pipeline_args.website_html_filename = args.website_html_filename
    if args.duration is not None:
        pipeline_args.duration = args.duration
    if args.recreate_venv:
        pipeline_args.recreate_venv = True
    if args.dev:
        pipeline_args.dev = True
    if getattr(args, "setup_core_only", False):
        pipeline_args.setup_core_only = True

    # Resolve relative paths relative to input directory
    input_dir = Path("input")
    # If target_dir is relative, make it relative to input directory, but avoid double prefixing
    if not pipeline_args.target_dir.is_absolute():
        target_str = str(pipeline_args.target_dir)
        if not target_str.startswith("input/"):
            pipeline_args.target_dir = input_dir / pipeline_args.target_dir

    return pipeline_args
