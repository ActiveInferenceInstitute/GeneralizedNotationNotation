"""
Streamlined Argument Handling for GNN Processing Pipeline.

Provides coherent argument parsing, validation, and passing across
all pipeline steps with centralized configuration and type safety.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
import logging
import re
import sys

# Import config loading functionality
from .config_loader import load_config, GNNPipelineConfig

logger = logging.getLogger(__name__)

@dataclass
class ArgumentDefinition:
    """Definition of a pipeline argument with metadata."""
    flag: str
    arg_type: Type = str
    default: Any = None
    required: bool = False
    help_text: str = ""
    choices: Optional[List[str]] = None
    action: Optional[str] = None
    
    def add_to_parser(self, parser: argparse.ArgumentParser):
        """Add this argument to an ArgumentParser."""
        kwargs = {
            'help': self.help_text,
            'default': self.default
        }
        
        if self.action:
            kwargs['action'] = self.action
            if self.action in ['store_true', 'store_false']:
                # Boolean flags don't need type
                pass
            else:
                kwargs['type'] = self.arg_type
        else:
            kwargs['type'] = self.arg_type
            
        if self.required:
            kwargs['required'] = True
            
        if self.choices:
            kwargs['choices'] = self.choices
            
        parser.add_argument(self.flag, **kwargs)

@dataclass 
class PipelineArguments:
    """Centralized argument configuration for the entire pipeline."""
    
    # Core directories
    target_dir: Path = field(default_factory=lambda: Path("input/gnn_files"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Processing options
    recursive: bool = True
    verbose: bool = False
    
    # Validation options (enabled by default for comprehensive testing)
    enable_round_trip: bool = True      # Enable round-trip testing across all 21 formats
    enable_cross_format: bool = True    # Enable cross-format consistency validation
    
    # Step control  
    skip_steps: Optional[str] = None
    only_steps: Optional[str] = None
    
    # Type checking options
    strict: bool = False
    estimate_resources: bool = False
    
    # File references
    ontology_terms_file: Optional[Path] = None
    pipeline_summary_file: Optional[Path] = None
    
    # LLM options
    llm_tasks: str = "all"
    llm_timeout: int = 360
    
    # Website generation
    website_html_filename: str = "gnn_pipeline_summary_website.html"
    
    # Setup options
    recreate_venv: bool = False  # renamed from recreate_venv but keeping for compatibility
    dev: bool = False
    # Optional setup groups to install (comma-separated), used by step 1
    install_optional: Optional[str] = None
    
    # Audio generation options
    duration: float = 30.0
    audio_backend: str = "auto"
    
    # Test options
    fast_only: bool = False
    include_slow: bool = False
    include_performance: bool = False
    comprehensive: bool = False

    # MCP/performance mode (used by step 22)
    performance_mode: str = "low"
    
    def __post_init__(self):
        """Post-initialization validation and path resolution."""
        # Ensure Path objects
        if isinstance(self.target_dir, str):
            self.target_dir = Path(self.target_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        # Set defaults for optional paths
        if self.ontology_terms_file is None:
            self.ontology_terms_file = Path("src/ontology/act_inf_ontology_terms.json")
        elif isinstance(self.ontology_terms_file, str):
            self.ontology_terms_file = Path(self.ontology_terms_file)
            
        if self.pipeline_summary_file is None:
            self.pipeline_summary_file = self.output_dir / "pipeline_execution_summary.json"
        elif isinstance(self.pipeline_summary_file, str):
            self.pipeline_summary_file = Path(self.pipeline_summary_file)
    
    def validate(self) -> List[str]:
        """Validate argument values and return list of errors."""
        errors = []
        
        # Check that target directory exists if not a special placeholder
        if not str(self.target_dir).startswith("<"):
            # Try to resolve path relative to project root if not found
            if not self.target_dir.exists():
                # Check if we're running from src directory and target_dir is relative to project root
                import sys
                if hasattr(sys, '_getframe'):
                    try:
                        current_file = Path(sys._getframe(1).f_code.co_filename)
                        if current_file.name == 'main.py' and current_file.parent.name == 'src':
                            project_root = current_file.parent.parent
                            project_target_dir = project_root / self.target_dir.name
                            if project_target_dir.exists():
                                self.target_dir = project_target_dir
                            else:
                                errors.append(f"Target directory does not exist: {self.target_dir}")
                        else:
                            errors.append(f"Target directory does not exist: {self.target_dir}")
                    except (ValueError, AttributeError):
                        errors.append(f"Target directory does not exist: {self.target_dir}")
                else:
                    errors.append(f"Target directory does not exist: {self.target_dir}")
            
        # Check that ontology terms file exists if specified and not placeholder
        if (self.ontology_terms_file and 
            not str(self.ontology_terms_file).startswith("<") and 
            not self.ontology_terms_file.exists()):
            errors.append(f"Ontology terms file does not exist: {self.ontology_terms_file}")
            
        # Validate LLM timeout
        if self.llm_timeout <= 0:
            errors.append(f"LLM timeout must be positive: {self.llm_timeout}")
            
        # Validate step lists format
        if self.skip_steps:
            try:
                [s.strip() for s in self.skip_steps.split(',')]
            except Exception:
                errors.append(f"Invalid skip_steps format: {self.skip_steps}")
                
        if self.only_steps:
            try:
                [s.strip() for s in self.only_steps.split(',')]
            except Exception:
                errors.append(f"Invalid only_steps format: {self.only_steps}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with string representation of paths."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

class ArgumentParser:
    """Centralized argument parser for the GNN pipeline."""
    
    # Define all available arguments
    ARGUMENT_DEFINITIONS = {
        'target_dir': ArgumentDefinition(
            flag='--target-dir',
            arg_type=Path,
            default=Path("input/gnn_files"),
            help_text='Target directory for GNN files'
        ),
        'output_dir': ArgumentDefinition(
            flag='--output-dir', 
            arg_type=Path,
            default=Path("output"),
            help_text='Directory to save outputs'
        ),
        'recursive': ArgumentDefinition(
            flag='--recursive',
            action=argparse.BooleanOptionalAction,
            default=True,
            help_text='Recursively process directories'
        ),
        'verbose': ArgumentDefinition(
            flag='--verbose',
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text='Enable verbose output'
        ),
        'enable_round_trip': ArgumentDefinition(
            flag='--enable-round-trip',
            action='store_true',
            help_text='Enable comprehensive round-trip testing across all 21 formats'
        ),
        'enable_cross_format': ArgumentDefinition(
            flag='--enable-cross-format',
            action='store_true',
            help_text='Enable cross-format consistency validation'
        ),
        'skip_steps': ArgumentDefinition(
            flag='--skip-steps',
            default=[],
            help_text='Comma-separated list of steps to skip'
        ),
        'only_steps': ArgumentDefinition(
            flag='--only-steps',
            default=[],
            help_text='Comma-separated list of steps to run exclusively'
        ),
        'strict': ArgumentDefinition(
            flag='--strict',
            action='store_true',
            help_text='Enable strict mode'
        ),
        'estimate_resources': ArgumentDefinition(
            flag='--estimate-resources',
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text='Estimate computational resources'
        ),
        'ontology_terms_file': ArgumentDefinition(
            flag='--ontology-terms-file',
            arg_type=Path,
            help_text='Path to ontology terms file'
        ),
        'pipeline_summary_file': ArgumentDefinition(
            flag='--pipeline-summary-file',
            arg_type=Path,
            help_text='Path to save pipeline summary'
        ),
        'llm_tasks': ArgumentDefinition(
            flag='--llm-tasks',
            help_text='Comma-separated list of LLM tasks'
        ),
        'llm_timeout': ArgumentDefinition(
            flag='--llm-timeout',
            arg_type=int,
            help_text='Timeout for LLM processing in seconds'
        ),
        'website_html_filename': ArgumentDefinition(
            flag='--website-html-filename',
            help_text='Filename for generated HTML website'
        ),
        'performance_mode': ArgumentDefinition(
            flag='--performance-mode',
            arg_type=str,
            default='low',
            help_text='Performance mode for applicable steps (low, medium, high)',
            choices=['low', 'medium', 'high']
        ),
        'recreate_venv': ArgumentDefinition(
            flag='--recreate-uv-env',
            action='store_true',
            help_text='Recreate UV virtual environment'
        ),
        'dev': ArgumentDefinition(
            flag='--dev',
            action='store_true', 
            help_text='Install development dependencies'
        ),
        'duration': ArgumentDefinition(
            flag='--duration',
            arg_type=float,
            default=30.0,
            help_text='Audio duration in seconds for audio generation'
        ),
        'audio_backend': ArgumentDefinition(
            flag='--audio-backend',
            arg_type=str,
            default='auto',
            help_text='Audio backend to use (auto, sapf, pedalboard, default: auto)'
        ),
        'fast_only': ArgumentDefinition(
            flag='--fast-only',
            action='store_true',
            help_text='Run only fast tests, skip slow and performance tests'
        ),
        'include_slow': ArgumentDefinition(
            flag='--include-slow',
            action='store_true',
            help_text='Include slow test categories'
        ),
        'include_performance': ArgumentDefinition(
            flag='--include-performance',
            action='store_true',
            help_text='Include performance test categories'
        ),
        'comprehensive': ArgumentDefinition(
            flag='--comprehensive',
            action='store_true',
            help_text='Run all test categories including comprehensive suite'
        ),
        'install_optional': ArgumentDefinition(
            flag='--install-optional',
            help_text='Install optional package groups (comma-separated): ml_ai,llm,visualization,audio,graphs,research,active_inference'
        )
    }
    
    # Define which arguments each step supports
    STEP_ARGUMENTS = {
        "0_template.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "1_setup.py": ["target_dir", "output_dir", "recursive", "verbose", "recreate_venv", "dev", "install_optional"],
        "2_tests.py": ["target_dir", "output_dir", "verbose", "fast_only"],
        # Extend step 2 to accept comprehensive test selection flags
        "2_tests.py": ["target_dir", "output_dir", "verbose", "fast_only", "include_slow", "include_performance", "comprehensive"],
        "3_gnn.py": ["target_dir", "output_dir", "recursive", "verbose", "enable_round_trip", "enable_cross_format"],
        "4_model_registry.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "5_type_checker.py": ["target_dir", "output_dir", "recursive", "verbose", "strict", "estimate_resources"],
        "6_validation.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "7_export.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "8_visualization.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "9_advanced_viz.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "10_ontology.py": ["target_dir", "output_dir", "recursive", "verbose", "ontology_terms_file"],
        "11_render.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "12_execute.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "13_llm.py": ["target_dir", "output_dir", "recursive", "verbose", "llm_tasks", "llm_timeout"],
        "14_ml_integration.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "15_audio.py": ["target_dir", "output_dir", "recursive", "verbose", "duration", "audio_backend"],
        "16_analysis.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "17_integration.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "18_security.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "19_research.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "20_website.py": ["target_dir", "output_dir", "recursive", "verbose", "website_html_filename"],
        "21_mcp.py": ["target_dir", "output_dir", "recursive", "verbose", "performance_mode"],
        "22_gui.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "23_report.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "main.py": list(ARGUMENT_DEFINITIONS.keys())
    }
    
    @classmethod
    def create_main_parser(cls) -> argparse.ArgumentParser:
        """Create the main pipeline argument parser with all arguments."""
        parser = argparse.ArgumentParser(
            description="GNN Processing Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add all arguments
        for arg_name, arg_def in cls.ARGUMENT_DEFINITIONS.items():
            arg_def.add_to_parser(parser)
            
        return parser
    
    @classmethod
    def create_step_parser(cls, step_name: str, description: str = None) -> argparse.ArgumentParser:
        """Create a parser for a specific pipeline step."""
        if description is None:
            description = f"GNN Processing Pipeline - {step_name}"
            
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add arguments supported by this step
        supported_args = cls.STEP_ARGUMENTS.get(step_name, [])
        for arg_name in supported_args:
            if arg_name in cls.ARGUMENT_DEFINITIONS:
                cls.ARGUMENT_DEFINITIONS[arg_name].add_to_parser(parser)
            else:
                logger.warning(f"Unknown argument '{arg_name}' for step {step_name}")
                
        return parser
    
    @classmethod
    def parse_main_arguments(cls, args: Optional[List[str]] = None) -> PipelineArguments:
        """Parse main pipeline arguments and return PipelineArguments object."""
        parser = cls.create_main_parser()
        parsed = parser.parse_args(args)
        
        # Convert to PipelineArguments
        kwargs = {}
        for key, value in vars(parsed).items():
            if value is not None:
                kwargs[key] = value
                
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
    def parse_step_arguments(cls, step_name: str, args: Optional[List[str]] = None) -> argparse.Namespace:
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
                    if arg_name == 'recursive':
                        setattr(parsed_args, arg_name, True)
                    elif arg_name == 'verbose':
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == 'strict':
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == 'estimate_resources':
                        setattr(parsed_args, arg_name, True)
                    elif arg_name.endswith('_dir'):
                        setattr(parsed_args, arg_name, Path("output") if "output" in arg_name else Path("input/gnn_files"))
                    elif arg_name == 'llm_timeout':
                        setattr(parsed_args, arg_name, 360)
                    elif arg_name == 'llm_tasks':
                        setattr(parsed_args, arg_name, "all")
                    elif arg_name == 'website_html_filename':
                        setattr(parsed_args, arg_name, "gnn_pipeline_summary_website.html")
                    elif arg_name in ['recreate_venv', 'dev']:
                        setattr(parsed_args, arg_name, False)
                    elif arg_name == 'duration':
                        setattr(parsed_args, arg_name, 30.0)
                    else:
                        setattr(parsed_args, arg_name, None)
                    
            return parsed_args
            
        except SystemExit as e:
            # Handle argument parsing errors gracefully
            logger.error(f"Argument parsing failed for step {step_name}: {e}")
            # Return a namespace with all required attributes set to defaults
            fallback_args = argparse.Namespace()
            step_supported_args = cls.STEP_ARGUMENTS.get(step_name, [])
            
            for arg_name in step_supported_args:
                if arg_name == 'recursive':
                    setattr(fallback_args, arg_name, True)
                elif arg_name == 'verbose':
                    setattr(fallback_args, arg_name, False)
                elif arg_name == 'strict':
                    setattr(fallback_args, arg_name, False)
                elif arg_name == 'estimate_resources':
                    setattr(fallback_args, arg_name, True)
                elif arg_name.endswith('_dir'):
                    setattr(fallback_args, arg_name, Path("output") if "output" in arg_name else Path("input/gnn_files"))
                elif arg_name == 'llm_timeout':
                    setattr(fallback_args, arg_name, 360)
                elif arg_name == 'llm_tasks':
                    setattr(fallback_args, arg_name, "all")
                elif arg_name == 'website_html_filename':
                    setattr(fallback_args, arg_name, "gnn_pipeline_summary_website.html")
                elif arg_name in ['recreate_venv', 'dev']:
                    setattr(fallback_args, arg_name, False)
                elif arg_name == 'duration':
                    setattr(fallback_args, arg_name, 30.0)
                else:
                    setattr(fallback_args, arg_name, None)
                
            return fallback_args

def build_step_command_args(step_name: str, pipeline_args: PipelineArguments, 
                           python_executable: str, script_path: Path) -> List[str]:
    """
    Build command line arguments for a pipeline step.
    
    Args:
        step_name: Name of the step (e.g., "1_gnn")
        pipeline_args: Main pipeline arguments
        python_executable: Path to Python executable
        script_path: Path to the step script
        
    Returns:
        List of command line arguments
    """
    cmd = [python_executable, str(script_path)]
    
    # Get supported arguments for this step
    # Try both with and without .py extension
    supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_name, [])
    if not supported_args and not step_name.endswith('.py'):
        supported_args = ArgumentParser.STEP_ARGUMENTS.get(f"{step_name}.py", [])
    elif not supported_args and step_name.endswith('.py'):
        supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_name[:-3], [])

    # Special handling for 2_tests.py - only pass test-specific arguments
    if step_name in ['2_tests', '2_tests.py']:
        # Filter to only test-relevant arguments that the test script can handle
        test_args = ['target_dir', 'output_dir', 'verbose', 'fast_only', 'include_slow', 'include_performance', 'comprehensive']
        supported_args = [arg for arg in supported_args if arg in test_args]

    # Build arguments
    for arg_name in supported_args:
        if not hasattr(pipeline_args, arg_name):
            continue
            
        value = getattr(pipeline_args, arg_name)
        if value is None:
            continue
            
        arg_def = ArgumentParser.ARGUMENT_DEFINITIONS.get(arg_name)
        if not arg_def:
            continue
            
        # Handle different argument types
        if arg_def.action == 'store_true':
            if value:
                cmd.append(arg_def.flag)
        elif arg_def.action == argparse.BooleanOptionalAction:
            # Only pass the flag if True; omit if False (don't pass --no-flag)
            # This ensures compatibility with steps that may not support --no-flag
            if value:
                cmd.append(arg_def.flag)
        else:
            cmd.extend([arg_def.flag, str(value)])
    
    return cmd

def get_step_output_dir(step_name: str, base_output_dir: Path) -> Path:
    """
    Get the appropriate output directory for a pipeline step.
    
    Args:
        step_name: Name of the step
        base_output_dir: Base output directory
        
    Returns:
        Output directory for the step
    """
    # Map of steps to their subdirectories  
    STEP_OUTPUT_MAPPING = {
        "1_setup": "setup_artifacts",
        "2_gnn": "gnn_processing_step",
        "3_tests": "test_reports", 
        "4_type_checker": "type_check",
        "5_export": "gnn_exports",
        "6_visualization": "visualization",
        "7_mcp": "mcp_processing_step",
        "8_ontology": "ontology_processing",
        "9_render": "gnn_rendered_simulators",
        "10_execute": "execution_results",
        "11_llm": "llm_processing_step",
        "12_website": "website",
        "13_website": "website",
"14_report": "report_processing_step"
    }
    
    if step_name in STEP_OUTPUT_MAPPING:
        return base_output_dir / STEP_OUTPUT_MAPPING[step_name]
    else:
        return base_output_dir 

# Add enhanced validation and step configuration
class StepConfiguration:
    """Configuration for individual pipeline steps."""
    
    # Define step-specific requirements and defaults
    STEP_CONFIGS = {
        "0_template": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose"],
            "defaults": {"verbose": False},
            "description": "Standardized pipeline step template"
        },
        "1_setup": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose", "recreate_venv", "dev"],
            "defaults": {"verbose": False, "recreate_venv": False, "dev": False},
            "description": "Project Setup & Environment Configuration"
        },
        "2_tests": {
            "required_args": [],
            "optional_args": [
                "target_dir",
                "output_dir",
                "verbose",
                "fast_only",
                "include_slow",
                "include_performance",
                "comprehensive"
            ],
            "defaults": {"verbose": False, "fast_only": False, "include_slow": False, "include_performance": False, "comprehensive": False},
            "description": "Test Execution & Validation"
        },
        "3_gnn": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "enable_round_trip", "enable_cross_format"],
            "defaults": {"recursive": True, "verbose": False, "enable_round_trip": True, "enable_cross_format": True},
            "description": "GNN Discovery & Basic Parse"
        },
        "4_model_registry": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Model Registry & Versioning"
        },
        "5_type_checker": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "strict", "estimate_resources"],
            "defaults": {"recursive": True, "verbose": False, "strict": False, "estimate_resources": True},
            "description": "Type Checking & Resource Estimation"
        },
        "6_validation": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Validation & Quality Assurance"
        },
        "7_export": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "GNN Export & Format Conversion"
        },
        "8_visualization": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Basic Visualization Generation"
        },
        "9_advanced_viz": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Advanced Visualization & Exploration"
        },
        "10_ontology": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "ontology_terms_file"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Ontology Processing & Validation"
        },
        "11_render": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Simulator Code Generation"
        },
        "12_execute": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Simulator Execution"
        },
        "13_llm": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "llm_tasks", "llm_timeout"],
            "defaults": {"recursive": True, "verbose": False, "llm_tasks": "all", "llm_timeout": 360},
            "description": "LLM Analysis & Processing"
        },
        "14_ml_integration": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Machine Learning Integration"
        },
        "15_audio": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "duration", "audio_backend"],
            "defaults": {"recursive": True, "verbose": False, "duration": 30.0, "audio_backend": "auto"},
            "description": "Audio Generation & Processing"
        },
        "16_analysis": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Advanced Analysis & Reporting"
        },
        "17_integration": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "API Gateway & Plugin System"
        },
        "18_security": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Security & Compliance Features"
        },
        "19_research": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Research Workflow Enhancement"
        },
        "20_website": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "website_html_filename"],
            "defaults": {"recursive": True, "verbose": False, "website_html_filename": "gnn_pipeline_summary_website.html"},
            "description": "HTML Website Generation"
        },
        "21_mcp": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "performance_mode"],
            "defaults": {"recursive": True, "verbose": False, "performance_mode": "low"},
            "description": "Model Context Protocol Processing"
        },
        "22_gui": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Interactive GUI for Constructing/Editing GNN Models"
        },
        "23_report": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Comprehensive Analysis Report Generation"
        }
    }
    
    @classmethod
    def get_step_config(cls, step_name: str) -> Dict[str, Any]:
        """Get configuration for a specific step."""
        return cls.STEP_CONFIGS.get(step_name, {})
    
    @classmethod
    def validate_step_args(cls, step_name: str, args: argparse.Namespace) -> List[str]:
        """Validate arguments for a specific step."""
        errors = []
        config = cls.get_step_config(step_name)
        
        if not config:
            errors.append(f"Unknown step: {step_name}")
            return errors
        
        # Check required arguments
        for req_arg in config.get("required_args", []):
            if not hasattr(args, req_arg) or getattr(args, req_arg) is None:
                errors.append(f"Missing required argument for {step_name}: --{req_arg.replace('_', '-')}")
        
        # Validate path arguments exist if they should
        path_args = ["target_dir", "output_dir", "ontology_terms_file"]
        for arg_name in path_args:
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value and isinstance(arg_value, Path):
                    # Only validate existence for input paths, not output paths
                    if arg_name in ["target_dir", "ontology_terms_file"]:
                        # Try to resolve path relative to project root if not found
                        if not arg_value.exists():
                            # Check if we're running from src directory and path is relative to project root
                            import sys
                            if hasattr(sys, '_getframe'):
                                try:
                                    current_file = Path(sys._getframe(1).f_code.co_filename)
                                    if current_file.name.endswith('.py') and current_file.parent.name == 'src':
                                        project_root = current_file.parent.parent
                                        project_path = project_root / arg_value.name
                                        if project_path.exists():
                                            # Update the argument with the correct path
                                            setattr(args, arg_name, project_path)
                                        else:
                                            errors.append(f"Path does not exist for {step_name}: {arg_value}")
                                    else:
                                        errors.append(f"Path does not exist for {step_name}: {arg_value}")
                                except (ValueError, AttributeError):
                                    errors.append(f"Path does not exist for {step_name}: {arg_value}")
                            else:
                                errors.append(f"Path does not exist for {step_name}: {arg_value}")
        
        return errors

class StepAwareArgumentParser:
    """Argument parser with step-specific validation and defaults."""
    
    @classmethod
    def create_step_parser(cls, step_name: str, description: str = None) -> argparse.ArgumentParser:
        """Create a parser for a specific pipeline step."""
        # Remove .py extension for config lookup if present
        config_key = step_name.replace('.py', '') if step_name.endswith('.py') else step_name
        config = StepConfiguration.get_step_config(config_key)
        
        if description is None:
            description = config.get("description", f"GNN Processing Pipeline - {step_name}")
            
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
            """
        )
        
        # Add arguments supported by this step
        # Check both with and without .py extension for STEP_ARGUMENTS lookup
        step_args_key = step_name if step_name in ArgumentParser.STEP_ARGUMENTS else f"{step_name}.py"
        supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_args_key, [])
        for arg_name in supported_args:
            if arg_name in ArgumentParser.ARGUMENT_DEFINITIONS:
                # Get step-specific default if available
                step_defaults = config.get("defaults", {})
                arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg_name]

                # Override default with step-specific value
                if arg_name in step_defaults:
                    modified_arg_def = ArgumentDefinition(
                        flag=arg_def.flag,
                        arg_type=arg_def.arg_type,
                        default=step_defaults[arg_name],
                        required=arg_def.required,
                        help_text=arg_def.help_text,
                        choices=arg_def.choices,
                        action=arg_def.action
                    )
                    modified_arg_def.add_to_parser(parser)
                else:
                    arg_def.add_to_parser(parser)
            else:
                logger.warning(f"Unknown argument '{arg_name}' for step {step_name}")

        return parser
    
    @classmethod
    def parse_step_arguments(cls, step_name: str, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse and validate arguments for a specific pipeline step."""
        parser = cls.create_step_parser(step_name)
        parsed_args = parser.parse_args(args)
        
        # Validate step-specific requirements
        config_key = step_name.replace('.py', '') if step_name.endswith('.py') else step_name
        validation_errors = StepConfiguration.validate_step_args(config_key, parsed_args)
        if validation_errors:
            logger.error(f"Argument validation failed for {step_name}:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Invalid arguments for {step_name}: {'; '.join(validation_errors)}")
        
        return parsed_args
    
    @classmethod
    def get_step_help(cls, step_name: str) -> str:
        """Get help text for a specific step."""
        config_key = step_name.replace('.py', '') if step_name.endswith('.py') else step_name
        config = StepConfiguration.get_step_config(config_key)
        if not config:
            return f"Unknown step: {step_name}"
        
        help_text = [f"Step {step_name}: {config.get('description', 'No description available')}"]
        
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
                    help_text.append(f"  {arg_def.flag}: {arg_def.help_text} (default: {default})")
        
        return "\n".join(help_text)

# Command building with validation
def build_step_command_args(step_name: str, pipeline_args: PipelineArguments,
                                   python_executable: str, script_path: Path) -> List[str]:
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
    config_key = step_name.replace('.py', '') if step_name.endswith('.py') else step_name
    # Also try with .py extension for STEP_ARGUMENTS lookup
    step_key = f"{config_key}.py" if not step_name.endswith('.py') else step_name
    config = StepConfiguration.get_step_config(config_key)
    if not config:
        raise ValueError(f"Unknown pipeline step: {step_name}")
    
    cmd = [python_executable, str(script_path)]
    
    # Get all arguments this step supports
    # First try from StepConfiguration
    all_supported_args = config.get("required_args", []) + config.get("optional_args", [])
    
    # If no arguments found, try from STEP_ARGUMENTS as fallback
    if not all_supported_args and step_key in ArgumentParser.STEP_ARGUMENTS:
        all_supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_key, [])
    
    # Build arguments from pipeline configuration
    for arg_name in all_supported_args:
        if hasattr(pipeline_args, arg_name):
            arg_value = getattr(pipeline_args, arg_name)
            
            # Skip None values for optional arguments
            if arg_value is None and arg_name not in config.get("required_args", []):
                continue
            
            # Get argument definition for proper formatting
            if arg_name in ArgumentParser.ARGUMENT_DEFINITIONS:
                arg_def = ArgumentParser.ARGUMENT_DEFINITIONS[arg_name]
                flag = arg_def.flag
                
                # Handle different argument types
                if arg_def.action == 'store_true':
                    if arg_value:
                        cmd.append(flag)
                elif arg_def.action == argparse.BooleanOptionalAction:
                    # Only pass the flag if True; omit if False (don't pass --no-flag)
                    # This ensures compatibility with steps that may not support --no-flag
                    if arg_value is True:
                        cmd.append(flag)
                else:
                    # Regular arguments with values
                    cmd.extend([flag, str(arg_value)])
    
    return cmd

# Utility for step introspection
def get_pipeline_step_info() -> Dict[str, Any]:
    """Get comprehensive information about all pipeline steps."""
    step_info = {}
    
    for step_name, config in StepConfiguration.STEP_CONFIGS.items():
        step_info[step_name] = {
            "description": config.get("description", ""),
            "required_args": config.get("required_args", []),
            "optional_args": config.get("optional_args", []),
            "defaults": config.get("defaults", {}),
            "total_args": len(config.get("required_args", [])) + len(config.get("optional_args", []))
        }
    
    return step_info

# Validation utility for the entire pipeline
def parse_step_arguments(step_name: str, args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments for a specific pipeline step (standalone function)."""
    return ArgumentParser.parse_step_arguments(step_name, args)

def validate_arguments(args: argparse.Namespace) -> List[str]:
    """Validate parsed arguments and return list of errors."""
    errors = []
    
    # Basic validation
    if hasattr(args, 'target_dir') and args.target_dir:
        if not Path(args.target_dir).exists():
            errors.append(f"Target directory does not exist: {args.target_dir}")
    
    if hasattr(args, 'output_dir') and args.output_dir:
        # Output directory can be created if it doesn't exist
        pass
    
    return errors

def convert_path_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Convert string paths to Path objects in parsed arguments."""
    for attr_name in dir(args):
        if not attr_name.startswith('_'):
            attr_value = getattr(args, attr_name)
            if isinstance(attr_value, str) and ('dir' in attr_name or 'path' in attr_name or 'file' in attr_name):
                setattr(args, attr_name, Path(attr_value))
    return args

def validate_pipeline_configuration(pipeline_args: PipelineArguments) -> Dict[str, List[str]]:
    """
    Validate pipeline configuration against all step requirements.
    
    Returns:
        Dictionary mapping step names to lists of validation errors
    """
    validation_results = {}
    
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

def parse_step_list(step_str: str) -> List[int]:
    """Parse a comma-separated list of step numbers."""
    if not step_str:
        return []
    
    steps = []
    for item in step_str.split(','):
        item = item.strip()
        # Extract number from formats like "1", "1_gnn", etc.
        match = re.match(r'^(\d+)', item)
        if match:
            steps.append(int(match.group(1)))
    return steps

def parse_arguments() -> PipelineArguments:
    """Parse command line arguments and load configuration."""
    # Create argument parser for command line options
    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline with YAML configuration support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add configuration file option
    parser.add_argument(
        '--config-file',
        type=Path,
        default=Path("input/config.yaml"),
        help='Path to configuration file (default: input/config.yaml)'
    )
    
    # Add all other options that can override config
    parser.add_argument('--target-dir', type=Path, help='Target directory for GNN files (overrides config)')
    parser.add_argument('--output-dir', type=Path, help='Directory to save outputs (overrides config)')
    parser.add_argument('--recursive', action=argparse.BooleanOptionalAction, help='Recursively process directories (overrides config)')
    parser.add_argument('--skip-steps', help='Comma-separated list of steps to skip (overrides config)')
    parser.add_argument('--only-steps', help='Comma-separated list of steps to run (overrides config)')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Enable verbose output (overrides config)')
    parser.add_argument('--enable-round-trip', action='store_true', help='Enable comprehensive round-trip testing across all 21 formats (overrides config)')
    parser.add_argument('--enable-cross-format', action='store_true', help='Enable cross-format consistency validation (overrides config)')
    parser.add_argument('--strict', action='store_true', help='Enable strict type checking mode')
    parser.add_argument('--estimate-resources', action=argparse.BooleanOptionalAction, help='Estimate computational resources')
    parser.add_argument('--ontology-terms-file', type=Path, help='Path to ontology terms file (overrides config)')
    parser.add_argument('--llm-tasks', help='Comma-separated list of LLM tasks')
    parser.add_argument('--llm-timeout', type=int, help='Timeout for LLM processing in seconds')
    parser.add_argument('--pipeline-summary-file', type=Path, help='Path to save pipeline summary')
    parser.add_argument('--website-html-filename', help='Filename for generated HTML website')
    parser.add_argument('--duration', type=float, help='Audio duration in seconds for audio generation')
    parser.add_argument('--audio-backend', type=str, default='auto', 
                       help='Audio backend to use (auto, sapf, pedalboard, default: auto)')
    parser.add_argument('--recreate-venv', action='store_true', help='Recreate virtual environment')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Determine project root - this should be the parent of the src directory
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    project_root = current_dir.parent.parent  # Go up from utils/ to src/ to project root
    
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
    if pipeline_args.ontology_terms_file and not pipeline_args.ontology_terms_file.is_absolute():
        pipeline_args.ontology_terms_file = project_root / pipeline_args.ontology_terms_file
    if pipeline_args.pipeline_summary_file and not pipeline_args.pipeline_summary_file.is_absolute():
        pipeline_args.pipeline_summary_file = project_root / pipeline_args.pipeline_summary_file
    
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
    
    # Resolve relative paths relative to input directory
    input_dir = Path("input")
    # If target_dir is relative, make it relative to input directory, but avoid double prefixing
    if not pipeline_args.target_dir.is_absolute():
        target_str = str(pipeline_args.target_dir)
        if not target_str.startswith("input/"):
            pipeline_args.target_dir = input_dir / pipeline_args.target_dir
    
    return pipeline_args 

def validate_and_convert_paths(args: PipelineArguments, logger: logging.Logger):
    path_args_to_check = [
        'output_dir', 'target_dir', 'ontology_terms_file', 'pipeline_summary_file'
    ]

    for arg_name in path_args_to_check:
        if not hasattr(args, arg_name):
            logger.debug(f"Argument --{arg_name.replace('_', '-')} not present in args namespace.")
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
                if arg_name in ['output_dir', 'target_dir']:
                    logger.critical(f"Critical path argument --{arg_name.replace('_', '-')} could not be converted to Path. Exiting.")
                    sys.exit(1)
        elif arg_value is None and arg_name in ['output_dir', 'target_dir']:
             logger.critical(
                f"Critical path argument --{arg_name.replace('_', '-')} is None after parsing. "
                "This indicates a problem with default value setup in argparse. Exiting."
             )
             sys.exit(1) 