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
    target_dir: Path = field(default_factory=lambda: Path("gnn/examples"))
    output_dir: Path = field(default_factory=lambda: Path("../output"))
    
    # Processing options
    recursive: bool = True
    verbose: bool = True
    
    # Step control  
    skip_steps: Optional[str] = None
    only_steps: Optional[str] = None
    
    # Type checking options
    strict: bool = False
    estimate_resources: bool = True
    
    # File references
    ontology_terms_file: Optional[Path] = None
    pipeline_summary_file: Optional[Path] = None
    
    # LLM options
    llm_tasks: str = "all"
    llm_timeout: int = 360
    
    # Site generation
    site_html_filename: str = "gnn_pipeline_summary_site.html"
    
    # DisCoPy options
    discopy_gnn_input_dir: Optional[Path] = None
    discopy_jax_gnn_input_dir: Optional[Path] = None
    discopy_jax_seed: int = 0
    
    # Setup options
    recreate_venv: bool = False
    dev: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and path resolution."""
        # Ensure Path objects
        if isinstance(self.target_dir, str):
            self.target_dir = Path(self.target_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        # Set defaults for optional paths
        if self.ontology_terms_file is None:
            self.ontology_terms_file = Path("ontology/act_inf_ontology_terms.json")
        elif isinstance(self.ontology_terms_file, str):
            self.ontology_terms_file = Path(self.ontology_terms_file)
            
        if self.pipeline_summary_file is None:
            self.pipeline_summary_file = self.output_dir / "pipeline_execution_summary.json"
        elif isinstance(self.pipeline_summary_file, str):
            self.pipeline_summary_file = Path(self.pipeline_summary_file)
            
        # Set DisCoPy input dirs to target_dir if not specified
        if self.discopy_gnn_input_dir is None:
            self.discopy_gnn_input_dir = self.target_dir
        elif isinstance(self.discopy_gnn_input_dir, str):
            self.discopy_gnn_input_dir = Path(self.discopy_gnn_input_dir)
            
        if self.discopy_jax_gnn_input_dir is None:
            self.discopy_jax_gnn_input_dir = self.target_dir
        elif isinstance(self.discopy_jax_gnn_input_dir, str):
            self.discopy_jax_gnn_input_dir = Path(self.discopy_jax_gnn_input_dir)
    
    def validate(self) -> List[str]:
        """Validate argument values and return list of errors."""
        errors = []
        
        # Check that target directory exists if not a special placeholder
        if not str(self.target_dir).startswith("<") and not self.target_dir.exists():
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
            help_text='Target directory for GNN files'
        ),
        'output_dir': ArgumentDefinition(
            flag='--output-dir', 
            arg_type=Path,
            help_text='Directory to save outputs'
        ),
        'recursive': ArgumentDefinition(
            flag='--recursive',
            action=argparse.BooleanOptionalAction,
            help_text='Recursively process directories'
        ),
        'verbose': ArgumentDefinition(
            flag='--verbose',
            action=argparse.BooleanOptionalAction, 
            help_text='Enable verbose output'
        ),
        'skip_steps': ArgumentDefinition(
            flag='--skip-steps',
            help_text='Comma-separated list of steps to skip'
        ),
        'only_steps': ArgumentDefinition(
            flag='--only-steps',
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
        'site_html_filename': ArgumentDefinition(
            flag='--site-html-filename',
            help_text='Filename for generated HTML site'
        ),
        'discopy_gnn_input_dir': ArgumentDefinition(
            flag='--discopy-gnn-input-dir',
            arg_type=Path,
            help_text='Input directory for DisCoPy processing'
        ),
        'discopy_jax_gnn_input_dir': ArgumentDefinition(
            flag='--discopy-jax-gnn-input-dir',
            arg_type=Path,
            help_text='Input directory for DisCoPy JAX evaluation'
        ),
        'discopy_jax_seed': ArgumentDefinition(
            flag='--discopy-jax-seed',
            arg_type=int,
            help_text='Random seed for DisCoPy JAX evaluation'
        ),
        'recreate_venv': ArgumentDefinition(
            flag='--recreate-venv',
            action='store_true',
            help_text='Recreate virtual environment'
        ),
        'dev': ArgumentDefinition(
            flag='--dev',
            action='store_true', 
            help_text='Install development dependencies'
        )
    }
    
    # Define which arguments each step supports
    STEP_ARGUMENTS = {
        "1_gnn": ["target_dir", "output_dir", "recursive", "verbose"],
        "2_setup": ["target_dir", "output_dir", "verbose", "recreate_venv", "dev"],
        "3_tests": ["target_dir", "output_dir", "verbose"],
        "4_gnn_type_checker": ["target_dir", "output_dir", "recursive", "verbose", "strict", "estimate_resources"],
        "5_export": ["target_dir", "output_dir", "recursive", "verbose"],
        "6_visualization": ["target_dir", "output_dir", "recursive", "verbose"],
        "7_mcp": ["target_dir", "output_dir", "verbose"],
        "8_ontology": ["target_dir", "output_dir", "recursive", "verbose", "ontology_terms_file"],
        "9_render": ["output_dir", "recursive", "verbose"],
        "10_execute": ["target_dir", "output_dir", "recursive", "verbose"],
        "11_llm": ["target_dir", "output_dir", "recursive", "verbose", "llm_tasks", "llm_timeout"],
        "12_discopy": ["target_dir", "output_dir", "verbose", "discopy_gnn_input_dir"],
        "13_discopy_jax_eval": ["target_dir", "output_dir", "verbose", "discopy_jax_gnn_input_dir", "discopy_jax_seed"],
        "14_site": ["target_dir", "output_dir", "verbose", "site_html_filename"],
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
        """Parse arguments for a specific pipeline step."""
        parser = cls.create_step_parser(step_name)
        return parser.parse_args(args)

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
    supported_args = ArgumentParser.STEP_ARGUMENTS.get(step_name, [])
    
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
            if value:
                cmd.append(arg_def.flag)
            else:
                cmd.append(f"--no-{arg_def.flag[2:]}")
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
        "1_gnn": "gnn_processing_step",
        "3_tests": "test_reports", 
        "4_gnn_type_checker": "gnn_type_check",
        "5_export": "gnn_exports",
        "6_visualization": "visualization",
        "7_mcp": "mcp_processing_step",
        "8_ontology": "ontology_processing",
        "9_render": "gnn_rendered_simulators",
        "10_execute": "execute_logs",
        "11_llm": "llm_processing_step",
        "12_discopy": "discopy_gnn",
        "13_discopy_jax_eval": "discopy_jax_eval"
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
        "1_gnn": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "GNN Discovery & Basic Parse"
        },
        "2_setup": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose", "recreate_venv", "dev"],
            "defaults": {"verbose": False, "recreate_venv": False, "dev": False},
            "description": "Project Setup & Environment Configuration"
        },
        "3_tests": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose"],
            "defaults": {"verbose": False},
            "description": "Test Execution & Validation"
        },
        "4_gnn_type_checker": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "strict", "estimate_resources"],
            "defaults": {"recursive": True, "verbose": False, "strict": False, "estimate_resources": True},
            "description": "GNN Type Checking & Resource Estimation"
        },
        "5_export": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": False, "verbose": False},
            "description": "GNN Export & Format Conversion"
        },
        "6_visualization": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": False, "verbose": False},
            "description": "Visualization Generation"
        },
        "7_mcp": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose"],
            "defaults": {"verbose": False},
            "description": "MCP Integration & Tool Discovery"
        },
        "8_ontology": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "ontology_terms_file"],
            "defaults": {"recursive": False, "verbose": False},
            "description": "Ontology Processing & Validation"
        },
        "9_render": {
            "required_args": ["output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": False, "verbose": False},
            "description": "Simulator Code Generation"
        },
        "10_execute": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose"],
            "defaults": {"recursive": True, "verbose": False},
            "description": "Simulator Execution"
        },
        "11_llm": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["recursive", "verbose", "llm_tasks", "llm_timeout"],
            "defaults": {"recursive": False, "verbose": False, "llm_tasks": "all", "llm_timeout": 360},
            "description": "LLM Analysis & Processing"
        },
        "12_discopy": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose", "discopy_gnn_input_dir"],
            "defaults": {"verbose": False},
            "description": "DisCoPy Diagram Generation"
        },
        "13_discopy_jax_eval": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose", "discopy_jax_gnn_input_dir", "discopy_jax_seed"],
            "defaults": {"verbose": False, "discopy_jax_seed": 0},
            "description": "DisCoPy JAX Evaluation"
        },
        "14_site": {
            "required_args": ["target_dir", "output_dir"],
            "optional_args": ["verbose", "site_html_filename"],
            "defaults": {"verbose": False, "site_html_filename": "gnn_pipeline_summary_site.html"},
            "description": "HTML Site Generation"
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
                    if arg_name in ["target_dir", "ontology_terms_file"] and not arg_value.exists():
                        errors.append(f"Path does not exist for {step_name}: {arg_value}")
        
        return errors

# Enhanced argument parser with step awareness
class EnhancedArgumentParser(ArgumentParser):
    """Enhanced argument parser with step-specific validation and defaults."""
    
    @classmethod
    def create_step_parser(cls, step_name: str, description: str = None) -> argparse.ArgumentParser:
        """Create an enhanced parser for a specific pipeline step."""
        config = StepConfiguration.get_step_config(step_name)
        
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
        supported_args = cls.STEP_ARGUMENTS.get(step_name, [])
        for arg_name in supported_args:
            if arg_name in cls.ARGUMENT_DEFINITIONS:
                # Get step-specific default if available
                step_defaults = config.get("defaults", {})
                arg_def = cls.ARGUMENT_DEFINITIONS[arg_name]
                
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
        validation_errors = StepConfiguration.validate_step_args(step_name, parsed_args)
        if validation_errors:
            logger.error(f"Argument validation failed for {step_name}:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise ValueError(f"Invalid arguments for {step_name}: {'; '.join(validation_errors)}")
        
        return parsed_args
    
    @classmethod
    def get_step_help(cls, step_name: str) -> str:
        """Get help text for a specific step."""
        config = StepConfiguration.get_step_config(step_name)
        if not config:
            return f"Unknown step: {step_name}"
        
        help_text = [f"Step {step_name}: {config.get('description', 'No description available')}"]
        
        # Required arguments
        req_args = config.get("required_args", [])
        if req_args:
            help_text.append("\nRequired arguments:")
            for arg in req_args:
                if arg in cls.ARGUMENT_DEFINITIONS:
                    arg_def = cls.ARGUMENT_DEFINITIONS[arg]
                    help_text.append(f"  {arg_def.flag}: {arg_def.help_text}")
        
        # Optional arguments
        opt_args = config.get("optional_args", [])
        if opt_args:
            help_text.append("\nOptional arguments:")
            for arg in opt_args:
                if arg in cls.ARGUMENT_DEFINITIONS:
                    arg_def = cls.ARGUMENT_DEFINITIONS[arg]
                    default = config.get("defaults", {}).get(arg, arg_def.default)
                    help_text.append(f"  {arg_def.flag}: {arg_def.help_text} (default: {default})")
        
        return "\n".join(help_text)

# Enhanced command building with validation
def build_enhanced_step_command_args(step_name: str, pipeline_args: PipelineArguments, 
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
    config = StepConfiguration.get_step_config(step_name)
    if not config:
        raise ValueError(f"Unknown pipeline step: {step_name}")
    
    cmd = [python_executable, str(script_path)]
    
    # Get all arguments this step supports
    all_supported_args = config.get("required_args", []) + config.get("optional_args", [])
    
    # Build arguments from pipeline configuration
    for arg_name in all_supported_args:
        if hasattr(pipeline_args, arg_name):
            arg_value = getattr(pipeline_args, arg_name)
            
            # Skip None values for optional arguments
            if arg_value is None and arg_name not in config.get("required_args", []):
                continue
            
            # Get argument definition for proper formatting
            if arg_name in EnhancedArgumentParser.ARGUMENT_DEFINITIONS:
                arg_def = EnhancedArgumentParser.ARGUMENT_DEFINITIONS[arg_name]
                flag = arg_def.flag
                
                # Handle different argument types
                if arg_def.action == 'store_true':
                    if arg_value:
                        cmd.append(flag)
                elif arg_def.action == argparse.BooleanOptionalAction:
                    if arg_value is True:
                        cmd.append(flag)
                    elif arg_value is False:
                        cmd.append(f"--no-{flag[2:]}")  # Convert --flag to --no-flag
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