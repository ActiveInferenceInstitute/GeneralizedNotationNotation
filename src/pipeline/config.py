#!/usr/bin/env python3
"""
Centralized Pipeline Configuration

This module provides unified configuration management for the entire GNN pipeline.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    name: str
    description: str
    module_path: str
    dependencies: List[str] = field(default_factory=list)
    output_subdir: str = ""
    timeout: Optional[int] = None
    required: bool = True
    performance_tracking: bool = False

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    project_name: str = "GeneralizedNotationNotation"
    version: str = "1.0.0"
    base_output_dir: Path = Path("output")
    base_target_dir: Path = Path("input/gnn_files")
    log_level: str = "INFO"
    correlation_id_length: int = 8
    
    # Environment variable overrides
    env_prefix: str = "GNN_PIPELINE_"
    
    # Step configurations
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize step configurations and apply environment overrides."""
        self._initialize_step_configs()
        self._apply_environment_overrides()
    
    def _initialize_step_configs(self):
        """Initialize default step configurations."""
        self.steps = {
            "0_template.py": StepConfig(
                name="template",
                description="Standardized pipeline step template",
                module_path="0_template.py",
                output_subdir="template",
                required=False  # Template is not required for pipeline execution
            ),
            "1_setup.py": StepConfig(
                name="setup",
                description="Project setup and environment validation",
                module_path="1_setup.py",
                output_subdir="setup_artifacts",
                timeout=600,
                required=True  # Setup is critical for the pipeline
            ),
            "2_tests.py": StepConfig(
                name="tests",
                description="Test suite execution with coverage reporting",
                module_path="2_tests.py",
                output_subdir="test_reports",
                dependencies=["1_setup.py"],
                timeout=600,
                required=False,  # Tests can fail without stopping pipeline
                performance_tracking=True
            ),
            "3_gnn.py": StepConfig(
                name="gnn_processing",
                description="GNN file discovery and parsing",
                module_path="3_gnn.py",
                output_subdir="gnn_processing_step",
                performance_tracking=True,
                required=True  # GNN processing is critical for the pipeline
            ),
            "4_model_registry.py": StepConfig(
                name="model_registry",
                description="Model versioning and management",
                module_path="4_model_registry.py",
                output_subdir="model_registry",
                dependencies=["3_gnn.py"],
                performance_tracking=True,
                required=False  # Model registry can fail without stopping pipeline
            ),
            "5_type_checker.py": StepConfig(
                name="type_checking",
                description="GNN syntax and type validation",
                module_path="5_type_checker.py",
                output_subdir="type_check",
                dependencies=["3_gnn.py"],
                performance_tracking=True,
                required=False  # Type checking can fail without stopping pipeline
            ),
            "6_validation.py": StepConfig(
                name="validation",
                description="Enhanced validation and quality assurance",
                module_path="6_validation.py",
                output_subdir="validation",
                dependencies=["5_type_checker.py"],
                performance_tracking=True,
                required=False  # Validation can fail without stopping pipeline
            ),
            "7_export.py": StepConfig(
                name="export",
                description="Multi-format export generation",
                module_path="7_export.py",
                output_subdir="gnn_exports",
                dependencies=["5_type_checker.py"],
                performance_tracking=True,
                required=False  # Export can fail without stopping pipeline
            ),
            "8_visualization.py": StepConfig(
                name="visualization",
                description="Basic graph and statistical visualizations",
                module_path="8_visualization.py",
                output_subdir="visualization",
                dependencies=["7_export.py"],
                performance_tracking=True,
                required=False  # Visualization can fail without stopping pipeline
            ),
            "9_advanced_viz.py": StepConfig(
                name="advanced_visualization",
                description="Advanced visualization and exploration",
                module_path="9_advanced_viz.py",
                output_subdir="advanced_visualization",
                dependencies=["8_visualization.py"],
                performance_tracking=True,
                required=False  # Advanced visualization can fail without stopping pipeline
            ),
            "10_ontology.py": StepConfig(
                name="ontology",
                description="Ontology processing and validation",
                module_path="10_ontology.py",
                output_subdir="ontology_processing",
                dependencies=["7_export.py"],
                required=False  # Ontology can fail without stopping pipeline
            ),
            "11_render.py": StepConfig(
                name="render",
                description="Code generation for simulation environments",
                module_path="11_render.py",
                output_subdir="gnn_rendered_simulators",
                dependencies=["7_export.py"],
                performance_tracking=True,
                required=False  # Rendering can fail without stopping pipeline
            ),
            "12_execute.py": StepConfig(
                name="execute",
                description="Execute rendered simulators",
                module_path="12_execute.py",
                output_subdir="execution_results",
                dependencies=["11_render.py"],
                performance_tracking=True,
                required=False  # Execution can fail without stopping pipeline
            ),
            "13_llm.py": StepConfig(
                name="llm",
                description="LLM-enhanced analysis and processing",
                module_path="13_llm.py",
                output_subdir="llm_processing_step",
                dependencies=["7_export.py"],
                performance_tracking=True,
                required=False  # LLM can fail without stopping pipeline
            ),
            "14_ml_integration.py": StepConfig(
                name="ml_integration",
                description="Machine learning integration",
                module_path="14_ml_integration.py",
                output_subdir="ml_integration",
                dependencies=["13_llm.py"],
                performance_tracking=True,
                required=False  # ML integration can fail without stopping pipeline
            ),
            "15_audio.py": StepConfig(
                name="audio",
                description="Audio generation for GNN models (SAPF, Pedalboard, and other backends)",
                module_path="15_audio.py",
                output_subdir="audio_processing_step",
                dependencies=["3_gnn.py"],
                performance_tracking=True,
                required=False  # Audio generation can fail without stopping pipeline
            ),
            "16_analysis.py": StepConfig(
                name="analysis",
                description="Advanced statistical analysis and reporting",
                module_path="16_analysis.py",
                output_subdir="analysis",
                dependencies=["12_execute.py", "14_ml_integration.py"],
                performance_tracking=True,
                required=False  # Analysis can fail without stopping pipeline
            ),
            "17_integration.py": StepConfig(
                name="integration",
                description="API gateway and plugin system",
                module_path="17_integration.py",
                output_subdir="integration",
                dependencies=["16_analysis.py"],
                performance_tracking=True,
                required=False  # Integration can fail without stopping pipeline
            ),
            "18_security.py": StepConfig(
                name="security",
                description="Security and compliance features",
                module_path="18_security.py",
                output_subdir="security",
                dependencies=["17_integration.py"],
                performance_tracking=True,
                required=False  # Security can fail without stopping pipeline
            ),
            "19_research.py": StepConfig(
                name="research",
                description="Research workflow enhancement",
                module_path="19_research.py",
                output_subdir="research",
                dependencies=["16_analysis.py"],
                performance_tracking=True,
                required=False  # Research workflow can fail without stopping pipeline
            ),
            "20_website.py": StepConfig(
                name="website",
                description="Static website generation",
                module_path="20_website.py",
                output_subdir="website",
                dependencies=["8_visualization.py", "10_ontology.py", "16_analysis.py"],
                required=False  # Website generation can fail without stopping pipeline
            ),
            "21_report.py": StepConfig(
                name="report",
                description="Comprehensive analysis report generation",
                module_path="21_report.py",
                output_subdir="report_processing_step",
                dependencies=["16_analysis.py", "20_website.py"],
                performance_tracking=True,
                required=False  # Report generation can fail without stopping pipeline
            ),
            "main.py": StepConfig(
                name="main",
                description="Main pipeline orchestrator",
                module_path="main.py",
                output_subdir="pipeline_logs"
            )
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Override base directories
        if env_output_dir := os.getenv(f"{self.env_prefix}OUTPUT_DIR"):
            self.base_output_dir = Path(env_output_dir)
        
        if env_target_dir := os.getenv(f"{self.env_prefix}TARGET_DIR"):
            self.base_target_dir = Path(env_target_dir)
        
        # Override log level
        if env_log_level := os.getenv(f"{self.env_prefix}LOG_LEVEL"):
            self.log_level = env_log_level.upper()
        
        # Override verbosity
        if os.getenv(f"{self.env_prefix}VERBOSE", "").lower() in ("true", "1", "yes"):
            self.log_level = "DEBUG"
    
    def get_step_config(self, step_name: str) -> Optional[StepConfig]:
        """Get configuration for a specific step."""
        return self.steps.get(step_name)
    
    def get_output_dir_for_step(self, step_name: str, base_output_dir: Optional[Path] = None) -> Path:
        """Get the output directory for a specific step."""
        output_dir = base_output_dir or self.base_output_dir
        step_config = self.get_step_config(step_name)
        
        if step_config and step_config.output_subdir:
            return output_dir / step_config.output_subdir
        else:
            # Fallback to step name without extension
            step_base = step_name.replace(".py", "").replace("_", "-")
            return output_dir / step_base
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get dependencies for a specific step."""
        step_config = self.get_step_config(step_name)
        return step_config.dependencies if step_config else []
    
    def is_performance_tracking_enabled(self, step_name: str) -> bool:
        """Check if performance tracking is enabled for a step."""
        step_config = self.get_step_config(step_name)
        return step_config.performance_tracking if step_config else False
    
    def get_step_timeout(self, step_name: str) -> Optional[int]:
        """Get timeout for a specific step."""
        step_config = self.get_step_config(step_name)
        return step_config.timeout if step_config else None
    
    def is_step_required(self, step_name: str) -> bool:
        """Check if a step is required for pipeline completion."""
        step_config = self.get_step_config(step_name)
        return step_config.required if step_config else True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "base_output_dir": str(self.base_output_dir),
            "base_target_dir": str(self.base_target_dir),
            "log_level": self.log_level,
            "correlation_id_length": self.correlation_id_length,
            "env_prefix": self.env_prefix,
            "steps": {
                name: {
                    "name": step.name,
                    "description": step.description,
                    "module_path": step.module_path,
                    "dependencies": step.dependencies,
                    "output_subdir": step.output_subdir,
                    "timeout": step.timeout,
                    "required": step.required,
                    "performance_tracking": step.performance_tracking
                }
                for name, step in self.steps.items()
            }
        }
    
    def save_to_file(self, file_path: Path):
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'PipelineConfig':
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.project_name = data.get("project_name", config.project_name)
        config.version = data.get("version", config.version)
        config.base_output_dir = Path(data.get("base_output_dir", config.base_output_dir))
        config.base_target_dir = Path(data.get("base_target_dir", config.base_target_dir))
        config.log_level = data.get("log_level", config.log_level)
        config.correlation_id_length = data.get("correlation_id_length", config.correlation_id_length)
        config.env_prefix = data.get("env_prefix", config.env_prefix)
        
        # Load step configurations
        if "steps" in data:
            for step_name, step_data in data["steps"].items():
                config.steps[step_name] = StepConfig(
                    name=step_data["name"],
                    description=step_data["description"],
                    module_path=step_data["module_path"],
                    dependencies=step_data.get("dependencies", []),
                    output_subdir=step_data.get("output_subdir", ""),
                    timeout=step_data.get("timeout"),
                    required=step_data.get("required", True),
                    performance_tracking=step_data.get("performance_tracking", False)
                )
        
        return config

# Global pipeline configuration instance
_pipeline_config = None

def get_pipeline_config() -> PipelineConfig:
    """Get the global pipeline configuration instance."""
    global _pipeline_config
    if _pipeline_config is None:
        _pipeline_config = PipelineConfig()
    return _pipeline_config

def set_pipeline_config(config: PipelineConfig):
    """Set the global pipeline configuration instance."""
    global _pipeline_config
    _pipeline_config = config

def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get the output directory for a specific script."""
    config = get_pipeline_config()
    return config.get_output_dir_for_step(script_name, base_output_dir)

def validate_pipeline_config(config: PipelineConfig) -> bool:
    """
    Validate a pipeline configuration for completeness and correctness.
    Args:
        config: PipelineConfig instance to validate
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check that all required steps are present
        required_steps = ["1_setup.py", "2_gnn.py", "main.py"]
        for step in required_steps:
            if step not in config.steps:
                return False
        
        # Check that output directories are valid
        if not config.base_output_dir:
            return False
        
        # Check that step configurations are valid
        for step_name, step_config in config.steps.items():
            if not step_config.name or not step_config.description:
                return False
        
        return True
    except Exception:
        return False

def update_pipeline_config(**kwargs) -> PipelineConfig:
    """Update the global pipeline configuration with new values."""
    global _pipeline_config
    config = get_pipeline_config()
    
    # Update values from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    _pipeline_config = config
    return config

def get_step_metadata_dict() -> Dict[str, Dict[str, Any]]:
    """
    Get step metadata as dictionary for backward compatibility.
    
    Returns:
        Dictionary with step metadata in the legacy STEP_METADATA format
    """
    config = get_pipeline_config()
    metadata_dict = {}
    
    for step_name, step_config in config.steps.items():
        metadata_dict[step_name] = {
            "description": step_config.description,
            "required": step_config.required,
            "dependencies": step_config.dependencies,
            "output_subdir": step_config.output_subdir
        }
    
    return metadata_dict

# Backward compatibility - provide STEP_METADATA as a property that returns current config
def _get_step_metadata():
    """Lazy loader for backward compatibility."""
    return get_step_metadata_dict()

# Create a property-like access for STEP_METADATA for backward compatibility
class StepMetadataProxy(dict):
    """Proxy class to provide STEP_METADATA as a dict while using dataclass config."""
    
    def __init__(self):
        super().__init__()
        self._update_dict()
    
    def _update_dict(self):
        """Update the internal dict with current metadata."""
        self.clear()
        self.update(get_step_metadata_dict())
    
    def __getitem__(self, key):
        self._update_dict()  # Always get fresh data
        return super().__getitem__(key)
    
    def __contains__(self, key):
        self._update_dict()  # Always get fresh data
        return super().__contains__(key)
    
    def __iter__(self):
        self._update_dict()  # Always get fresh data
        return super().__iter__()
    
    def keys(self):
        self._update_dict()  # Always get fresh data
        return super().keys()
    
    def values(self):
        self._update_dict()  # Always get fresh data
        return super().values()
    
    def items(self):
        self._update_dict()  # Always get fresh data
        return super().items()
    
    def get(self, key, default=None):
        self._update_dict()  # Always get fresh data
        return super().get(key, default)
    
    def __len__(self):
        self._update_dict()  # Always get fresh data
        return super().__len__()
    
    def __bool__(self):
        self._update_dict()  # Always get fresh data
        return super().__bool__()

# Provide STEP_METADATA for backward compatibility
STEP_METADATA = StepMetadataProxy() 