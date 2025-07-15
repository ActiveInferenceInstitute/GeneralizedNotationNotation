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
            "1_gnn.py": StepConfig(
                name="gnn_processing",
                description="GNN file discovery and parsing",
                module_path="1_gnn.py",
                output_subdir="gnn_processing_step",
                performance_tracking=True,
                required=True  # GNN processing is critical for the pipeline
            ),
            "2_setup.py": StepConfig(
                name="setup",
                description="Project setup and environment validation",
                module_path="2_setup.py",
                output_subdir="setup_artifacts",
                timeout=300,
                required=True  # Setup is critical for the pipeline
            ),
            "3_tests.py": StepConfig(
                name="tests",
                description="Test suite execution with coverage reporting",
                module_path="3_tests.py",
                output_subdir="test_reports",
                dependencies=["2_setup.py"],
                timeout=600,
                required=False,  # Tests can fail without stopping pipeline
                performance_tracking=True
            ),
            "4_type_checker.py": StepConfig(
                name="type_checking",
                description="GNN syntax and type validation",
                module_path="4_type_checker.py",
                output_subdir="type_check",
                dependencies=["1_gnn.py"],
                performance_tracking=True,
                required=False  # Type checking can fail without stopping pipeline
            ),
            "5_export.py": StepConfig(
                name="export",
                description="Multi-format export generation",
                module_path="5_export.py",
                output_subdir="gnn_exports",
                dependencies=["4_type_checker.py"],
                performance_tracking=True,
                required=False  # Export can fail without stopping pipeline
            ),
            "6_visualization.py": StepConfig(
                name="visualization",
                description="Graph visualization generation",
                module_path="6_visualization.py",
                output_subdir="visualization",
                dependencies=["5_export.py"],
                performance_tracking=True,
                required=False  # Visualization can fail without stopping pipeline
            ),
            "7_mcp.py": StepConfig(
                name="mcp",
                description="Model Context Protocol operations",
                module_path="7_mcp.py",
                output_subdir="mcp_processing_step",
                required=False  # MCP can fail without stopping pipeline
            ),
            "8_ontology.py": StepConfig(
                name="ontology",
                description="Ontology processing and validation",
                module_path="8_ontology.py",
                output_subdir="ontology_processing",
                dependencies=["5_export.py"],
                required=False  # Ontology can fail without stopping pipeline
            ),
            "9_render.py": StepConfig(
                name="render",
                description="Code generation for simulation environments",
                module_path="9_render.py",
                output_subdir="gnn_rendered_simulators",
                dependencies=["5_export.py"],
                performance_tracking=True,
                required=False  # Rendering can fail without stopping pipeline
            ),
            "10_execute.py": StepConfig(
                name="execute",
                description="Execute rendered simulators",
                module_path="10_execute.py",
                output_subdir="execution_results",
                dependencies=["9_render.py"],
                performance_tracking=True,
                required=False  # Execution can fail without stopping pipeline
            ),
            "11_llm.py": StepConfig(
                name="llm",
                description="LLM-enhanced analysis and processing",
                module_path="11_llm.py",
                output_subdir="llm_processing_step",
                dependencies=["5_export.py"],
                performance_tracking=True,
                required=False  # LLM can fail without stopping pipeline
            ),
            "12_website.py": StepConfig(
                name="website",
                description="Static website generation",
                module_path="12_website.py",
                output_subdir="website",
                dependencies=["6_visualization.py", "8_ontology.py"],
                required=False  # Website generation can fail without stopping pipeline
            ),
            "13_sapf.py": StepConfig(
                name="sapf",
                description="SAPF audio generation for GNN models",
                module_path="13_sapf.py",
                output_subdir="sapf_processing_step",
                dependencies=["1_gnn.py"],
                performance_tracking=True,
                required=False  # SAPF can fail without stopping pipeline
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
        required_steps = ["1_gnn.py", "2_setup.py", "main.py"]
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
    """
    Update the global pipeline configuration with new values.
    Args:
        **kwargs: Configuration parameters to update
    Returns:
        Updated PipelineConfig instance
    """
    config = get_pipeline_config()
    
    # Update basic attributes
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Update step configurations if provided
    if 'steps' in kwargs:
        for step_name, step_data in kwargs['steps'].items():
            if step_name in config.steps:
                step_config = config.steps[step_name]
                for attr, value in step_data.items():
                    if hasattr(step_config, attr):
                        setattr(step_config, attr, value)
    
    return config

# STEP_METADATA constant for backward compatibility
STEP_METADATA = {
    "1_gnn.py": {
        "description": "GNN file discovery and parsing",
        "required": True,
        "dependencies": [],
        "output_subdir": "gnn_processing_step"
    },
    "2_setup.py": {
        "description": "Project setup and environment validation", 
        "required": True,
        "dependencies": [],
        "output_subdir": "setup_artifacts"
    },
    "3_tests.py": {
        "description": "Test suite execution",
        "required": False,
        "dependencies": ["2_setup.py"],
        "output_subdir": "test_reports"
    },
    "4_type_checker.py": {
        "description": "GNN syntax and type validation",
        "required": False,
        "dependencies": ["1_gnn.py"],
        "output_subdir": "type_check"
    },
    "5_export.py": {
        "description": "Multi-format export generation",
        "required": False,
        "dependencies": ["4_type_checker.py"],
        "output_subdir": "gnn_exports"
    },
    "6_visualization.py": {
        "description": "Graph visualization generation",
        "required": False,
        "dependencies": ["5_export.py"],
        "output_subdir": "visualization"
    },
    "7_mcp.py": {
        "description": "Model Context Protocol operations",
        "required": False,
        "dependencies": [],
        "output_subdir": "mcp_processing_step"
    },
    "8_ontology.py": {
        "description": "Ontology processing and validation",
        "required": False,
        "dependencies": ["5_export.py"],
        "output_subdir": "ontology_processing"
    },
    "9_render.py": {
        "description": "Code generation for simulation environments",
        "required": False,
        "dependencies": ["5_export.py"],
        "output_subdir": "gnn_rendered_simulators"
    },
    "10_execute.py": {
        "description": "Execute rendered simulators",
        "required": False,
        "dependencies": ["9_render.py"],
        "output_subdir": "execution_results"
    },
    "11_llm.py": {
        "description": "LLM-enhanced analysis and processing",
        "required": False,
        "dependencies": ["5_export.py"],
        "output_subdir": "llm_processing_step"
    },
    "12_website.py": {
        "description": "Static website generation",
        "required": False,
        "dependencies": ["6_visualization.py", "8_ontology.py"],
        "output_subdir": "website"
    },
    "13_sapf.py": {
        "description": "SAPF audio generation for GNN models",
        "required": False,
        "dependencies": ["1_gnn.py"],
        "output_subdir": "sapf_processing_step"
    },
    "main.py": {
        "description": "Main pipeline orchestrator",
        "required": True,
        "dependencies": [],
        "output_subdir": "pipeline_logs"
    }
} 