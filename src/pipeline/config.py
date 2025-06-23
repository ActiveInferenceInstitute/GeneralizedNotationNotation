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
    base_target_dir: Path = Path("src/gnn/examples")
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
                description="Test suite execution",
                module_path="3_tests.py",
                output_subdir="test_reports",
                dependencies=["2_setup.py"],
                timeout=600,
                required=False  # Tests can fail without stopping pipeline
            ),
            "4_gnn_type_checker.py": StepConfig(
                name="type_checking",
                description="GNN syntax and type validation",
                module_path="4_gnn_type_checker.py",
                output_subdir="gnn_type_check",
                dependencies=["1_gnn.py"],
                performance_tracking=True,
                required=False  # Type checking can fail without stopping pipeline
            ),
            "5_export.py": StepConfig(
                name="export",
                description="Multi-format export generation",
                module_path="5_export.py",
                output_subdir="gnn_exports",
                dependencies=["4_gnn_type_checker.py"],
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
            "12_discopy.py": StepConfig(
                name="discopy",
                description="DisCoPy categorical diagram translation",
                module_path="12_discopy.py",
                output_subdir="discopy_gnn",
                dependencies=["5_export.py"],
                required=False  # DisCoPy can fail without stopping pipeline
            ),
            "13_discopy_jax_eval.py": StepConfig(
                name="discopy_jax_eval",
                description="JAX-based evaluation of DisCoPy diagrams",
                module_path="13_discopy_jax_eval.py",
                output_subdir="discopy_jax_eval",
                dependencies=["12_discopy.py"],
                required=False  # JAX eval can fail without stopping pipeline
            ),
            "14_site.py": StepConfig(
                name="site",
                description="Static site generation",
                module_path="14_site.py",
                output_subdir="site",
                dependencies=["6_visualization.py", "8_ontology.py"],
                required=False  # Site generation can fail without stopping pipeline
            ),
            "15_sapf.py": StepConfig(
                name="sapf",
                description="SAPF audio generation for GNN models",
                module_path="15_sapf.py",
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

# Convenience functions for backward compatibility
def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get output directory for a script (backward compatibility)."""
    config = get_pipeline_config()
    return config.get_output_dir_for_step(script_name, base_output_dir)

# Step metadata for backward compatibility
STEP_METADATA = {
    step_name: {
        "name": step_config.name,
        "description": step_config.description,
        "dependencies": step_config.dependencies,
        "performance_tracking": step_config.performance_tracking
    }
    for step_name, step_config in get_pipeline_config().steps.items()
} 