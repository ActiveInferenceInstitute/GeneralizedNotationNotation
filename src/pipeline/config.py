#!/usr/bin/env python3
"""
Pipeline configuration module.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

# Make PyYAML optional to avoid hard failures during import time
try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

# Pipeline configuration
STEP_METADATA = {
    "0_template": {"name": "Template", "description": "Pipeline template and initialization"},
    "1_setup": {"name": "Setup", "description": "Environment setup and dependency installation"},
    "2_tests": {"name": "Tests", "description": "Comprehensive test suite execution"},
    "3_gnn": {"name": "GNN Processing", "description": "GNN file discovery and parsing"},
    "4_model_registry": {"name": "Model Registry", "description": "Model registry management"},
    "5_type_checker": {"name": "Type Checker", "description": "GNN syntax validation"},
    "6_validation": {"name": "Validation", "description": "Advanced validation and consistency"},
    "7_export": {"name": "Export", "description": "Multi-format export"},
    "8_visualization": {"name": "Visualization", "description": "Graph and matrix visualization"},
    "9_advanced_viz": {"name": "Advanced Visualization", "description": "Advanced visualization"},
    "10_ontology": {"name": "Ontology", "description": "Active Inference Ontology processing"},
    "11_render": {"name": "Render", "description": "Code generation for simulation environments"},
    "12_execute": {"name": "Execute", "description": "Execute rendered simulation scripts"},
    "13_llm": {"name": "LLM", "description": "LLM-enhanced analysis"},
    "14_ml_integration": {"name": "ML Integration", "description": "Machine learning integration"},
    "15_audio": {"name": "Audio", "description": "Audio generation"},
    "16_analysis": {"name": "Analysis", "description": "Advanced analysis and statistics"},
    "17_integration": {"name": "Integration", "description": "System integration"},
    "18_security": {"name": "Security", "description": "Security validation"},
    "19_research": {"name": "Research", "description": "Research tools"},
    "20_website": {"name": "Website", "description": "Static HTML website generation"},
    "21_report": {"name": "Report", "description": "Comprehensive report generation"},
    "22_mcp": {"name": "MCP", "description": "Model Context Protocol processing"}
}

class StepConfig:
    """Configuration for a pipeline step."""
    
    def __init__(self, step_name: str, **kwargs):
        self.step_name = step_name
        self.enabled = kwargs.get('enabled', True)
        self.timeout = kwargs.get('timeout', 300)
        self.retries = kwargs.get('retries', 3)
        self.dependencies = kwargs.get('dependencies', [])
        self.parameters = kwargs.get('parameters', {})

class PipelineConfig:
    """Pipeline configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("pipeline_config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix in ('.yaml', '.yml'):
                        if _YAML_AVAILABLE:
                            return yaml.safe_load(f) or {}
                        else:
                            # Gracefully degrade: cannot parse YAML; return empty config
                            # Downstream code should use sensible defaults
                            return {}
                    else:
                        return json.load(f)
            except Exception:
                # Any parsing error should not crash pipeline; return empty config
                return {}
        return {}
    
    def get_step_config(self, step_name: str) -> StepConfig:
        """Get configuration for a specific step."""
        step_data = self.config.get(step_name, {})
        return StepConfig(step_name, **step_data)
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix in ('.yaml', '.yml') and _YAML_AVAILABLE:
                    yaml.dump(self.config, f)  # type: ignore
                else:
                    json.dump(self.config, f, indent=2)
        except Exception:
            # If saving fails (e.g., YAML not available), attempt JSON fallback
            try:
                json_path = self.config_path.with_suffix('.json')
                with open(json_path, 'w') as jf:
                    json.dump(self.config, jf, indent=2)
            except Exception:
                # Silently ignore saving issues to avoid breaking pipeline
                pass

def get_pipeline_config() -> dict:
    """Get pipeline configuration as a plain dict for compatibility with tests."""
    cfg = PipelineConfig()
    data = cfg.config if isinstance(cfg.config, dict) else {}
    # Ensure required keys exist for tests with sensible defaults
    if 'steps' not in data:
        data['steps'] = list(STEP_METADATA.keys())
    if 'timeout' not in data:
        data['timeout'] = 300
    if 'parallel' not in data:
        data['parallel'] = True
    return data

def get_pipeline_config_dict() -> Dict[str, Any]:
    """Get the pipeline configuration as a plain dict (compatibility helper)."""
    cfg = PipelineConfig()
    return cfg.config if isinstance(cfg.config, dict) else {}

def set_pipeline_config(config: PipelineConfig):
    """Set the pipeline configuration."""
    config.save_config()

def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get output directory for a specific script.
    
    Use a consistent numbered '<N_name>_output' directory for every step to
    keep the pipeline coherent and simple.
    """
    script_stem = Path(script_name).stem
    normalized = script_stem  # e.g., '7_export'

    # Map stems to standardized numbered output directories
    strict_mapping = {
        "0_template": base_output_dir / "0_template_output",
        "1_setup": base_output_dir / "1_setup_output",
        "2_tests": base_output_dir / "2_tests_output",
        "3_gnn": base_output_dir / "3_gnn_output",
        "4_model_registry": base_output_dir / "4_model_registry_output",
        "5_type_checker": base_output_dir / "5_type_checker_output",
        "6_validation": base_output_dir / "6_validation_output",
        "7_export": base_output_dir / "7_export_output",
        "8_visualization": base_output_dir / "8_visualization_output",
        "9_advanced_viz": base_output_dir / "9_advanced_viz_output",
        "10_ontology": base_output_dir / "10_ontology_output",
        "11_render": base_output_dir / "11_render_output",
        "12_execute": base_output_dir / "12_execute_output",
        "13_llm": base_output_dir / "13_llm_output",
        "14_ml_integration": base_output_dir / "14_ml_integration_output",
        "15_audio": base_output_dir / "15_audio_output",
        "16_analysis": base_output_dir / "16_analysis_output",
        "17_integration": base_output_dir / "17_integration_output",
        "18_security": base_output_dir / "18_security_output",
        "19_research": base_output_dir / "19_research_output",
        "20_website": base_output_dir / "20_website_output",
        "21_report": base_output_dir / "21_report_output",
        "22_mcp": base_output_dir / "22_mcp_output",
    }

    # Accept '.py' suffix keys as well
    if script_name.endswith('.py'):
        normalized = script_name[:-3]

    # Exact matches
    if script_name in strict_mapping:
        return strict_mapping[script_name]
    if normalized in strict_mapping:
        return strict_mapping[normalized]

    # Default fallback
    return base_output_dir / f"{script_stem}_output"