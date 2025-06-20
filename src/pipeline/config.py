"""
Pipeline Configuration Module

Centralizes all pipeline configuration including step definitions,
timeouts, argument mappings, and default settings.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Environment-based configuration overrides
def get_env_override(key: str, default: Any, type_func=str) -> Any:
    """Get configuration value from environment with fallback to default."""
    env_value = os.getenv(f"GNN_PIPELINE_{key.upper()}")
    if env_value is not None:
        try:
            return type_func(env_value)
        except (ValueError, TypeError):
            pass
    return default

# Dynamic path resolution
def get_project_root() -> Path:
    """Get project root directory dynamically."""
    current_file = Path(__file__).resolve()
    # Assuming config.py is in src/pipeline/, project root is 2 levels up
    return current_file.parent.parent.parent

# Default paths configuration
PROJECT_ROOT = get_project_root()
DEFAULT_PATHS = {
    "target_dir": PROJECT_ROOT / "src" / "gnn" / "examples",
    "output_dir": PROJECT_ROOT / "output",
    "logs_dir": PROJECT_ROOT / "logs",
    "temp_dir": PROJECT_ROOT / "temp",
    "venv_dir": PROJECT_ROOT / "venv"
}

# Step dependencies (steps that must succeed before others can run)
STEP_DEPENDENCIES = {
    "3_tests.py": ["2_setup.py"],
    "4_gnn_type_checker.py": ["1_gnn.py"],
    "5_export.py": ["1_gnn.py"],
    "6_visualization.py": ["5_export.py"],
    "9_render.py": ["5_export.py"],
    "10_execute.py": ["9_render.py"],
    "11_llm.py": ["5_export.py"],
    "12_discopy.py": ["5_export.py"],
    "13_discopy_jax_eval.py": ["12_discopy.py"],
    "14_site.py": ["1_gnn.py"]
}

# Enhanced step configuration with metadata
STEP_METADATA = {
    "1_gnn.py": {
        "description": "GNN file discovery and basic parsing",
        "category": "input_processing",
        "required_for": ["4_gnn_type_checker.py", "5_export.py"]
    },
    "2_setup.py": {
        "description": "Environment setup and dependency installation", 
        "category": "infrastructure",
        "required_for": ["3_tests.py"]
    },
    "3_tests.py": {
        "description": "Test suite execution",
        "category": "validation", 
        "optional": get_env_override("skip_tests", False, bool)
    },
    "4_gnn_type_checker.py": {
        "description": "GNN syntax and type validation",
        "category": "validation"
    },
    "5_export.py": {
        "description": "Multi-format export generation",
        "category": "transformation",
        "required_for": ["6_visualization.py", "9_render.py", "11_llm.py", "12_discopy.py"]
    },
    "6_visualization.py": {
        "description": "Graph visualization generation",
        "category": "output"
    },
    "7_mcp.py": {
        "description": "Model Context Protocol operations",
        "category": "integration"
    },
    "8_ontology.py": {
        "description": "Ontology processing and validation",
        "category": "semantic"
    },
    "9_render.py": {
        "description": "Code generation for simulation environments",
        "category": "transformation",
        "required_for": ["10_execute.py"]
    },
    "10_execute.py": {
        "description": "Execute rendered simulators",
        "category": "execution"
    },
    "11_llm.py": {
        "description": "LLM-enhanced analysis and processing",
        "category": "ai_enhancement"
    },
    "12_discopy.py": {
        "description": "DisCoPy categorical diagram translation",
        "category": "mathematical",
        "required_for": ["13_discopy_jax_eval.py"]
    },
    "13_discopy_jax_eval.py": {
        "description": "JAX-based evaluation of DisCoPy diagrams",
        "category": "mathematical",
        "experimental": True
    },
    "14_site.py": {
        "description": "Static site generation for documentation",
        "category": "output"
    }
}

# Pipeline Step Configuration
# This dictionary controls which pipeline steps are enabled by default.
# Steps are identified by their script basename.
PIPELINE_STEP_CONFIGURATION: Dict[str, bool] = {
    "1_gnn.py": True,
    "2_setup.py": True,
    "3_tests.py": True, 
    "4_gnn_type_checker.py": True,
    "5_export.py": True,
    "6_visualization.py": True,
    "7_mcp.py": True,
    "8_ontology.py": True,
    "9_render.py": True,
    "10_execute.py": True,
    "11_llm.py": True,
    "12_discopy.py": True,
    "13_discopy_jax_eval.py": False,  # Disabled by default - experimental
    "14_site.py": True,
}

# Enhanced timeout configuration with environment overrides
STEP_TIMEOUTS: Dict[str, int] = {
    "1_gnn.py": get_env_override("timeout_1_gnn", 120, int),
    "2_setup.py": get_env_override("timeout_2_setup", 1200, int),
    "3_tests.py": get_env_override("timeout_3_tests", 300, int),
    "4_gnn_type_checker.py": get_env_override("timeout_4_type_checker", 120, int),
    "5_export.py": get_env_override("timeout_5_export", 120, int),
    "6_visualization.py": get_env_override("timeout_6_visualization", 300, int),
    "7_mcp.py": get_env_override("timeout_7_mcp", 120, int),
    "8_ontology.py": get_env_override("timeout_8_ontology", 120, int),
    "9_render.py": get_env_override("timeout_9_render", 120, int),
    "10_execute.py": get_env_override("timeout_10_execute", 300, int),
    "11_llm.py": None,  # Uses --llm-timeout argument
    "12_discopy.py": get_env_override("timeout_12_discopy", 120, int),
    "13_discopy_jax_eval.py": get_env_override("timeout_13_jax", 300, int),
    "14_site.py": get_env_override("timeout_14_site", 120, int),
}

# Critical Steps (pipeline halts if these fail)
CRITICAL_STEPS = {"2_setup.py"}

# Argument Properties Configuration
ARG_PROPERTIES = {
    'target_dir': {'flag': '--target-dir', 'type': 'value'},
    'output_dir': {'flag': '--output-dir', 'type': 'value'},
    'recursive': {'flag': '--recursive', 'type': 'bool_optional'},
    'verbose': {'flag': '--verbose', 'type': 'bool_optional'},
    'strict': {'flag': '--strict', 'type': 'store_true'},
    'estimate_resources': {'flag': '--estimate-resources', 'type': 'bool_optional'},
    'ontology_terms_file': {'flag': '--ontology-terms-file', 'type': 'value'},
    'llm_tasks': {'flag': '--llm-tasks', 'type': 'value'},
    'llm_timeout': {'flag': '--llm-timeout', 'type': 'value'},
    'site_html_filename': {'flag': '--site-html-filename', 'type': 'value'},
    'discopy_gnn_input_dir': {'flag': '--discopy-gnn-input-dir', 'type': 'value'},
    'discopy_jax_gnn_input_dir': {'flag': '--discopy-jax-gnn-input-dir', 'type': 'value'},
    'discopy_jax_seed': {'flag': '--discopy-jax-seed', 'type': 'value'},
    'recreate_venv': {'flag': '--recreate-venv', 'type': 'store_true'},
    'dev': {'flag': '--dev', 'type': 'store_true'},
}

# Script Argument Support Matrix
# Defines which arguments are passed to each pipeline script
SCRIPT_ARG_SUPPORT = {
    "1_gnn.py": ["target_dir", "output_dir", "recursive", "verbose"],
    "2_setup.py": ["target_dir", "output_dir", "verbose", "recreate_venv", "dev"],
    "3_tests.py": ["target_dir", "output_dir", "verbose"],
    "4_gnn_type_checker.py": ["target_dir", "output_dir", "recursive", "verbose", "strict", "estimate_resources"],
    "5_export.py": ["target_dir", "output_dir", "recursive", "verbose"],
    "6_visualization.py": ["target_dir", "output_dir", "recursive", "verbose"],
    "7_mcp.py": ["target_dir", "output_dir", "verbose"],
    "8_ontology.py": ["target_dir", "output_dir", "recursive", "verbose", "ontology_terms_file"],
    "9_render.py": ["output_dir", "recursive", "verbose"],
    "10_execute.py": ["target_dir", "output_dir", "recursive", "verbose"],
    "11_llm.py": ["target_dir", "output_dir", "recursive", "verbose", "llm_tasks", "llm_timeout"],
    "12_discopy.py": ["target_dir", "output_dir", "verbose", "discopy_gnn_input_dir"],
    "13_discopy_jax_eval.py": ["target_dir", "output_dir", "verbose", "discopy_jax_gnn_input_dir", "discopy_jax_seed"],
    "14_site.py": ["target_dir", "output_dir", "verbose", "site_html_filename"],
}

# Output Directory Mapping
# Maps scripts to their specific output subdirectories
OUTPUT_DIR_MAPPING = {
    "4_gnn_type_checker.py": "gnn_type_check",
    "5_export.py": "gnn_exports",
    "6_visualization.py": "visualization",
    "7_mcp.py": "mcp_processing_step",
    "8_ontology.py": "ontology_processing",
    "11_llm.py": "llm_processing_step",
    "12_discopy.py": "discopy_gnn",
    # Scripts not listed use the main output directory
}

def get_step_timeout(script_name: str, args) -> int:
    """Get timeout for a specific step, considering special cases."""
    if script_name == "11_llm.py":
        return getattr(args, 'llm_timeout', 120)
    return STEP_TIMEOUTS.get(script_name, 120)

def is_critical_step(script_name: str) -> bool:
    """Check if a step is critical (pipeline halts on failure)."""
    return script_name in CRITICAL_STEPS

def get_output_dir_for_script(script_name: str, base_output_dir: Path) -> Path:
    """Get the appropriate output directory for a script."""
    if script_name in OUTPUT_DIR_MAPPING:
        return base_output_dir / OUTPUT_DIR_MAPPING[script_name]
    return base_output_dir 