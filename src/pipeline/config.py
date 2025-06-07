"""
Pipeline Configuration Module

Centralizes all pipeline configuration including step definitions,
timeouts, argument mappings, and default settings.
"""

from typing import Dict, List, Any
from pathlib import Path

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

# Step Timeout Configuration (in seconds)
STEP_TIMEOUTS: Dict[str, int] = {
    "1_gnn.py": 120,
    "2_setup.py": 1200,  # 20 minutes - can be slow on first run or slow connections
    "3_tests.py": 300,   # 5 minutes - allow time for comprehensive testing
    "4_gnn_type_checker.py": 120,
    "5_export.py": 120,
    "6_visualization.py": 300,  # Can be slow for complex visualizations
    "7_mcp.py": 120,
    "8_ontology.py": 120,
    "9_render.py": 120,
    "10_execute.py": 300,  # May need time to execute rendered code
    "11_llm.py": None,  # Uses --llm-timeout argument
    "12_discopy.py": 120,
    "13_discopy_jax_eval.py": 300,
    "14_site.py": 120,
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