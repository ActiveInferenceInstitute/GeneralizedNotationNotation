"""
Test Helpers Module

Provides reusable helper functions and utilities for test execution.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Import helper modules
try:
    from .render_recovery import RenderRecoveryHelper
except ImportError:
    RenderRecoveryHelper = None


def get_test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent.parent / "test_data"


def get_sample_gnn_model() -> Path:
    """Get path to sample GNN model file."""
    return get_test_data_dir() / "sample_gnn_model.md"


def load_sample_gnn_spec() -> Dict[str, Any]:
    """Load and parse the sample GNN specification."""
    sample_file = get_sample_gnn_model()
    if not sample_file.exists():
        return {
            "name": "sample_model",
            "states": ["s1", "s2"],
            "observations": ["o1"],
            "parameters": {}
        }
    
    # Basic parsing of GNN markdown
    content = sample_file.read_text()
    spec = {"name": "sample_model", "raw_content": content}
    
    # Extract model name if present
    for line in content.splitlines():
        if line.startswith("## ModelName"):
            # Next non-empty line is the name
            idx = content.find(line) + len(line)
            remaining = content[idx:].strip()
            if remaining:
                spec["name"] = remaining.split("\n")[0].strip()
            break
    
    return spec


__all__ = [
    "RenderRecoveryHelper",
    "get_test_data_dir",
    "get_sample_gnn_model",
    "load_sample_gnn_spec",
]
