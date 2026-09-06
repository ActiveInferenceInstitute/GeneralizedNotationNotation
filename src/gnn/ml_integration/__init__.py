"""
ML Integration module for GNN Processing Pipeline.

This module provides machine learning model integration capabilities:
structural feature extraction from GNN files, scikit-learn classifier
training (Step 14), framework availability detection, and inference with
the trained classifier artifacts.
"""

from typing import Any

from .frameworks import check_ml_frameworks
from .inference import (
    InferenceError,
    load_classifier,
    predict_batch,
    predict_with_model,
)
from .processor import (
    COMPLEXITY_LABELS,
    COMPLEXITY_THRESHOLDS,
    NUMERIC_FEATURE_NAMES,
    SUMMARY_STATISTIC_KEYS,
    complexity_label,
    extract_gnn_features,
    feature_vector,
    process_ml_integration,
    summarize_features,
)

__version__ = "3.2.0"

FEATURES: dict[str, Any] = {
    "model_training": True,
    "model_inference": True,
    "pipeline_integration": True,
    "mcp_integration": True,
}


def get_module_info() -> dict[str, Any]:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "ml_integration",
        "version": __version__,
        "description": "Machine learning model training and evaluation",
        "features": FEATURES,
    }


__all__: list[str] = [
    "COMPLEXITY_LABELS",
    "COMPLEXITY_THRESHOLDS",
    "FEATURES",
    "InferenceError",
    "NUMERIC_FEATURE_NAMES",
    "SUMMARY_STATISTIC_KEYS",
    "__version__",
    "check_ml_frameworks",
    "complexity_label",
    "extract_gnn_features",
    "feature_vector",
    "get_module_info",
    "load_classifier",
    "predict_batch",
    "predict_with_model",
    "process_ml_integration",
    "summarize_features",
]
