"""
Resource estimation subsystem.

Analyzes GNN models and estimates computational resources needed for
memory usage, inference time, and storage requirements.
"""

from .estimator import GNNResourceEstimator
from .strategies import (
    calculate_complexity,
    detailed_memory_breakdown,
    estimate_batched_inference,
    estimate_flops,
    estimate_inference,
    estimate_inference_time,
    estimate_matrix_operation_costs,
    estimate_memory,
    estimate_model_overhead,
    estimate_storage,
)

__all__ = [
    "GNNResourceEstimator",
    "calculate_complexity",
    "detailed_memory_breakdown",
    "estimate_batched_inference",
    "estimate_flops",
    "estimate_inference",
    "estimate_inference_time",
    "estimate_matrix_operation_costs",
    "estimate_memory",
    "estimate_model_overhead",
    "estimate_storage",
]
