"""
Pure computational strategies for estimating GNN resource requirements.

NOTE: This module is maintained for backward compatibility.
The actual implementation has been moved to the `estimation` subpackage.
"""

# Re-export from the estimation subpackage
from .estimation.strategies import (
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
