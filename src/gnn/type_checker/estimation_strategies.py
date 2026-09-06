"""
Pure computational strategies for estimating GNN resource requirements.
"""

# Re-export from the estimation subpackage.
from typing import Any

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

__all__: list[Any] = [
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
