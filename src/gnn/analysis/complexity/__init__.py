"""Static complexity estimation for parsed GNN models.

Pure-stdlib subpackage (plus the existing GNN parser / type-checker /
render-contract modules). Zero framework imports and zero executor
imports: importable without the execute stack.

Public API:
    estimate_model_complexity(model_or_path) -> dict
        Build the pinned ``gnn.complexity_estimate/v1`` receipt for a
        ``GNNInternalRepresentation`` or a spec-file path.
    to_json_text(receipt) -> str
        Stable sorted-key JSON serialization (the CLI lane consumes this).
"""

from .estimator import (
    ESTIMATOR_VERSION,
    RECEIPT_TYPE,
    UNBOUNDED_HORIZON,
    estimate_model_complexity,
    to_json_text,
)

__all__: list[str] = [
    "ESTIMATOR_VERSION",
    "RECEIPT_TYPE",
    "UNBOUNDED_HORIZON",
    "estimate_model_complexity",
    "to_json_text",
]
