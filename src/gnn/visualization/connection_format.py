"""Connection-record format normalization shared by the visualization layers.

The visualization data model carries connections either as scalar
``{"source": .., "target": ..}`` dicts or as the multi-variable
``{"source_variables": [...], "target_variables": [...]}`` format. This
module is the single home for normalizing between them; it lives in the
base ``visualization`` package so ``advanced_visualization`` (and tests)
import downstream, never the reverse.
"""

from typing import Any, Dict, List, Tuple


def normalize_connection_format(conn_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize connection format to handle both old and new formats."""
    if "source_variables" in conn_info and "target_variables" in conn_info:
        return conn_info
    elif "source" in conn_info and "target" in conn_info:
        return {
            "source_variables": [conn_info["source"]],
            "target_variables": [conn_info["target"]],
            **{k: v for k, v in conn_info.items() if k not in ["source", "target"]},
        }
    else:
        return conn_info


def conn_endpoints(conn_info: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    """Return ``(source_variables, target_variables)`` for a connection dict.

    Normalizes scalar ``{"source": .., "target": ..}`` format first, so
    callers never repeat the normalize-then-extract dance.
    """
    normalized = normalize_connection_format(conn_info)
    sources = normalized.get("source_variables", [])
    targets = normalized.get("target_variables", [])
    return sources, targets
