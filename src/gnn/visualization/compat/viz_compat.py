"""
Shared matplotlib/numpy/seaborn imports for visualization and analysis.

Both visualization (step 8) and analysis (step 16) import from this module
directly; it is the single implementation home for the shared plotting
imports. The seaborn module itself resolves lazily via `sns`/`get_sns()`
on first access.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

MATPLOTLIB_AVAILABLE = False
plt: Any = None
np: Any = None

try:
    import numpy as _np

    np = _np
except ImportError:
    logger.debug("numpy not available")

try:
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    plt = _plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    logger.debug("matplotlib not available")

_sns_loaded: bool = False
_sns_module: Any = None


def get_sns() -> Any:
    """Return the seaborn module, importing it on first call. Returns None
    when seaborn is not installed. Importing this module never imports seaborn."""
    global _sns_loaded, _sns_module
    if not _sns_loaded:
        try:
            import seaborn as _sns

            _sns_module = _sns
        except ImportError as e:
            logger.debug("seaborn not available: %s", e)
        _sns_loaded = True
    return _sns_module


def __getattr__(name: str) -> Any:
    if name == "sns":
        return get_sns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def viz_var_type(var_info: "dict") -> str:
    """Extract the variable type from a parsed variable dict.

    Checks ``var_type``, ``type``, and ``node_type`` keys in order,
    returning ``"unknown"`` when none are present.
    """
    var_info_obj: object = var_info
    if not isinstance(var_info_obj, dict):
        return "unknown"
    var_type = var_info_obj.get(
        "var_type", var_info_obj.get("type", var_info_obj.get("node_type", "unknown"))
    )
    return str(var_type)
