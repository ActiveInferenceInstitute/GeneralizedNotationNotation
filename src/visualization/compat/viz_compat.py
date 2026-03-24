"""
Shared matplotlib/numpy/seaborn imports for visualization and analysis.

Both visualization (step 8) and analysis (step 16) import from the package-root
`visualization._viz_compat` shim, which re-exports this module.
"""

import logging

logger = logging.getLogger(__name__)

MATPLOTLIB_AVAILABLE = False
plt = None
np = None
sns = None

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

try:
    import seaborn as _sns

    sns = _sns
except ImportError as e:
    logger.debug("seaborn not available: %s", e)
