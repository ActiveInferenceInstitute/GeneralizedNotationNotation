"""
Visualization compatibility shim — shared matplotlib/numpy/seaborn imports.

Provides plt, np, sns, and MATPLOTLIB_AVAILABLE for visualization modules
without creating an upward dependency on analysis.viz_base (step 16).
Both visualization (step 8) and analysis (step 16) import from here.
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
    _mpl.use('Agg')
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
