"""Public entry facade for argument handling (S2-33 Step 2).

``gnn.utils.argument_utils`` remains the single public entry point for the
argument family, whose implementation now lives in the
``gnn/utils/arguments/`` concern package (design §3.2). Import from
``gnn.utils.arguments`` instead: this facade re-exports the family's full
public surface, plus the historical ``logger`` re-export.
"""

import warnings

warnings.warn(
    "gnn.utils.argument_utils is the earlier name; import gnn.utils.arguments instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.arguments import *  # noqa: E402,F401,F403 — intentional re-export
from gnn.utils.arguments import __all__ as _family_all
from gnn.utils.arguments.arg_parsing import logger  # noqa: E402,F401

__all__ = [*_family_all, "logger"]
