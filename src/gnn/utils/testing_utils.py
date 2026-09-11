"""Earlier name; implementation moved to ``gnn/utils/testing/`` (S2-33 Step 1).

``gnn.utils.testing_utils`` is the old module path, kept importable for the
deprecation window (design §4.1 / §6 mechanics 2). Import from
``gnn.utils.testing`` instead: this facade re-exports the family's full public
surface — including the submodule-path-only constants ``PROJECT_ROOT``,
``SRC_DIR`` and ``TEST_DIR`` that ``tests/__init__.py`` depends on (design R7)
— plus the historical ``get_memory_usage`` pass-through re-export
(``tests/utils/test_shared_helpers.py`` pins that delegation identity).
``_PerformanceTracker`` stays private to ``gnn.utils.testing.perf`` (§4.3.4)
and is intentionally not re-exported here.
"""

import warnings

warnings.warn(
    "gnn.utils.testing_utils is the earlier name; import gnn.utils.testing instead",
    DeprecationWarning,
    stacklevel=2,
)
from gnn.utils.runtime_safety.resource_manager import (
    get_memory_usage,  # noqa: E402,F401
)
from gnn.utils.testing import *  # noqa: E402,F401,F403 — intentional re-export
from gnn.utils.testing import __all__ as _family_all

# The facade covers exactly the moved module's public names (§4.3.4): the
# family's 67 public names plus the historical get_memory_usage pass-through.
__all__ = [*_family_all, "get_memory_usage"]
