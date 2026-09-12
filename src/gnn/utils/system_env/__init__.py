"""System-environment concern package (S2-33 Step 7, family 1/3): canonical
home of the "what machine is this / which Python runs it / how does plotting
initialize" concerns — system information gathering, virtual-environment
Python discovery, and matplotlib backend configuration — code that used to
live in the ``gnn/utils`` top-level grab-bag.

Re-exports the family's public names as real objects (not lazy — design
§4.3.1), so intra-family reads use ``from gnn.utils.system_env import
get_venv_python``. Import-weight note (I1): this package is NOT imported by
``import gnn.utils`` — the top-level facade stays lazy through its PEP 562
map (guarded by ``tests/tests/test_light_import.py``). Importing this
package eagerly imports every leaf, which is the same cost the old
``import gnn.utils.system_utils`` paid; the family's optional heavy
dependencies stay guarded at leaf scope (``system_utils`` probes psutil in a
try/except and publishes ``PSUTIL_AVAILABLE``, ``matplotlib_setup`` imports
matplotlib only inside ``apply_env_backend_if_set``), so no leaf import can
fail or unconditionally pull a heavy dependency.

Leaf inventory:
- system_utils: ``get_system_info`` system-info probe (+ ``PSUTIL_AVAILABLE``)
- venv_utils: ``get_venv_python`` virtual-environment Python/site-packages discovery
- matplotlib_setup: ``apply_env_backend_if_set`` MPLBACKEND application before pyplot import

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/system_utils.py`` etc.) are deprecation
facades over this package.
"""

from gnn.utils.system_env.matplotlib_setup import apply_env_backend_if_set
from gnn.utils.system_env.system_utils import PSUTIL_AVAILABLE, get_system_info
from gnn.utils.system_env.venv_utils import get_venv_python

__all__ = [
    "PSUTIL_AVAILABLE",
    "apply_env_backend_if_set",
    "get_system_info",
    "get_venv_python",
]