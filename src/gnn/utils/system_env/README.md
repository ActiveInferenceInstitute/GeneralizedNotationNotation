# `gnn/utils/system_env/` — README

Concern-package home (S2-33 Step 7, family 1/3; design:
[`docs/development/utils_split_design.md`](../../../../docs/development/utils_split_design.md)).
The old `gnn/utils/<module>.py` paths are DeprecationWarning facades — import
from this package. Layout, invariants (frozen `_EXPORT_MAP` values, guarded
optional-dependency probes, eager family re-exports per §4.3.1), and gating
tests are listed in [`AGENTS.md`](AGENTS.md).

## Leaves

- [`system_utils.py`](system_utils.py): `get_system_info` system-info probe
  (plus the guarded `PSUTIL_AVAILABLE` psutil probe)
- [`venv_utils.py`](venv_utils.py): `get_venv_python` virtual-environment
  Python/site-packages discovery
- [`matplotlib_setup.py`](matplotlib_setup.py): `apply_env_backend_if_set`
  applies `MPLBACKEND` before `matplotlib.pyplot` initializes GUI state